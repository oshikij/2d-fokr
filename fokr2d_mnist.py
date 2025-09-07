"""
fokr2d_mnist.py  

Functional Output Kernel Regression (FOKR) for 2D images (MNIST demo).

- Input X: one-hot label vector (dim=10)
- Output Y: image (28x28)

Model:
  Y_hat(X, t) = sum_{i=1..d} beta_i(X, z) * k(t, s_i) + mu(t)

  * {s_i} are 2D RBF kernel centers on an equally spaced grid over image coordinates
  * k(t, s) = A * exp(-||t - s||^2 / (2 * l^2))
  * beta(X, z) is predicted from one-hot labels X; optional latent z with reparameterization
  * mu(t) is a learnable intercept per pixel (optional)

This file provides:
  - FOKR2D module with beta_arch={'mlp','deconv'} and reparam options
  - Training utilities (MNIST and synthetic)
  - Self-tests (synthetic) without internet

References:
  Iwayama et al., "Functional Output Regression for Machine Learning in Materials Science",
  J. Chem. Inf. Model. 2022.

"""

from __future__ import annotations

import argparse
import math
import os
import random
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

# torchvision is only needed for the MNIST example
try:
    from torchvision import datasets, transforms, utils as vutils
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels to one-hot vectors."""
    out = torch.zeros(labels.shape[0], num_classes, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def make_coord_grid(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return NxN coordinates normalized to [0, 1] x [0, 1], shape (size*size, 2)."""
    lin = torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1)  # (H,W,2)
    return coords.reshape(size * size, 2)


def make_center_grid(grid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return dx2 centers normalized to [0,1], equally spaced on a grid."""
    if grid <= 1:
        return torch.full((1, 2), 0.5, device=device, dtype=dtype)
    lin = torch.linspace(0.0, 1.0, steps=grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    centers = torch.stack([yy, xx], dim=-1)  # (grid,grid,2)
    return centers.reshape(grid * grid, 2)


# -------------------------------
# RBF Kernel Features
# -------------------------------

class RBFGrid2D(nn.Module):
    """
    Precomputes the kernel dictionary Phi of size (m, d) for an image size (N x N)
    and a grid of d=GxG Gaussian RBF centers.
    """
    def __init__(self, img_size: int, grid: int, lengthscale: float = 0.15, amplitude: float = 1.0, learn_hyper: bool = False):
        super().__init__()
        self.img_size = img_size
        self.grid = grid
        self.m = img_size * img_size
        self.d = grid * grid

        # Register positive parameters via softplus if learnable
        if learn_hyper:
            self._raw_l = nn.Parameter(torch.as_tensor(math.log(math.exp(lengthscale) - 1.0), dtype=torch.float32))
            self._raw_A = nn.Parameter(torch.as_tensor(math.log(math.exp(amplitude) - 1.0), dtype=torch.float32))
        else:
            self.register_buffer("_raw_l", torch.tensor(math.log(math.exp(lengthscale) - 1.0), dtype=torch.float32))
            self.register_buffer("_raw_A", torch.tensor(math.log(math.exp(amplitude) - 1.0), dtype=torch.float32))
        self._learn_hyper = learn_hyper

        # Build coordinate grids
        coords = make_coord_grid(img_size, device=torch.device("cpu"), dtype=torch.float32)   # (m, 2)
        centers = make_center_grid(grid, device=torch.device("cpu"), dtype=torch.float32)     # (d, 2)
        self.register_buffer("coords", coords)    # (m,2)
        self.register_buffer("centers", centers)  # (d,2)

        # Precompute Phi at init so it can be reused quickly
        Phi = self._compute_phi(coords, centers)
        self.register_buffer("Phi", Phi)  # (m, d)

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-8

    @property
    def lengthscale(self) -> torch.Tensor:
        return self._softplus(self._raw_l)

    @property
    def amplitude(self) -> torch.Tensor:
        return self._softplus(self._raw_A)

    def _compute_phi(self, coords: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        # coords: (m,2), centers: (d,2)
        t = coords[:, None, :]           # (m,1,2)
        s = centers[None, :, :]          # (1,d,2)
        sq = (t - s).pow(2).sum(dim=-1)  # (m,d)
        l = self.lengthscale
        A = self.amplitude
        Phi = A * torch.exp(-sq / (2.0 * l * l))
        return Phi  # (m,d)

    def refresh(self):
        """Recompute Phi (useful if hyperparameters become learnable)."""
        with torch.no_grad():
            self.Phi.copy_(self._compute_phi(self.coords, self.centers))

    def forward(self) -> torch.Tensor:
        """
        Returns Phi (m,d). If hyperparameters are learnable, recompute on the fly.
        """
        if self._learn_hyper:
            return self._compute_phi(self.coords, self.centers)
        return self.Phi


# -------------------------------
# Beta networks
# -------------------------------

class BetaNetDeconv(nn.Module):
    """
    Deconvolutional decoder producing a (grid x grid) beta map.

    Strategy:
      - Project dense input to a small spatial tensor (C0 x base x base) through FC
      - [Deconv(k=3,s=1,p=1) -> BN -> LeakyReLU] x N times (size unchanged)
      - Final Deconv(k=4,s=2,p=1) to upsample 2x, output 1 channel (no BN, no activation)
    """
    def __init__(self, in_dim: int, grid: int, C0: int = 128, Cmid: int = 64, N: int = 1):
        super().__init__()
        assert grid % 2 == 0 and grid >= 14, "grid が偶数(>=14)を想定(例:14,28,56)"
        base = grid // 2
        self.grid = grid
        self.base = base

        # 1) ベクトル→(C0, base, base)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, C0 * base * base),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2) サイズ据え置きの Deconv ブロック（N回）
        blocks = []
        in_ch = C0
        for _ in range(N):
            blocks += [
                nn.ConvTranspose2d(in_ch, Cmid, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(Cmid),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = Cmid
        self.body = nn.Sequential(*blocks) if blocks else nn.Identity()

        # 3) 最終アップサンプル（厳密 2倍）＋線形1ch出力
        self.head = nn.ConvTranspose2d(in_ch, 1, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.fc(x).view(B, -1, self.base, self.base)  # (B, C0, base, base)
        h = self.body(h)                                  # (B, Cmid or C0, base, base)
        h = self.head(h)                                  # (B, 1, grid, grid)  ← 最後でアップ
        return h.view(B, self.grid * self.grid)           # (B, d)


# -------------------------------
# Reparameterization block
# -------------------------------

class LatentReparam(nn.Module):
    """
    Produce Gaussian latent z ~ N(mu(X), diag(sigma^2(X))) via reparameterization.
    If use_prior=True during eval, z=0 (or mu), but for training we sample.
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, sample: bool = True):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x).clamp(min=-10.0, max=10.0)  # avoid extreme variances
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        # KL divergence to standard normal
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return z, kl


# -------------------------------
# FOKR 2D Model (with options)
# -------------------------------

class FOKR2D(nn.Module):
    """
    Functional Output Kernel Regression for 2D images (deconv-only BetaNet).

    y_hat(X, t) = sum_i beta_i([X,z]) * k(t, s_i) + mu(t)
    where beta is produced by a deconvolutional decoder from [X, z] (paper-style).
    """
    def __init__(
        self,
        in_dim: int = 10,
        img_size: int = 28,
        grid: int = 14,
        lengthscale: float = 0.2,
        amplitude: float = 1.0,
        learn_hyper: bool = False,
        learn_mu: bool = True,
        latent_dim: int = 8,
        kl_weight: float = 1e-4,
        mu_tv_weight: float = 1e-4,
        deconv_repeats: int = 1,
        deconv_C0: int = 128,
        deconv_Cmid: int = 64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.img_size = img_size
        self.grid = grid
        self.m = img_size * img_size
        self.d = grid * grid

        self.kl_weight = kl_weight
        self.mu_tv_weight = mu_tv_weight

        # RBF dictionary
        self.features = RBFGrid2D(img_size=img_size, grid=grid,
                                  lengthscale=lengthscale, amplitude=amplitude,
                                  learn_hyper=learn_hyper)

        # latent (reparam)
        self.latent_dim = int(latent_dim)
        if self.latent_dim > 0:
            self.reparam = LatentReparam(in_dim=in_dim, latent_dim=self.latent_dim)
            beta_in = in_dim + self.latent_dim
        else:
            self.reparam = None
            beta_in = in_dim

        # Beta network 
        self.betanet = BetaNetDeconv(
            in_dim=beta_in, grid=grid, C0=deconv_C0, Cmid=deconv_Cmid, N=deconv_repeats
        )

        # mu(t)
        if learn_mu:
            self.mu = nn.Parameter(torch.zeros(self.m, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(self.m, dtype=torch.float32))
            self.mu_tv_weight = 0.0  # disable regularization if mu is fixed

    # ---- helpers

    def _mu_tv_l2(self) -> torch.Tensor:
        # TV-L2 (smoothness) regularization for mu
        if not isinstance(self.mu, torch.Tensor) or self.mu_tv_weight <= 0:
            return torch.tensor(0.0, device=self.mu.device if isinstance(self.mu, torch.Tensor) else "cpu")
        H = W = self.img_size
        mu_img = self.mu.view(1, 1, H, W)
        dy = mu_img[:, :, 1:, :] - mu_img[:, :, :-1, :]
        dx = mu_img[:, :, :, 1:] - mu_img[:, :, :, :-1]
        tv = dy.pow(2).mean() + dx.pow(2).mean()
        return self.mu_tv_weight * tv

    # ---- core

    def forward(self, x: torch.Tensor, *, sample_latent: bool = True, return_kl: bool = False):
        """
        x: (B, in_dim)
        returns y: (B,1,H,W) and optionally KL term.
        """
        B = x.size(0)
        kl = x.new_tensor(0.0)

        if self.reparam is not None:
            sample_flag = sample_latent
            z, kl_val = self.reparam(x, sample=sample_flag)
            #z, kl_val = self.reparam(x, sample=sample_latent and self.training)
            x_in = torch.cat([x, z], dim=1)
            kl = kl_val
        else:
            x_in = x

        beta = self.betanet(x_in)                  # (B,d)
        Phi = self.features().to(beta.device)      # (m,d)
        # y_flat = beta @ Phi.T + mu  -> use addmm for speed
        y_flat = torch.addmm(self.mu, beta, Phi.t())  # (B,m)
        y = y_flat.view(B, 1, self.img_size, self.img_size)

        if return_kl:
            return y, kl
        return y

    def loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.reparam is not None:
            y_pred, kl = self.forward(x, sample_latent=True, return_kl=True)
            base = F.mse_loss(y_pred, y_true)
            return base + self.kl_weight * kl + self._mu_tv_l2()
        else:
            y_pred = self.forward(x, sample_latent=False, return_kl=False)
            base = F.mse_loss(y_pred, y_true)
            return base + self._mu_tv_l2()

    # convenience: expose beta for visualization
    @torch.no_grad()
    def predict_beta_map(self, x: torch.Tensor, sample_latent: bool = False) -> torch.Tensor:
        """
        Return beta maps as (B, 1, grid, grid) without building y.
        """
        if self.reparam is not None and sample_latent:
            z, _ = self.reparam(x, sample=True)
            x_in = torch.cat([x, z], dim=1)
        elif self.reparam is not None:
            z, _ = self.reparam(x, sample=False)
            x_in = torch.cat([x, z], dim=1)
        else:
            x_in = x
        beta = self.betanet(x_in)                  # (B,d)
        return beta.view(-1, 1, self.grid, self.grid)


# -------------------------------
# Training / Evaluation
# -------------------------------

# -------------------------------
# Data
# -------------------------------

def get_mnist_loaders(batch_size: int = 128, num_workers: int = 2, pin_memory: bool = True):
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is not available. Cannot build MNIST loaders.")
    transform = transforms.ToTensor()  # values in [0,1]
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


# -------------------------------
# Training / Eval
# -------------------------------

@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grid: int = 14
    latent_dim: int = 8
    kl_weight: float = 1e-4
    mu_tv_weight: float = 1e-4
    deconv_repeats: int = 1
    deconv_C0: int = 128
    deconv_Cmid: int = 64
    lengthscale: float = 0.2
    amplitude: float = 1.0
    learn_mu: bool = True
    learn_hyper: bool = False
    amp: bool = False
    seed: int = 42
    save_dir: str = "./runs/fokr2d_mnist"


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_onehot = one_hot(labels.to(device), num_classes=10)
            preds = model(labels_onehot, sample_latent=False)  # deterministic eval
            mse = F.mse_loss(preds, imgs, reduction="sum")
            mse_sum += float(mse.item())
            n += imgs.numel()
    return mse_sum / n


@torch.no_grad()
def save_label_grid(model: FOKR2D, device: torch.device, epoch: int, out_dir: str, stochastic: bool = False):
    """Generate one image per digit (0..9) and save a grid."""
    if not _HAS_TORCHVISION:
        return
    model.eval()
    labels = torch.arange(10, device=device)
    X = one_hot(labels, 10)
    Y = model(X, sample_latent=stochastic)  # (10,1,28,28)
    grid = vutils.make_grid(Y, nrow=10, padding=2, normalize=True)
    os.makedirs(out_dir, exist_ok=True)
    tag = "stoch" if stochastic else "det"
    vutils.save_image(grid, os.path.join(out_dir, f"samples_{tag}_epoch{epoch:03d}.png"))


def train_mnist(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, test_loader = get_mnist_loaders(cfg.batch_size)

    model = FOKR2D(
        in_dim=10, img_size=28, grid=cfg.grid,
        lengthscale=cfg.lengthscale, amplitude=cfg.amplitude,
        learn_hyper=cfg.learn_hyper, learn_mu=cfg.learn_mu,
        latent_dim=cfg.latent_dim, kl_weight=cfg.kl_weight,
        mu_tv_weight=cfg.mu_tv_weight,
        deconv_repeats=cfg.deconv_repeats, deconv_C0=cfg.deconv_C0, deconv_Cmid=cfg.deconv_Cmid,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_test = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_onehot = one_hot(labels.to(device, non_blocking=True), num_classes=10)

            opt.zero_grad(set_to_none=True)
            if cfg.amp:
                with torch.cuda.amp.autocast():
                    loss = model.loss(labels_onehot, imgs)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = model.loss(labels_onehot, imgs)
                loss.backward()
                opt.step()

        train_mse = evaluate(model, train_loader, device)
        test_mse  = evaluate(model, test_loader, device)
        print(f"[Epoch {ep:03d}] train MSE={train_mse:.6f} | test MSE={test_mse:.6f}")

        # Save samples
        save_label_grid(model, device, ep, cfg.save_dir, stochastic=False)
        if cfg.latent_dim > 0:
            save_label_grid(model, device, ep, cfg.save_dir, stochastic=True)

        # Save best checkpoint
        if test_mse < best_test:
            best_test = test_mse
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best.pt"))
        # Save last
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last.pt"))

    print(f"Best test MSE: {best_test:.6f}")
    return model


# -------------------------------
# CLI
# -------------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grid", type=int, default=14)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--mu_tv_weight", type=float, default=1e-4)
    p.add_argument("--deconv_repeats", type=int, default=1)
    p.add_argument("--deconv_C0", type=int, default=128)
    p.add_argument("--deconv_Cmid", type=int, default=64)
    p.add_argument("--lengthscale", type=float, default=0.2)
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--learn_mu", action="store_true", default=True)
    p.add_argument("--no_learn_mu", dest="learn_mu", action="store_false")
    p.add_argument("--learn_hyper", action="store_true", default=False)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./runs/fokr2d_mnist")
    return p


def main():
    args = build_argparser().parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        grid=args.grid,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        mu_tv_weight=args.mu_tv_weight,
        deconv_repeats=args.deconv_repeats,
        deconv_C0=args.deconv_C0,
        deconv_Cmid=args.deconv_Cmid,
        lengthscale=args.lengthscale,
        amplitude=args.amplitude,
        learn_mu=args.learn_mu,
        learn_hyper=args.learn_hyper,
        amp=args.amp,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    train_mnist(cfg)


if __name__ == "__main__":
    main()