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

What's new vs the basic version:
  - Beta network can be an MLP ("mlp") or a small ConvTranspose2d decoder ("deconv"),
    which outputs a (grid x grid) map that is flattened to beta (length d = grid^2).
  - Optional reparameterization trick: z ~ N(mu(X), diag(sigma^2(X))),
    concatenated to X before the Beta network. Adds KL term to the loss.

This file provides:
  - FOKR2D module with beta_arch={'mlp','deconv'} and reparam options
  - Training utilities (MNIST and synthetic)
  - Self-tests (synthetic) without internet

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

# torchvision is only needed for the MNIST example (internet may be required for first-time download).
try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# -------------------------------
# Utilities
# -------------------------------

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

class BetaNetMLP(nn.Module):
    """
    MLP mapping input vector X (plus optional z) to beta (length d).
    """
    def __init__(self, in_dim: int, d: int, hidden: Tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2, inplace=True)]
            last = h
        layers += [nn.Linear(last, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetaNetDeconv(nn.Module):
    """
    Deconvolutional decoder producing a (grid x grid) beta map.

    Strategy:
      - Project dense input to a small spatial tensor (C0 x 7 x 7)
      - ConvTranspose2d to upsample to (C1 x 14 x 14)  [works for grid=14]
      - 1x1 conv to 1 channel -> flatten to length d=grid*grid
    """
    def __init__(self, in_dim: int, grid: int, ch: Tuple[int, int] = (128, 64)):
        super().__init__()
        assert grid in (7, 14, 28, 32, 56), "This demo assumes grid = 14 by default (others require tweaking)."
        self.grid = grid
        self.C0, self.C1 = ch

        # pick base size that doubles to grid
        if grid % 14 == 0:
            base = grid // 2  # 7 if grid=14
            up_stride = 2
            out_pad = 1 if base * 2 == grid and base % 2 != 0 else 0
        else:
            # fallback: nearest scheme
            base = 7
            up_stride = 2
            out_pad = 1

        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.C0 * base * base),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.base = base

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.C0, self.C1, kernel_size=3, stride=up_stride, padding=1, output_padding=out_pad, bias=False),
            nn.BatchNorm2d(self.C1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.C1, 1, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.fc(x)                              # (B, C0*base*base)
        h = h.view(B, self.C0, self.base, self.base)
        h = self.deconv(h)                          # (B,1,grid,grid)
        return h.view(B, self.grid * self.grid)     # (B, d)


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
    Functional Output Kernel Regression for 2D images.

    Given input X (B, in_dim) [+ optional z] -> beta (B, d) via selected BetaNet.
    Using precomputed Phi (m,d), we form Y_hat_flat = beta @ Phi^T + mu
    and reshape to (B, 1, img_size, img_size).
    """
    def __init__(
        self,
        in_dim: int = 10,
        img_size: int = 28,
        grid: int = 14,
        lengthscale: float = 0.15,
        amplitude: float = 1.0,
        learn_hyper: bool = False,
        learn_mu: bool = True,
        beta_arch: str = "mlp",           # {"mlp","deconv"}
        hidden: Tuple[int, ...] = (256, 256, 256),
        latent_dim: int = 0,              # 0 disables reparam trick
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.img_size = img_size
        self.grid = grid
        self.m = img_size * img_size
        self.d = grid * grid
        self.beta_arch = beta_arch
        self.kl_weight = kl_weight

        # Features (kernel dictionary)
        self.features = RBFGrid2D(img_size=img_size, grid=grid,
                                  lengthscale=lengthscale, amplitude=amplitude,
                                  learn_hyper=learn_hyper)

        # Optional latent
        self.latent_dim = int(latent_dim)
        if self.latent_dim > 0:
            self.reparam = LatentReparam(in_dim=in_dim, latent_dim=self.latent_dim)
            beta_in = in_dim + self.latent_dim
        else:
            self.reparam = None
            beta_in = in_dim

        # Beta net
        if beta_arch == "mlp":
            self.betanet = BetaNetMLP(in_dim=beta_in, d=self.d, hidden=hidden)
        elif beta_arch == "deconv":
            self.betanet = BetaNetDeconv(in_dim=beta_in, grid=grid, ch=(128, 64))
        else:
            raise ValueError("beta_arch must be 'mlp' or 'deconv'")

        # Intercept mu(t)
        if learn_mu:
            self.mu = nn.Parameter(torch.zeros(self.m, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(self.m, dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, sample_latent: bool = True, return_kl: bool = False):
        """
        x: (B, in_dim)
        returns:
          - y: (B, 1, img_size, img_size)
          - (optional) kl: scalar tensor (mean KL)
        """
        B = x.shape[0]

        # latent
        kl = x.new_tensor(0.0)
        if self.reparam is not None:
            z, kl_val = self.reparam(x, sample=sample_latent and self.training)
            x_in = torch.cat([x, z], dim=1)
            kl = kl_val
        else:
            x_in = x

        # beta
        beta = self.betanet(x_in)          # (B,d)
        Phi = self.features()               # (m,d) on CPU buffer
        Phi = Phi.to(beta.device)           # ensure device match
        y_flat = beta @ Phi.T + self.mu     # (B,m)
        y = y_flat.view(B, 1, self.img_size, self.img_size)

        if return_kl:
            return y, kl
        return y

    def loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        MSE + KL (if latent used)
        y_true: (B,1,H,W) in [0,1]
        """
        if self.reparam is not None:
            y_pred, kl = self.forward(x, sample_latent=True, return_kl=True)
            return F.mse_loss(y_pred, y_true) + self.kl_weight * kl
        else:
            y_pred = self.forward(x, sample_latent=False, return_kl=False)
            return F.mse_loss(y_pred, y_true)


# -------------------------------
# Training / Evaluation
# -------------------------------

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    beta_arch: str = "deconv"     # 'mlp' or 'deconv'
    latent_dim: int = 8           # set 0 to disable reparam
    kl_weight: float = 1e-4
    grid: int = 14
    lengthscale: float = 0.2
    amplitude: float = 1.0


def get_mnist_loaders(batch_size: int = 128) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is not available. Cannot build MNIST loaders.")
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


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


def train_mnist(cfg: TrainConfig):
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is not available. Cannot train on MNIST in this environment.")
    device = torch.device(cfg.device)
    train_loader, test_loader = get_mnist_loaders(cfg.batch_size)

    model = FOKR2D(
        in_dim=10, img_size=28, grid=cfg.grid,
        lengthscale=cfg.lengthscale, amplitude=cfg.amplitude,
        learn_hyper=False, learn_mu=True,
        beta_arch=cfg.beta_arch, hidden=(256, 256, 256),
        latent_dim=cfg.latent_dim, kl_weight=cfg.kl_weight
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(cfg.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels_onehot = one_hot(labels.to(device), num_classes=10)
            loss = model.loss(labels_onehot, imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        train_mse = evaluate(model, train_loader, device)
        test_mse = evaluate(model, test_loader, device)
        print(f"[Epoch {ep+1:02d}] train MSE={train_mse:.6f} | test MSE={test_mse:.6f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/fokr2d_mnist.pt")
    print("Saved checkpoint to ./checkpoints/fokr2d_mnist.pt")


# -------------------------------
# Synthetic self-test (no internet/data required)
# -------------------------------

def synthetic_dataset(n: int = 2048, seed: int = 0, img_size: int = 28, grid: int = 14) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic dataset by generating ground-truth beta = W x + b and
    images via RBF features + mu. X is one-hot in {0..9}. Returns (X_onehot, images).
    """
    g = torch.Generator().manual_seed(seed)

    # One-hot labels
    labels = torch.randint(0, 10, (n,), generator=g)
    X = one_hot(labels, num_classes=10)

    # Ground-truth parameters
    rbf = RBFGrid2D(img_size=img_size, grid=grid, lengthscale=0.2, amplitude=1.0, learn_hyper=False)
    Phi = rbf()  # (m,d)
    m, d = Phi.shape
    W = torch.randn(10, d, generator=g) * 0.3
    b = torch.randn(d, generator=g) * 0.1
    mu = torch.randn(m, generator=g) * 0.05

    beta = X @ W + b          # (n,d)
    y_flat = beta @ Phi.T + mu  # (n,m)
    y = y_flat.view(n, 1, img_size, img_size).clamp(0.0, 1.0)  # clamp to image range

    return X, y


def unit_test_synthetic(epochs: int = 3, batch_size: int = 256, lr: float = 1e-3,
                        beta_arch: str = "mlp", latent_dim: int = 0, kl_weight: float = 1e-4):
    """
    Train FOKR2D on the synthetic dataset and report MSE before/after training,
    verifying gradients and learning dynamics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = synthetic_dataset(n=1024, img_size=28, grid=14)
    X = X.to(device)
    Y = Y.to(device)

    model = FOKR2D(
        in_dim=10, img_size=28, grid=14, lengthscale=0.2, amplitude=1.0,
        learn_hyper=False, learn_mu=True, beta_arch=beta_arch,
        hidden=(256, 256, 256), latent_dim=latent_dim, kl_weight=kl_weight
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def mse_all(sample_latent: bool):
        with torch.no_grad():
            if latent_dim > 0:
                yp = model(X, sample_latent=sample_latent, return_kl=False)
            else:
                yp = model(X, sample_latent=False, return_kl=False)
            return F.mse_loss(yp, Y).item()

    mse0 = mse_all(sample_latent=False)
    print(f"[Synthetic] Initial MSE: {mse0:.6f}")

    ds = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        for x, y in loader:
            loss = model.loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"[Synthetic] Epoch {ep+1:02d}, MSE (deterministic eval): {mse_all(sample_latent=False):.6f}")

    mseF = mse_all(sample_latent=False)
    print(f"[Synthetic] Final MSE: {mseF:.6f}, reduction factor={(mse0 / (mseF+1e-12)):.2f}")
    return mse0, mseF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "mnist"], default="synthetic",
                        help="Run synthetic self-test (no data needed) or MNIST training (requires torchvision and data download).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_arch", choices=["mlp", "deconv"], default="deconv")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    args = parser.parse_args()

    if args.mode == "synthetic":
        unit_test_synthetic(epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            beta_arch=args.beta_arch,
                            latent_dim=args.latent_dim,
                            kl_weight=args.kl_weight)
    elif args.mode == "mnist":
        cfg = TrainConfig(epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr=args.lr,
                          beta_arch=args.beta_arch,
                          latent_dim=args.latent_dim,
                          kl_weight=args.kl_weight)
        train_mnist(cfg)


if __name__ == "__main__":
    main()
