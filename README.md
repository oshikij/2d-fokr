# 2D Functional Output Kernel Regression (FOKR)

This repository presents a PyTorch implementation of the **2D image generation experiment** using Functional Output Kernel Regression (FOKR), specifically applied to the MNIST dataset.

The implementation includes an extension where a Variational Autoencoder (VAE)-style latent variable is incorporated into the input for the $\beta$-network, enhancing the model's capacity for generating diverse functional outputs (i.e., different styles of the same digit).

This code was developed to implement the 2D image synthesis method described, but not provided, by the authors of the following reference. 


## Reference Paper
**Functional Output Regression for Machine Learning in Materials Science**
* **Authors:** Megumi Iwayama, Stephen Wu, Chang Liu and Ryo Yoshida
* **Journal:** J. Chem. Inf. Model. 2022, 62, 23, 4837-4851
* **DOI:** [10.1021/acs.jcim.2c00626](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00626)

## Model Architecture
The core FOKR formulation is used to predict the image surface $Y(x, t)$ by linearly combining Gaussian RBF basis functions $\Phi(t)$ with input-dependent coefficients $\beta(x)$.

### Core Equation
$$Y(\mathbf{x}, t) = \sum_{i=1}^{d} \beta_i(\mathbf{x}, \mathbf{z}) \cdot \Phi_i(t) + \mu(t)$$

