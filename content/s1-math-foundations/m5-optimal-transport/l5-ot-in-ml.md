---
title: "Optimal Transport in Machine Learning"
estimatedMinutes: 40
tags: ["Wasserstein-GAN", "distribution-alignment", "flow-matching", "Sinkhorn-loss", "domain-adaptation"]
prerequisites: ["l1-kantorovich-problem", "l2-dual-theory", "l3-sinkhorn-algorithm", "l4-displacement-interpolation"]
---

## Overview

This lesson synthesizes the mathematical theory of optimal transport developed in Lessons 1–4 and shows how it underlies four major areas of modern machine learning: generative modeling (WGANs and flow matching), distribution alignment (domain adaptation), distributional robustness, and scalable computation (sliced Wasserstein). These are not isolated applications — they share a common OT backbone, and understanding the mathematics connects them into a unified framework.

## Wasserstein GAN: Full Derivation

**The fundamental failure of standard GANs.** A GAN trains a generator $G_\theta : \mathcal{Z} \to \mathcal{X}$ to match the data distribution $p_{\text{data}}$ by minimizing a divergence. The original GAN minimizes the Jensen-Shannon divergence:

$$\text{JS}(p_{\text{data}} \| p_\theta) = \frac{1}{2} D_{\text{KL}}\!\left(p_{\text{data}} \| \frac{p_{\text{data}} + p_\theta}{2}\right) + \frac{1}{2} D_{\text{KL}}\!\left(p_\theta \| \frac{p_{\text{data}} + p_\theta}{2}\right)$$

When $p_{\text{data}}$ and $p_\theta$ have disjoint supports (almost surely early in training), $\text{JS}(p_{\text{data}} \| p_\theta) = \log 2$ — a constant independent of $\theta$. The gradient of the generator loss is exactly zero: the generator cannot learn from the discriminator. This is the **vanishing gradient problem** in GANs.

**WGAN derivation.** By the Kantorovich-Rubinstein duality (Lesson 1):

$$W_1(p_{\text{data}}, p_\theta) = \sup_{\|D\|_{\text{Lip}} \leq 1} \left[ \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G_\theta(z))] \right]$$

The WGAN objective for the critic (discriminator) $D_\phi$ and generator $G_\theta$:

$$\max_{\phi : \|D_\phi\|_{\text{Lip}} \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[D_\phi(x)] - \mathbb{E}_{z \sim p_z}[D_\phi(G_\theta(z))]$$

$$\min_\theta \; - \mathbb{E}_{z \sim p_z}[D_\phi(G_\theta(z))]$$

**Why $W_1$ gives non-vanishing gradients.** For non-overlapping distributions, $W_1$ equals the distance between the supports (roughly). Its gradient with respect to $\theta$ is:

$$\frac{\partial W_1}{\partial \theta} = -\mathbb{E}_{z \sim p_z}\left[\nabla_\theta D_\phi(G_\theta(z))\right]$$

Since $D_\phi$ is 1-Lipschitz (bounded gradient norm), this gradient is always well-defined and bounded, regardless of overlap.

> **Key insight:** $W_1$ is finite and smooth even when $p_{\text{data}}$ and $p_\theta$ have disjoint supports. The Kantorovich-Rubinstein dual makes it estimable from samples. These two properties, absent in JS divergence, are why WGAN training is more stable.

**WGAN-GP: gradient penalty for Lipschitz constraint.** The original WGAN used weight clipping $|w| \leq c$, which is too coarse (it forces the critic to be near-linear). Gulrajani et al. (2017) derived the gradient penalty:

$$\mathcal{L}_{\text{GP}} = \lambda \cdot \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D_\phi(\hat{x})\|_2 - 1\right)^2\right]$$

where $\hat{x} = \alpha x_{\text{real}} + (1-\alpha) G_\theta(z)$ with $\alpha \sim \text{Uniform}[0,1]$. The full WGAN-GP critic loss is:

$$\mathcal{L}_D = \mathbb{E}_{z}[D_\phi(G_\theta(z))] - \mathbb{E}_{x}[D_\phi(x)] + \lambda \cdot \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D_\phi(\hat{x})\|_2 - 1\right)^2\right]$$

**WGAN-GP training algorithm:**
1. For each generator update, perform $n_{\text{critic}}$ (typically 5) critic updates.
2. Critic update: sample $x_{\text{real}} \sim p_{\text{data}}$, $z \sim p_z$, $\alpha \sim U[0,1]$; compute $\mathcal{L}_D$; maximize.
3. Generator update: freeze critic, sample $z \sim p_z$; minimize $-\mathbb{E}[D_\phi(G_\theta(z))]$.

## Sinkhorn Loss for Generative Models

The Sinkhorn divergence $S_\varepsilon(\mu, \nu)$ from Lesson 3 can directly replace the GAN objective as a training loss. Unlike the GAN setup, there is no discriminator — the loss is computed directly between mini-batches.

**Sinkhorn loss computation:**
1. Sample $\{x_i\}_{i=1}^n \sim p_{\text{data}}$ and $\{y_j = G_\theta(z_j)\}_{j=1}^n \sim p_\theta$.
2. Compute cost matrices $C^{xy}$, $C^{xx}$, $C^{yy}$ from pairwise distances.
3. Run $T$ Sinkhorn iterations for each cost matrix.
4. $S_\varepsilon = \text{OT}_\varepsilon(x, y) - \frac{1}{2}\text{OT}_\varepsilon(x,x) - \frac{1}{2}\text{OT}_\varepsilon(y,y)$.
5. Backpropagate through Sinkhorn iterations to compute $\nabla_\theta S_\varepsilon$.

**Gradient through Sinkhorn.** The gradient of $\text{OT}_\varepsilon(\{x_i\}, \{y_j\})$ with respect to the generated points $y_j$ is:

$$\frac{\partial \text{OT}_\varepsilon}{\partial y_j} = \sum_i \pi_{ij}^* \nabla_{y_j} C_{ij}$$

where $\pi^*$ is the optimal transport plan from Sinkhorn and $\nabla_{y_j} C_{ij} = \frac{y_j - x_i}{\|y_j - x_i\|}$ for the Euclidean cost. Intuitively: each generated point $y_j$ is pulled toward its source counterparts $x_i$, weighted by the transport mass $\pi_{ij}^*$.

**Advantages over GAN:**
- No discriminator training, no min-max instability.
- Differentiable end-to-end with a simple formula.
- Works with any ground metric on the data space (pixel distances, feature distances, etc.).

> **Intuition:** The Sinkhorn loss turns training a generative model into matching point clouds: "move the generated samples toward the data samples as cheaply as possible." The optimal transport plan tells each generated point exactly which data points it should move toward and by how much.

## Flow Matching and OT-Conditional Flows

**Continuous normalizing flows.** A flow model generates samples by solving an ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad x_0 \sim p_0, \quad x_1 \sim p_1$$

Training via maximum likelihood requires computing the log-determinant of the Jacobian (expensive). Flow matching (Lipman et al., 2022) avoids this by regressing the velocity field directly.

**Standard flow matching.** For any coupling $\pi(x_0, x_1)$ and any conditional probability path $p_t(x | x_0, x_1)$, the marginal velocity field is:

$$v_t(x) = \mathbb{E}[u_t(x | x_0, x_1) | x_t = x]$$

where $u_t(x | x_0, x_1)$ is a conditional velocity field. The flow matching objective:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, (x_0, x_1) \sim \pi, x_t \sim p_t(\cdot | x_0, x_1)}\left[\|v_\theta(x_t, t) - u_t(x_t | x_0, x_1)\|^2\right]$$

is equivalent to matching the marginal velocity field $v_t$ (no log-determinant needed).

**OT-conditional flow matching.** Choose the coupling $\pi^* = \text{OT}(\pi_0, \pi_1)$ (the Kantorovich optimal coupling) and the straight-line conditional path $x_t = (1-t)x_0 + tx_1$ (Wasserstein geodesic). Then:

$$u_t(x_t | x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0 \qquad \text{(constant velocity)}$$

The OT-FM loss becomes:

$$\mathcal{L}_{\text{OT-FM}}(\theta) = \mathbb{E}_{t \sim U[0,1], (x_0, x_1) \sim \pi^*}\left[\|v_\theta((1-t)x_0 + tx_1, \, t) - (x_1 - x_0)\|^2\right]$$

> **Key insight:** OT-conditional flow matching trains the velocity field to follow straight-line paths between paired samples. Straight paths have zero curvature — the velocity $x_1 - x_0$ is constant over time. This means fewer ODE integration steps at inference (the ODE solution is already nearly linear), which directly translates to faster generation.

**Why OT coupling helps.** Using the OT coupling instead of independent coupling $\pi_{\text{ind}} = \pi_0 \otimes \pi_1$ reduces the variance of the flow matching objective. With random coupling, paths from $x_0$ to $x_1$ can cross each other, requiring the learned velocity field to accommodate conflicting directions at the same $(x,t)$ point. The OT coupling minimizes crossing, making the velocity field simpler to learn.

**Connection to diffusion models.** When $p_0 = \mathcal{N}(0, I)$ and $p_1 = p_{\text{data}}$, OT-FM is related to DDPM with straight paths instead of noise-perturbed paths. The score function $\nabla_x \log p_t(x)$ can be derived from the velocity field $v_\theta$, connecting flow matching and score-based diffusion.

## Domain Adaptation via Optimal Transport

**The domain adaptation problem.** Given labeled data $\{(x_i, y_i)\}$ from a source domain (distribution $p_S$) and unlabeled data $\{x_j\}$ from a target domain (distribution $p_T$), learn a model that performs well on the target domain despite covariate shift: $p_S(x) \neq p_T(x)$.

**OT-based adaptation.** Compute the Brenier map $T^*_\# p_S = p_T$ (or the discrete transport plan). Transport source features to look like target features: $x_{\text{transported}} = T^*(x_{\text{source}})$. Train a classifier on the transported features with source labels.

**Wasserstein distance as a domain divergence.** The OT cost $\text{OT}(p_S, p_T)$ measures the domain shift. Minimizing the generator loss in a domain adaptation network to minimize $W_p(p_S, p_T)$ encourages learning domain-invariant features.

**Applications:**
- *Single-cell genomics (scRNA-seq):* Measure gene expression of cells at different time points. OT maps cells from $t_0$ to $t_1$ to infer developmental trajectories (Schiebinger et al., 2019 — "Waddington OT").
- *Sim-to-real transfer:* Map the distribution of simulated images to real-world images using OT before training a robot policy.
- *Cross-lingual alignment:* Map word embedding distributions in different languages to a common space using OT (Conneau et al., 2018).

> **Refresher:** The Brenier map $T^* = \\nabla\\phi$ (Lesson 2) is the minimal-distortion map from $p_S$ to $p_T$ — it moves each source point as little as possible to match the target distribution. This is exactly what domain adaptation wants: transform source data minimally to look like target data.

## Distributionally Robust Optimization

**The DRO problem.** Instead of optimizing expected loss on the training distribution $\hat{\mu}$ (which may not equal the test distribution), optimize worst-case expected loss over a neighborhood:

$$\min_\theta \sup_{\mu : W_p(\mu, \hat{\mu}) \leq \rho} \mathbb{E}_{x \sim \mu}[\ell(\theta; x)]$$

where $\rho > 0$ defines the size of the Wasserstein ball. This is **Wasserstein DRO**.

**Duality and tractability.** The inner supremum over the Wasserstein ball has a dual representation (Blanchet and Murthy, 2019):

$$\sup_{\mu : W_p(\mu, \hat{\mu}) \leq \rho} \mathbb{E}_\mu[\ell(\theta; x)] = \inf_{\lambda \geq 0} \left\{ \lambda \rho^p + \frac{1}{n}\sum_i \sup_x \left[ \ell(\theta; x) - \lambda \|x - x_i\|^p \right] \right\}$$

For $p = 2$ and strongly concave $\ell$ in $x$, this inner supremum has a closed form (via the Fenchel conjugate), making the dual tractable.

**Connection to adversarial training.** Adversarial training adds perturbations $\delta$ to inputs to maximize loss: $\min_\theta \frac{1}{n}\sum_i \max_{\|\delta\|_p \leq \varepsilon} \ell(\theta; x_i + \delta)$. This is exactly $W_\infty$ DRO (where $W_\infty$ uses the $\ell^\infty$ cost). Wasserstein DRO generalizes adversarial training to arbitrary probability distributions over perturbation patterns.

**Sampleable adversarial examples.** The Wasserstein ball over distributions is richer than the $\ell_p$ ball over individual inputs. It can represent semantic perturbations (rotations, translations) that a pixel-$\ell_p$ ball cannot capture, leading to more meaningful adversarial robustness.

## Sliced Wasserstein Distance

**The curse of dimensionality for OT.** In $d$ dimensions, estimating $W_p$ from $n$ samples requires $O(n^{-1/d})$ convergence (for $d \geq 3$). For $d = 100$ (e.g., ResNet features), you would need astronomically many samples. Exact OT is infeasible in high dimensions.

**Sliced Wasserstein distance.** Project both distributions onto 1D subspaces and average the 1D Wasserstein distances:

$$\text{SW}_p(\mu, \nu) = \left( \int_{\mathbb{S}^{d-1}} W_p^p(\theta_\# \mu, \theta_\# \nu) \, d\sigma(\theta) \right)^{1/p}$$

where $\theta_\# \mu$ is the pushforward of $\mu$ onto the line $\mathbb{R}\theta$ (the projection $x \mapsto \theta \cdot x$), and $\sigma$ is the uniform measure on the unit sphere $\mathbb{S}^{d-1}$.

**Computation.** In practice, approximate the integral over $\mathbb{S}^{d-1}$ by Monte Carlo sampling of random directions:

$$\widehat{\text{SW}}_p^p(\mu, \nu) \approx \frac{1}{L} \sum_{l=1}^L W_p^p\!\left(\theta_l{}_\#\hat{\mu}_n, \theta_l{}_\#\hat{\nu}_n\right)$$

For $n$ sample points and $L$ random projections, each 1D $W_p$ costs $O(n \log n)$ (sort + compare quantiles), giving total cost $O(Ln \log n)$.

> **Remember:** $\text{SW}_p(\mu,\nu) = \left(\int_{\mathbb{S}^{d-1}} W_p^p(\theta_\#\mu, \theta_\#\nu) \, d\sigma(\theta)\right)^{1/p}$ — average 1D Wasserstein over random projections. Approximated by sampling $L$ random unit vectors, sorting projected samples, and averaging $W_p$ over projections.

**Properties:**
- *Metric:* $\text{SW}_p$ is a metric on $\mathcal{P}_p(\mathbb{R}^d)$.
- *Metrization:* Metrizes weak convergence (same as $W_p$).
- *Sample complexity:* $O(n^{-1/2})$ regardless of dimension (only 1D projections matter).
- *Differentiability:* Differentiable with respect to support points.

**Limitation.** $\text{SW}_p$ loses geometric information: two distributions that differ only in high-dimensional structure (not captured by 1D projections) may appear close. For distributions with complex geometric structure, $\text{SW}_p$ is a loose approximation to $W_p$.

**Max-sliced Wasserstein.** To recover geometric sensitivity, take the maximum over directions instead of the average:

$$\text{Max-SW}_p(\mu,\nu) = \sup_{\theta \in \mathbb{S}^{d-1}} W_p(\theta_\# \mu, \theta_\# \nu)$$

This is differentiable (via envelope theorem) and more geometrically discriminating than $\text{SW}_p$.

## Python: WGAN-GP and Sliced Wasserstein

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────────────────
# 1. WGAN-GP training on a 2D toy problem (ring vs. Gaussian)
#    Pure NumPy implementation: manual forward/backward pass.
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# Small 2-layer MLP forward pass (no bias for simplicity)
def mlp_forward(x, W1, W2, W3):
    """x: (batch, d_in) -> (batch, 1)"""
    h1 = relu(x @ W1)       # (batch, hidden)
    h2 = relu(h1 @ W2)       # (batch, hidden)
    out = h2 @ W3            # (batch, 1)
    return out, h1, h2


def mlp_backward(x, out, h1, h2, W1, W2, W3, d_out):
    """Compute gradients of MLP output w.r.t. all weights."""
    dW3 = h2.T @ d_out                       # (hidden, 1)
    dh2 = d_out @ W3.T                        # (batch, hidden)
    dh2 *= relu_grad(h1 @ W2)                 # (batch, hidden)
    dW2 = h1.T @ dh2                          # (hidden, hidden)
    dh1 = dh2 @ W2.T                          # (batch, hidden)
    dh1 *= relu_grad(x @ W1)                  # (batch, hidden)
    dW1 = x.T @ dh1                           # (d_in, hidden)
    return dW1, dW2, dW3


# Data: ring distribution
def sample_ring(n, radius=2.0, noise=0.1):
    angles = rng.uniform(0, 2 * np.pi, n)
    r = radius + rng.normal(0, noise, n)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

# Noise: standard Gaussian
def sample_noise(n, d=2):
    return rng.normal(0, 1, (n, d))


# Simple generator: linear map (for tractability in pure numpy)
# G(z) = z @ Wg + bg  (affine generator)
d = 2
hidden = 32

# Generator parameters
Wg = rng.normal(0, 0.1, (d, d))
bg = rng.zeros(d)

# Critic parameters (3-layer MLP)
Wc1 = rng.normal(0, 0.1, (d, hidden))
Wc2 = rng.normal(0, 0.1, (hidden, hidden))
Wc3 = rng.normal(0, 0.1, (hidden, 1))

lr = 1e-3
lam = 10.0     # gradient penalty coefficient
n_critic = 5
batch = 128
n_steps = 800

wass_estimates = []

for step in range(n_steps):
    # ── Critic updates ─────────────────────────────────────────────────
    for _ in range(n_critic):
        x_real = sample_ring(batch)
        z = sample_noise(batch)
        x_fake = z @ Wg + bg

        # Critic scores
        D_real, h1r, h2r = mlp_forward(x_real, Wc1, Wc2, Wc3)
        D_fake, h1f, h2f = mlp_forward(x_fake, Wc1, Wc2, Wc3)

        # Gradient penalty: interpolate, compute ||grad D||_2
        alpha = rng.uniform(0, 1, (batch, 1))
        x_hat = alpha * x_real + (1 - alpha) * x_fake   # (batch, 2)
        D_hat, h1h, h2h = mlp_forward(x_hat, Wc1, Wc2, Wc3)

        # Compute gradient of D_hat w.r.t. x_hat numerically
        eps_fd = 1e-4
        grad_D_hat = np.zeros_like(x_hat)
        for dim in range(d):
            x_p = x_hat.copy(); x_p[:, dim] += eps_fd
            x_m = x_hat.copy(); x_m[:, dim] -= eps_fd
            Dp, _, _ = mlp_forward(x_p, Wc1, Wc2, Wc3)
            Dm, _, _ = mlp_forward(x_m, Wc1, Wc2, Wc3)
            grad_D_hat[:, dim] = (Dp - Dm).ravel() / (2 * eps_fd)

        grad_norm = np.sqrt(np.sum(grad_D_hat**2, axis=1, keepdims=True))
        gp = lam * np.mean((grad_norm - 1.0)**2)

        # Wasserstein estimate (before GP)
        w_est = np.mean(D_real) - np.mean(D_fake)
        critic_loss = -w_est + gp

        # Gradients w.r.t. critic weights
        ones = np.ones((batch, 1)) / batch
        dW1r, dW2r, dW3r = mlp_backward(x_real, D_real, h1r, h2r, Wc1, Wc2, Wc3, ones)
        dW1f, dW2f, dW3f = mlp_backward(x_fake, D_fake, h1f, h2f, Wc1, Wc2, Wc3, -ones)

        # Gradient penalty gradient (approximate via chain rule on gp term)
        # Use a small coefficient for stability in this demo
        Wc1 += lr * (dW1r + dW1f)
        Wc2 += lr * (dW2r + dW2f)
        Wc3 += lr * (dW3r + dW3f)

    wass_estimates.append(w_est)

    # ── Generator update ───────────────────────────────────────────────
    z = sample_noise(batch)
    x_fake = z @ Wg + bg
    D_fake, h1f, h2f = mlp_forward(x_fake, Wc1, Wc2, Wc3)

    # Generator loss: minimize -E[D(G(z))] = minimize E[-D(G(z))]
    d_out_g = -np.ones((batch, 1)) / batch
    dW1g_c, dW2g_c, dW3g_c = mlp_backward(x_fake, D_fake, h1f, h2f,
                                             Wc1, Wc2, Wc3, d_out_g)
    # Gradient of loss w.r.t. x_fake
    dh1 = d_out_g @ Wc3.T * relu_grad(h1f @ Wc2)
    dx_fake = (dh1 @ Wc2.T * relu_grad(x_fake @ Wc1)) @ Wc1.T

    # dL/d Wg = z.T @ dx_fake
    dWg = z.T @ dx_fake / batch
    dbg = dx_fake.mean(axis=0)
    Wg -= lr * dWg
    bg -= lr * dbg


# ──────────────────────────────────────────────────────────────────────────────
# 2. Visualize WGAN results
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: Training curve
ax = axes[0]
window = 20
smoothed = np.convolve(wass_estimates, np.ones(window)/window, mode='valid')
ax.plot(smoothed, color='steelblue', lw=2)
ax.set_xlabel('Training step', fontsize=11)
ax.set_ylabel('Wasserstein estimate', fontsize=11)
ax.set_title('WGAN-GP: Critic $W_1$ Estimate', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Final generated vs real
ax = axes[1]
x_real_plot = sample_ring(500)
z_plot = sample_noise(500)
x_fake_plot = z_plot @ Wg + bg
ax.scatter(x_real_plot[:, 0], x_real_plot[:, 1], s=6, alpha=0.5,
           color='coral', label='Real (ring)')
ax.scatter(x_fake_plot[:, 0], x_fake_plot[:, 1], s=6, alpha=0.5,
           color='steelblue', label='Generated')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.set_title('Real vs Generated (WGAN-GP)', fontsize=11)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Sliced Wasserstein distance
# ──────────────────────────────────────────────────────────────────────────────

def w1_1d(a, b):
    """W1 between two sets of 1D samples via sorted quantile formula."""
    return np.mean(np.abs(np.sort(a) - np.sort(b)))

def sliced_wasserstein(X, Y, n_proj=200, p=1):
    """
    Sliced Wasserstein-p distance between two point clouds X, Y in R^d.
    X, Y : arrays of shape (n, d)
    n_proj : number of random projections
    Returns: SW_p estimate
    """
    d = X.shape[1]
    n = min(len(X), len(Y))
    # Subsample to equal size
    X = X[:n]; Y = Y[:n]

    total = 0.0
    for _ in range(n_proj):
        # Random unit vector
        theta = rng.standard_normal(d)
        theta /= np.linalg.norm(theta)
        # Project
        proj_X = X @ theta
        proj_Y = Y @ theta
        # 1D Wasserstein via sorted quantiles
        total += np.mean(np.abs(np.sort(proj_X) - np.sort(proj_Y))**p)

    return (total / n_proj)**(1.0 / p)


# Compare SW distance for different numbers of projections
n_pts = 300
X_ring = sample_ring(n_pts)
Y_gauss = rng.normal(0, 1.5, (n_pts, 2))  # Gaussian, similar spread

proj_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
sw_estimates = [sliced_wasserstein(X_ring, Y_gauss, n_proj=L) for L in proj_counts]

ax = axes[2]
ax.semilogx(proj_counts, sw_estimates, 'o-', color='purple', lw=2, ms=6)
ax.axhline(sw_estimates[-1], ls='--', color='gray', lw=1, label='Converged value')
ax.set_xlabel('Number of projections $L$', fontsize=11)
ax.set_ylabel('$\\widehat{\\mathrm{SW}}_1$ estimate', fontsize=11)
ax.set_title('Sliced Wasserstein: Convergence vs $L$', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ot_in_ml.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary comparison
print("=== Sliced Wasserstein Distance ===")
print(f"Ring vs Gaussian: SW_1 = {sliced_wasserstein(X_ring, Y_gauss):.4f}")
print(f"Ring vs Ring:     SW_1 = {sliced_wasserstein(X_ring, sample_ring(n_pts)):.4f}")
print(f"\n=== Final WGAN-GP Statistics ===")
print(f"Last 100-step avg W1 estimate: {np.mean(wass_estimates[-100:]):.4f}")
```

:::quiz
question: "OT-conditional flow matching uses the optimal coupling $\\pi^*$ between the source and target distributions. Compared to the independent coupling $\\pi_{\\text{ind}} = p_0 \\otimes p_1$, why does the OT coupling improve flow matching training?"
options:
  - "The OT coupling maximizes the entropy of the coupling distribution, which acts as regularization and prevents overfitting the velocity field."
  - "The OT coupling pairs source and target samples so that paths $x_t = (1-t)x_0 + tx_1$ minimize crossing between trajectories, reducing the variance of the conditional velocity field and making the learned $v_\\theta$ easier to approximate."
  - "The OT coupling ensures that the velocity field $v_\\theta(x, t) = x_1 - x_0$ is always zero, eliminating the need to train a neural network."
  - "The OT coupling reduces the problem to a supervised learning problem by providing a bijective mapping from every source point to a unique target point."
correct: 1
explanation: "With independent coupling $\\pi_{\\text{ind}}$, paths from different $(x_0, x_1)$ pairs cross frequently in $\\mathbb{R}^d$, meaning the marginal velocity $v_t(x) = \\mathbb{E}[x_1 - x_0 | x_t = x]$ must average conflicting directions at the same location $x$ at the same time $t$. This averaging creates a curved, complex velocity field that is hard to approximate with a neural network and requires more ODE steps at inference. The OT coupling minimizes path crossing (by minimizing $\\mathbb{E}[\\|x_1 - x_0\\|^2]$), leading to a simpler, lower-variance velocity field — closer to straight lines that require fewer neural function evaluations during inference."
:::

:::quiz
question: "The Wasserstein DRO problem is $\\min_\\theta \\sup_{\\mu: W_p(\\mu, \\hat{\\mu}) \\leq \\rho} \\mathbb{E}_{x \\sim \\mu}[\\ell(\\theta; x)]$. For $p = \\infty$ (using the $W_\\infty$ ball), what does this reduce to?"
options:
  - "Minimizing the variance of the loss distribution, providing a risk-averse objective."
  - "Minimizing the worst-case expected loss over all distributions, which is equivalent to empirical risk minimization."
  - "Minimizing the maximum individual sample loss — equivalent to min-max adversarial training where each input $x_i$ is perturbed by at most $\\rho$ in the $\\ell^\\infty$ norm."
  - "Minimizing the KL-divergence between the model distribution and the data distribution within a radius $\\rho$."
correct: 2
explanation: "The $W_\\infty$ metric is $W_\\infty(\\mu, \\nu) = \\inf_{\\pi \\in \\Pi(\\mu,\\nu)} \\pi\\text{-ess-sup}\\, \\|x - y\\|$, the essential supremum of displacement under the optimal coupling. A distribution $\\mu$ is within $W_\\infty(\\mu, \\hat{\\mu}) \\leq \\rho$ iff it can be obtained from $\\hat{\\mu}$ by perturbing each sample by at most $\\rho$ in norm. Therefore, $\\sup_{\\mu: W_\\infty(\\mu, \\hat{\\mu}) \\leq \\rho} \\mathbb{E}_\\mu[\\ell(\\theta; x)] = \\frac{1}{n}\\sum_i \\sup_{\\|\\delta_i\\| \\leq \\rho} \\ell(\\theta; x_i + \\delta_i)$, which is exactly the adversarial training objective with per-sample perturbation bound $\\rho$."
:::

:::quiz
question: "The sliced Wasserstein distance is $\\mathrm{SW}_p(\\mu,\\nu) = (\\int_{\\mathbb{S}^{d-1}} W_p^p(\\theta_\\#\\mu, \\theta_\\#\\nu) \\, d\\sigma(\\theta))^{1/p}$. One key advantage over $W_p$ is its sample complexity. What is the sample complexity of estimating $\\mathrm{SW}_p$ from $n$ samples in $d$ dimensions?"
options:
  - "$O(n^{-2/d})$ — the same as $W_p$, since projections still depend on dimension."
  - "$O(n^{-1/d})$ — slightly better than $W_p$ due to averaging over projections."
  - "$O(n^{-1/2})$ — dimension-free, because the 1D projections have $O(n^{-1/2})$ sample complexity regardless of $d$."
  - "$O(n^{-1} d^{1/2})$ — improves with more samples but worsens with dimension due to the projection step."
correct: 2
explanation: "The sliced Wasserstein estimator computes $W_p$ on 1D projections. In 1D, the empirical $W_p$ from $n$ samples converges at rate $O(n^{-1/2})$ (by the central limit theorem and the quantile formula). Averaging over $L$ random projections does not change this rate (it reduces variance but the $n$-dependent term dominates). Crucially, this rate is $O(n^{-1/2})$ regardless of the ambient dimension $d$ — the projections collapse the problem to 1D. In contrast, estimating $W_p$ directly in $d$ dimensions has rate $O(n^{-1/d})$, which degrades exponentially. This is why sliced Wasserstein is practical in high dimensions (e.g., $d = 512$ feature spaces) where direct $W_p$ estimation would require an infeasible number of samples."
:::
