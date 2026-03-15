---
title: "Displacement Interpolation & Wasserstein Geometry"
estimatedMinutes: 35
tags: ["geodesics", "displacement-interpolation", "McCann", "Wasserstein-barycenter", "generative-models"]
prerequisites: ["l1-kantorovich-problem", "l2-dual-theory"]
---

## Wasserstein Space as a Metric Space

The Wasserstein-2 distance $W_2$ does not just compare distributions — it turns the space of probability measures into a rich geometric object. The space $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ is a complete metric space, and like all metric spaces it admits geodesics: shortest paths between points.

**Definition (geodesic).** A curve $(\mu_t)_{t \in [0,1]}$ in $(\mathcal{P}_2, W_2)$ is a **constant-speed geodesic** from $\mu_0$ to $\mu_1$ if:

$$W_2(\mu_s, \mu_t) = |t - s| \cdot W_2(\mu_0, \mu_1) \qquad \text{for all } s, t \in [0,1]$$

In other words, the curve traverses the "shortest path" from $\mu_0$ to $\mu_1$ at constant speed, and there are no detours.

**Construction from the optimal coupling.** Let $\pi^*$ be the optimal coupling of $\mu_0$ and $\mu_1$ under $W_2$. For each $t \in [0,1]$, define the interpolation map:

$$T_t : \mathcal{X} \times \mathcal{X} \to \mathcal{X}, \qquad T_t(x, y) = (1-t)x + ty$$

The **Wasserstein geodesic** is:

$$\mu_t = (T_t)_\# \pi^*$$

This measure $\mu_t$ is the law of $(1-t)X + tY$ where $(X,Y) \sim \pi^*$.

> **Intuition:** Each grain of sand at position $x$ is transported to position $y$ (according to the optimal plan $\\pi^*$). The geodesic at time $t$ places the grain at the convex combination $(1-t)x + ty$. The grain travels a straight line in $\\mathbb{R}^d$ at constant speed. This is the minimal-energy interpolation: no grain takes a detour, and there are no crossings.

**Theorem (geodesic completeness).** The curve $(\mu_t)_{t \in [0,1]}$ constructed above is the unique constant-speed geodesic from $\mu_0$ to $\mu_1$ when $\mu_0$ is absolutely continuous. When $\mu_0$ is not absolutely continuous, there may be multiple geodesics.

## Displacement Interpolation vs. Linear Interpolation

There are two natural ways to interpolate between probability measures $\mu_0$ and $\mu_1$:

**Linear (mixture) interpolation:**

$$\mu_t^{\text{lin}} = (1-t)\mu_0 + t\mu_1$$

**Displacement interpolation (Wasserstein geodesic):**

$$\mu_t^{\text{disp}} = (T_t)_\# \pi^*$$

These are fundamentally different operations with different qualitative behavior.

> **Key insight:** Linear interpolation mixes mass — it creates a superposition of $\\mu_0$ and $\\mu_1$. Displacement interpolation moves mass — each particle travels along a straight path. When the two measures have disjoint supports, linear interpolation creates a bimodal distribution for all $t \\in (0,1)$, while displacement interpolation remains unimodal for $t$ away from the endpoints.

**The Gaussian case.** This difference is most vivid for $\mu_0 = \mathcal{N}(-a, \sigma^2)$ and $\mu_1 = \mathcal{N}(+a, \sigma^2)$ with $a \gg \sigma$.
- *Linear interpolation:* $\mu_t^{\text{lin}} = (1-t)\mathcal{N}(-a,\sigma^2) + t\mathcal{N}(+a,\sigma^2)$, a bimodal Gaussian mixture for all $t \in (0,1)$.
- *Displacement interpolation:* The optimal map is $T(x) = x + 2a$ (a translation), so $\pi^* = (\text{Id}, T)_\# \mu_0$ and $\mu_t^{\text{disp}} = \mathcal{N}((2t-1)a, \sigma^2)$, a unimodal Gaussian that slides from $-a$ to $+a$ at constant speed.

For two general Gaussians $\mathcal{N}(m_0, \Sigma_0)$ and $\mathcal{N}(m_1, \Sigma_1)$, the geodesic is:

$$\mu_t = \mathcal{N}(m_t, \Sigma_t)$$

$$m_t = (1-t)m_0 + tm_1, \qquad \Sigma_t = \left[(1-t)I + t \Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}\Sigma_0^{-1/2}\right]^2 \Sigma_0$$

The interpolated covariance is **not** the linear interpolation $(1-t)\Sigma_0 + t\Sigma_1$ — it is the geodesic interpolation in the space of positive definite matrices.

## McCann's Theorem

McCann (1997) established the general theory of Wasserstein geodesics for the quadratic cost.

**Theorem (McCann).** Let $\mu_0$ be absolutely continuous. The unique constant-speed geodesic from $\mu_0$ to $\mu_1$ in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ is:

$$\mu_t = \left((1-t)\text{Id} + t\nabla\phi\right)_\# \mu_0$$

where $\phi$ is the Brenier potential (the convex function whose gradient gives the optimal map $T^* = \nabla\phi$).

Equivalently, $\mu_t$ is the pushforward of $\mu_0$ by the map $x \mapsto x + t(\nabla\phi(x) - x)$.

**Proof.** The optimal coupling is $\pi^* = (\text{Id}, \nabla\phi)_\# \mu_0$. The geodesic is:

$$\mu_t = (T_t)_\# \pi^* = \left((1-t)\text{Id} + t\nabla\phi\right)_\# \mu_0$$

since $T_t(x, y) = (1-t)x + ty$ applied to $(x, y) = (x, \nabla\phi(x))$ gives $(1-t)x + t\nabla\phi(x)$. $\square$

> **Remember:** The Wasserstein geodesic from $\\mu_0$ to $\\mu_1$ is $\\mu_t = ((1-t)\\text{Id} + t\\nabla\\phi)_\\# \\mu_0$, where $\\nabla\\phi$ is the Brenier map from $\\mu_0$ to $\\mu_1$. The interpolation is linear on particles, not on densities.

## The Wasserstein Barycenter

The Wasserstein barycenter is the natural notion of "average distribution" in Wasserstein space. Given $K$ measures $\mu_1, \ldots, \mu_K \in \mathcal{P}_2(\mathbb{R}^d)$ with positive weights $w_k \geq 0$, $\sum_k w_k = 1$, the **Wasserstein barycenter** is:

$$\bar{\mu} = \arg\min_{\mu \in \mathcal{P}_2} \sum_{k=1}^K w_k W_2^2(\mu, \mu_k)$$

This is the **Fréchet mean** of the measures $\mu_1, \ldots, \mu_K$ in the metric space $(\mathcal{P}_2, W_2)$.

**Existence and uniqueness.** If at least one $\mu_k$ is absolutely continuous, the barycenter $\bar{\mu}$ exists and is unique. For general discrete measures, existence holds but uniqueness may fail.

**Gaussian barycenter (closed form).** For $\mu_k = \mathcal{N}(m_k, \Sigma_k)$ with uniform weights $w_k = 1/K$, the barycenter is Gaussian $\bar{\mu} = \mathcal{N}(\bar{m}, \bar{\Sigma})$ where:

$$\bar{m} = \frac{1}{K}\sum_k m_k$$

$$\bar{\Sigma} = \frac{1}{K^2} \sum_{k=1}^K \bar{\Sigma}^{1/2} \Sigma_k \bar{\Sigma}^{1/2} \cdot \bar{\Sigma}^{-1} \quad \text{(fixed-point equation)}$$

More precisely, $\bar{\Sigma}$ is the unique positive definite solution of the **Bures-Wasserstein fixed point equation**:

$$\bar{\Sigma} = \frac{1}{K} \sum_{k=1}^K \left(\bar{\Sigma}^{1/2} \Sigma_k \bar{\Sigma}^{1/2}\right)^{1/2} \bar{\Sigma}^{-1/2} \cdot \bar{\Sigma}^{1/2}$$

This simplifies to: $\bar{\Sigma} = \frac{1}{K} \sum_{k=1}^K (\bar{\Sigma}^{1/2} \Sigma_k \bar{\Sigma}^{1/2})^{1/2} \bar{\Sigma}^{-1}$... In practice, for uniform weights over isotropic Gaussians $\Sigma_k = \sigma_k^2 I$, the barycenter is $\bar{\Sigma} = \bar{\sigma}^2 I$ with $\bar{\sigma} = \frac{1}{K}\sum_k \sigma_k$ (arithmetic mean of standard deviations, not variances).

**Fixed-point characterization (general case).** The barycenter satisfies:

$$\bar{\mu} = \left(\frac{1}{K} \sum_{k=1}^K T_k\right)_\# \bar{\mu}$$

where $T_k = \nabla\phi_k$ is the optimal transport map from $\bar{\mu}$ to $\mu_k$. This says $\bar{\mu}$ is a fixed point of the "average map" — a point is its own barycenter under the average of the maps to each $\mu_k$.

**Iterative algorithm (Alvarez-Esteban et al., 2016).** Starting from any $\bar{\mu}^{(0)}$:
1. Compute optimal maps $T_k^{(t)}$ from $\bar{\mu}^{(t)}$ to each $\mu_k$.
2. Update: $\bar{\mu}^{(t+1)} = \left(\frac{1}{K}\sum_k T_k^{(t)}\right)_\# \bar{\mu}^{(t)}$.

This converges under mild conditions and is the basis for discrete and continuous barycenter algorithms.

## Curvature and Displacement Convexity

**Wasserstein space has non-negative curvature** in the Alexandrov sense: for any geodesic triangle, the "comparison triangle" in Euclidean space is no thicker. Equivalently, the squared distance function $W_2^2(\cdot, \nu)$ is "not too non-convex" along geodesics.

**Consequences:** Non-negative curvature implies that the Wasserstein barycenter is unique for $W_2$ (since the uniqueness argument for barycenters requires that the squared distance is strictly convex along geodesics). This is in contrast to negatively curved spaces (like hyperbolic space) where barycenters can be non-unique.

**Displacement convexity (McCann, 1997).** A functional $\mathcal{F} : \mathcal{P}_2 \to \mathbb{R}$ is **displacement convex** if it is convex along every Wasserstein geodesic:

$$\mathcal{F}(\mu_t) \leq (1-t)\mathcal{F}(\mu_0) + t\mathcal{F}(\mu_1) \qquad \text{for all } t \in [0,1]$$

**Examples of displacement convex functionals:**
- *Entropy:* $H(\mu) = \int \rho \log \rho \, dx$ — strictly displacement convex.
- *Potential energy:* $V(\mu) = \int V(x) \, d\mu(x)$ for convex $V$ — displacement convex.
- *Interaction energy:* $W(\mu) = \frac{1}{2}\int\!\!\int W(x-y) \, d\mu(x) \, d\mu(y)$ for convex $W$ — displacement convex.

**Connection to gradient flows and diffusion.** The heat equation $\partial_t \rho = \Delta \rho$ is the gradient flow of the entropy $H(\rho)$ in Wasserstein space (Jordan-Kinderlehrer-Otto, 1998). This is the Otto calculus interpretation: "entropy decreases as fast as possible in Wasserstein space" = diffusion. Displacement convexity ensures convergence to the unique minimizer (the Gaussian, for bounded domains).

> **Key insight:** The connection between diffusion and Wasserstein gradient flows is not incidental — it is the mathematical foundation of diffusion models. Training a score function corresponds to learning the Wasserstein gradient of the entropy, and the reverse diffusion is the Wasserstein gradient descent on the KL divergence.

## ML Connection: Latent Space Interpolation and Flow Matching

**Latent space interpolation.** In a VAE or GAN, encoding images $x_0$ and $x_1$ gives latent codes $z_0$ and $z_1$. Simple linear interpolation $z_t = (1-t)z_0 + tz_1$ in latent space can produce unrealistic intermediate images if the latent space is not well-structured (the straight line may leave the region of high probability density).

Displacement interpolation provides a principled alternative: compute the Wasserstein geodesic between the latent distributions $p(z | x_0)$ and $p(z | x_1)$, then sample from $\mu_t$. For diagonal Gaussians (as in VAEs), this reduces to coordinate-wise interpolation of means and standard deviations along the Bures geodesic.

**Flow matching connection.** OT-conditional flow matching (Lipman et al., 2022) directly applies the Wasserstein geodesic idea:
- Sample $(x_0, x_1) \sim \pi^*$ (optimal coupling or independent coupling as approximation).
- The conditional vector field along the geodesic is $u_t(x | x_0, x_1) = x_1 - x_0$ (constant velocity field of the straight-line path).
- Train a neural vector field $v_\theta(x, t)$ to match: $\mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$.
- At inference, solve the ODE $\dot{x}_t = v_\theta(x_t, t)$ from $t=0$ to $t=1$.

When $\pi^*$ is the exact OT coupling, the velocity field $v = x_1 - x_0$ is constant (no acceleration), which corresponds to the straightest possible paths in distribution space. Straight paths → fewer ODE integration steps → faster inference.

## Python: Displacement Interpolation and Wasserstein Barycenter

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sqrtm_spd(A):
    """Matrix square root of a symmetric positive definite matrix."""
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.maximum(w, 0))) @ v.T

def brenier_map_gaussian(m0, Sigma0, m1, Sigma1):
    """
    Closed-form Brenier map T(x) = A(x - m0) + m1 between Gaussians.
    A = Sigma0^{-1/2} (Sigma0^{1/2} Sigma1 Sigma0^{1/2})^{1/2} Sigma0^{-1/2}
    """
    S0h = sqrtm_spd(Sigma0)
    S0ih = np.linalg.inv(S0h)
    M = sqrtm_spd(S0h @ Sigma1 @ S0h)
    A = S0ih @ M @ S0ih
    return A, m0, m1

def geodesic_gaussian(m0, Sigma0, m1, Sigma1, t):
    """
    Wasserstein-2 geodesic between two Gaussians at time t.
    Returns mean and covariance of mu_t.
    """
    S0h = sqrtm_spd(Sigma0)
    S0ih = np.linalg.inv(S0h)
    M = sqrtm_spd(S0h @ Sigma1 @ S0h)
    A = S0ih @ M @ S0ih   # Brenier map linear part
    # Map at time t: T_t(x) = (1-t)x + t*T(x) = ((1-t)I + t*A)(x - m0) + (1-t)*m0 + t*m1
    At = (1 - t) * np.eye(len(m0)) + t * A
    mt = (1 - t) * m0 + t * m1
    Sigmat = At @ Sigma0 @ At.T
    return mt, Sigmat


# ──────────────────────────────────────────────────────────────────────────────
# 1D case: displacement vs linear interpolation
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(99)
n_samples = 2000

# Two well-separated 1D Gaussians
mu0_mean, mu0_std = -2.5, 0.6
mu1_mean, mu1_std = +2.5, 0.6
samples0 = rng.normal(mu0_mean, mu0_std, n_samples)
samples1 = rng.normal(mu1_mean, mu1_std, n_samples)

# 1D optimal map: sort both and match (monotone rearrangement)
s0_sorted = np.sort(samples0)
s1_sorted = np.sort(samples1)

t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = cm.plasma(np.linspace(0.1, 0.9, len(t_vals)))
x_plot = np.linspace(-5, 5, 400)


def gaussian_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))


fig, axes = plt.subplots(2, 5, figsize=(16, 5))

for col, t in enumerate(t_vals):
    # Displacement interpolation: match t-th quantile
    # mu_t ~ N((1-t)*m0 + t*m1, ((1-t)*s0 + t*s1)^2) for 1D Gaussians
    mt_disp = (1 - t) * mu0_mean + t * mu1_mean
    st_disp = (1 - t) * mu0_std + t * mu1_std  # std interpolates linearly
    # (because the Brenier map in 1D Gaussian case is a scaling + shift)

    # Linear (mixture) interpolation
    # mu_t^lin = (1-t)*N(m0,s0^2) + t*N(m1,s1^2) — a Gaussian mixture

    axes[0, col].plot(x_plot, gaussian_pdf(x_plot, mt_disp, st_disp),
                      color=colors[col], lw=2.5)
    axes[0, col].set_title(f'$t={t}$ (disp.)', fontsize=9)
    axes[0, col].set_ylim(0, 0.8)
    axes[0, col].set_yticks([])

    lin_pdf = ((1 - t) * gaussian_pdf(x_plot, mu0_mean, mu0_std)
               + t * gaussian_pdf(x_plot, mu1_mean, mu1_std))
    axes[1, col].plot(x_plot, lin_pdf, color=colors[col], lw=2.5)
    axes[1, col].set_title(f'$t={t}$ (linear)', fontsize=9)
    axes[1, col].set_ylim(0, 0.8)
    axes[1, col].set_yticks([])

axes[0, 0].set_ylabel('Displacement\ninterp.', fontsize=9)
axes[1, 0].set_ylabel('Linear\ninterp.', fontsize=9)
fig.suptitle('Displacement vs Linear Interpolation between $\\mathcal{N}(-2.5, 0.36)$ and $\\mathcal{N}(2.5, 0.36)$',
             fontsize=11)
plt.tight_layout()
plt.savefig('displacement_interpolation.png', dpi=150, bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2D Wasserstein barycenter of K Gaussians (closed-form for uniform weights)
# Fixed-point iteration: Sigma_bar = (1/K) sum_k sqrtm(Sigma_bar^{1/2} Sigma_k Sigma_bar^{1/2}) Sigma_bar^{-1}
# ──────────────────────────────────────────────────────────────────────────────

def wasserstein_barycenter_gaussians(means, covs, weights=None, n_iter=50):
    """
    Compute Wasserstein barycenter of K Gaussian distributions.
    Uses fixed-point iteration for the covariance.
    """
    K = len(means)
    if weights is None:
        weights = np.ones(K) / K
    d = means[0].shape[0]

    m_bar = sum(w * m for w, m in zip(weights, means))
    Sigma_bar = np.eye(d)   # initialization

    for _ in range(n_iter):
        Sigma_bar_half = sqrtm_spd(Sigma_bar)
        Sigma_bar_inv_half = np.linalg.inv(Sigma_bar_half)

        T_sum = np.zeros((d, d))
        for w, Sigma_k in zip(weights, covs):
            M = sqrtm_spd(Sigma_bar_half @ Sigma_k @ Sigma_bar_half)
            T_sum += w * Sigma_bar_inv_half @ M @ Sigma_bar_inv_half

        Sigma_bar = T_sum @ Sigma_bar @ T_sum
    return m_bar, Sigma_bar


# K = 5 Gaussians in 2D arranged in a ring
K = 5
angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
radius = 3.0
means_list = [np.array([radius * np.cos(a), radius * np.sin(a)]) for a in angles]
# Each Gaussian is elongated in a different direction
covs_list = []
for a in angles:
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cov = rot @ np.diag([1.5, 0.3]) @ rot.T
    covs_list.append(cov)

m_bar, Sigma_bar = wasserstein_barycenter_gaussians(means_list, covs_list)
print(f"Wasserstein barycenter mean: {m_bar}")
print(f"Wasserstein barycenter covariance:\n{Sigma_bar}")

# Visualization
fig, ax = plt.subplots(figsize=(7, 7))
theta_plot = np.linspace(0, 2 * np.pi, 100)
unit_circle = np.column_stack([np.cos(theta_plot), np.sin(theta_plot)])

c_src = plt.cm.Set1(np.linspace(0, 0.8, K))
for i, (m, Sig, col) in enumerate(zip(means_list, covs_list, c_src)):
    # 2-sigma ellipse
    L = np.linalg.cholesky(Sig + 1e-9 * np.eye(2))
    ellipse = 2 * (unit_circle @ L.T) + m
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=col, lw=2, alpha=0.8)
    ax.plot(*m, 'o', color=col, ms=8)

# Barycenter
L_bar = np.linalg.cholesky(Sigma_bar + 1e-9 * np.eye(2))
ell_bar = 2 * (unit_circle @ L_bar.T) + m_bar
ax.plot(ell_bar[:, 0], ell_bar[:, 1], 'k-', lw=3, label='Barycenter (2$\\sigma$)')
ax.plot(*m_bar, 'k*', ms=14, label='Barycenter mean')
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.set_title('Wasserstein Barycenter of 5 Gaussians', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wasserstein_barycenter.png', dpi=150, bbox_inches='tight')
plt.show()
```

:::quiz
question: "Let $\\mu_0 = \\mathcal{N}(-3, 1)$ and $\\mu_1 = \\mathcal{N}(+3, 1)$ on $\\mathbb{R}$. What is the Wasserstein-2 geodesic $\\mu_{1/2}$ at $t = 1/2$?"
options:
  - "$\\mu_{1/2} = \\frac{1}{2}\\mathcal{N}(-3,1) + \\frac{1}{2}\\mathcal{N}(+3,1)$ — the equal-weight mixture of the two Gaussians."
  - "$\\mu_{1/2} = \\mathcal{N}(0, 1)$ — the Gaussian with mean zero and the same variance."
  - "$\\mu_{1/2} = \\mathcal{N}(0, 4)$ — the Gaussian with mean zero and doubled variance to account for the spread."
  - "$\\mu_{1/2} = \\mathcal{N}(0, 9)$ — the Gaussian centered at zero with variance equal to the average of the squared means."
correct: 1
explanation: "For $\\mu_0 = \\mathcal{N}(-3, \\sigma^2)$ and $\\mu_1 = \\mathcal{N}(+3, \\sigma^2)$ with equal variances, the optimal 1D transport map is the translation $T(x) = x + 6$. The geodesic at time $t$ pushes $\\mu_0$ by the fraction $t$ of the total displacement: $T_t(x) = x + 6t$, so $\\mu_t = \\mathcal{N}(-3 + 6t, 1)$. At $t = 1/2$, $\\mu_{1/2} = \\mathcal{N}(0, 1)$ — a single unimodal Gaussian at zero with the same variance. This is the key contrast with linear interpolation, which would give the bimodal mixture."
:::

:::quiz
question: "A functional $\\mathcal{F}(\\mu)$ is displacement convex if it is convex along Wasserstein geodesics. Which of the following statements about displacement convexity is correct?"
options:
  - "A functional that is convex in the usual sense (along linear mixtures) is always displacement convex."
  - "The entropy $H(\\mu) = \\int \\rho \\log \\rho \\, dx$ is displacement convex, which implies the heat equation is a gradient flow in Wasserstein space that converges to equilibrium."
  - "Displacement convexity implies the functional has a unique minimizer over all of $\\mathcal{P}_2$."
  - "The negative entropy $-H(\\mu)$ is displacement convex, which is why entropy maximization produces smooth distributions."
correct: 1
explanation: "The entropy $H(\\mu) = \\int \\rho \\log \\rho \\, dx$ is strictly displacement convex (McCann, 1997). This has profound consequences: the heat equation $\\partial_t \\rho = \\Delta \\rho$ is the Wasserstein gradient flow of $H$, meaning diffusion is 'steepest descent on entropy in Wasserstein space.' Displacement convexity ensures convergence to the unique minimizer (the uniform/Gaussian distribution on bounded domains). Usual convexity (convexity along linear paths) is a different property — the entropy is actually concave along linear paths ($H((1-t)\\mu + t\\nu) \\geq (1-t)H(\\mu) + tH(\\nu)$ by concavity of $-x\\log x$)."
:::

:::quiz
question: "The Wasserstein barycenter of measures $\\mu_1, \\ldots, \\mu_K$ with uniform weights satisfies the fixed-point equation $\\bar{\\mu} = (\\frac{1}{K}\\sum_k T_k)_\\# \\bar{\\mu}$ where $T_k$ is the optimal transport map from $\\bar{\\mu}$ to $\\mu_k$. For $K = 2$ and $\\mu_1 = \\delta_0$, $\\mu_2 = \\delta_2$ on $\\mathbb{R}$, what is $\\bar{\\mu}$?"
options:
  - "$\\bar{\\mu} = \\frac{1}{2}\\delta_0 + \\frac{1}{2}\\delta_2$ — the equal-weight mixture of the two point masses."
  - "$\\bar{\\mu} = \\delta_1$ — the point mass at the midpoint $x = 1$."
  - "$\\bar{\\mu} = \\delta_0$ — the barycenter collapses to the first measure."
  - "$\\bar{\\mu}$ does not exist because the Wasserstein barycenter requires absolutely continuous measures."
correct: 1
explanation: "For $\\mu_1 = \\delta_0$ and $\\mu_2 = \\delta_2$ with equal weights, $\\arg\\min_\\mu \\frac{1}{2}W_2^2(\\mu, \\delta_0) + \\frac{1}{2}W_2^2(\\mu, \\delta_2)$. For any measure $\\mu$, $W_2^2(\\mu, \\delta_a) = \\int (x - a)^2 \\, d\\mu(x) = \\text{Var}(\\mu) + (\\mathbb{E}[\\mu] - a)^2$. Minimizing over $\\mu$: the variance term is minimized by a point mass ($\\text{Var} = 0$), and the mean must equal $\\frac{1}{2}(0 + 2) = 1$. So $\\bar{\\mu} = \\delta_1$. The fixed-point equation confirms: $T_1(x) = 0$ (map $\\delta_1$ to $\\delta_0$) and $T_2(x) = 2$ (map $\\delta_1$ to $\\delta_2$), so $\\frac{1}{2}(T_1 + T_2)_\\# \\delta_1 = \\frac{1}{2}(0 + 2) = 1$, i.e., $\\delta_1$. Barycenter of point masses exists — it is the weighted average point mass."
:::
