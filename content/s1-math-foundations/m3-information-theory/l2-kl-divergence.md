---
title: "KL Divergence & f-Divergences"
estimatedMinutes: 30
tags: ["kl-divergence", "variational-inference", "f-divergences", "VAE"]
prerequisites: ["l1-entropy"]
---

## KL Divergence: The Fundamental Asymmetric Divergence

The **Kullback-Leibler divergence** measures how one probability distribution differs from another:

$$D_{\text{KL}}(P \| Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{X \sim P}\left[\log \frac{p(X)}{q(X)}\right]$$

For continuous distributions, the sum becomes an integral. The KL divergence has a clean operational interpretation: it is the **expected extra cost** (in nats or bits) of encoding samples from $P$ using a code optimized for $Q$ instead of $P$. Equivalently, from the cross-entropy decomposition:

$$D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)$$

where $H(P, Q) = -\sum p(x) \log q(x)$ is the cross-entropy and $H(P)$ is the entropy of $P$.

### Non-Negativity: Gibbs' Inequality

A fundamental property: $D_{\text{KL}}(P \| Q) \geq 0$ with equality iff $P = Q$ almost everywhere. The proof follows directly from Jensen's inequality applied to the convex function $-\log$:

$$D_{\text{KL}}(P \| Q) = -\mathbb{E}_P\left[\log \frac{q(X)}{p(X)}\right] \geq -\log \mathbb{E}_P\left[\frac{q(X)}{p(X)}\right] = -\log \sum_x p(x) \frac{q(x)}{p(x)} = -\log 1 = 0$$

Jensen's inequality is tight iff $q(x)/p(x)$ is constant $P$-a.s., which requires $P = Q$.

### KL Is Not a Metric

Despite measuring "distance" between distributions, KL divergence fails two metric axioms:

1. **Asymmetry:** $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ in general. The direction matters enormously.
2. **No triangle inequality:** There exist $P, Q, R$ where $D_{\text{KL}}(P \| R) > D_{\text{KL}}(P \| Q) + D_{\text{KL}}(Q \| R)$.

This asymmetry is not a defect — it reflects genuinely different optimization problems, as we'll see next.

> **Key insight:** KL divergence is not a distance. It is a *directed* measure: $D_{\text{KL}}(P \| Q)$ asks "how expensive is it to use $Q$ when the truth is $P$?" Swapping the arguments changes the question entirely.

## Forward KL vs Reverse KL: Mode-Covering vs Mode-Seeking

The choice of KL direction has profound consequences for approximate inference.

### Forward KL: $D_{\text{KL}}(P \| Q)$ — Mode-Covering

Minimizing $D_{\text{KL}}(P \| Q)$ with respect to $Q$ expands the expectation:

$$\min_Q \; \mathbb{E}_{X \sim P}\left[\log \frac{p(X)}{q(X)}\right] = \min_Q \; -\mathbb{E}_{X \sim P}[\log q(X)] + \text{const}$$

The penalty is infinite wherever $P$ has support but $Q$ does not: if $p(x) > 0$ and $q(x) = 0$, the term $p(x) \log(p(x)/q(x)) \to \infty$. This forces $Q$ to **cover all modes** of $P$, even at the cost of placing mass where $P$ is near zero. The resulting $Q$ tends to be overdispersed.

### Reverse KL: $D_{\text{KL}}(Q \| P)$ — Mode-Seeking

Minimizing $D_{\text{KL}}(Q \| P)$ with respect to $Q$:

$$\min_Q \; \mathbb{E}_{X \sim Q}\left[\log \frac{q(X)}{p(X)}\right]$$

Now the penalty is infinite wherever $Q$ has support but $P$ does not. The optimizer avoids placing any mass of $Q$ outside the support of $P$, which means $Q$ tends to **lock onto a single mode** and ignore others. The result is underdispersed but never places mass in low-probability regions.

This distinction is critical in variational inference: the standard ELBO objective optimizes the reverse KL $D_{\text{KL}}(q(z|x) \| p(z|x))$, which is why variational posteriors tend to be mode-seeking and underestimate uncertainty.

> **Key insight:** Forward KL produces conservative, spread-out approximations that cover the true distribution. Reverse KL produces confident, concentrated approximations that may miss modes. Neither is universally better — the choice depends on whether false negatives (missing modes) or false positives (hallucinating mass) are more costly.

:::quiz
question: "You are fitting a unimodal Gaussian $q$ to a bimodal distribution $p$ with modes at $x=-3$ and $x=3$. Under forward KL $D_{\text{KL}}(p \\| q)$, where will $q$ center?"
options:
  - "At $x = -3$"
  - "At $x = 3$"
  - "At $x = 0$, between the two modes"
  - "It depends on initialization"
correct: 2
explanation: "Forward KL is mode-covering: $q$ must place mass wherever $p$ has mass. A unimodal Gaussian achieves this best by centering between the modes ($x=0$) with high variance, covering both modes despite the valley between them."
:::

## The f-Divergence Family

KL divergence is one member of a broader family. An **f-divergence** is defined for any convex function $f$ with $f(1) = 0$:

$$D_f(P \| Q) = \sum_x q(x) \, f\left(\frac{p(x)}{q(x)}\right)$$

Different choices of $f$ yield different divergences:

| Divergence | $f(t)$ | Formula |
|---|---|---|
| KL divergence | $t \log t$ | $\sum p \log(p/q)$ |
| Reverse KL | $-\log t$ | $\sum q \log(q/p)$ |
| Total variation | $\frac{1}{2}|t - 1|$ | $\frac{1}{2}\sum |p(x) - q(x)|$ |
| Hellinger squared | $(\sqrt{t} - 1)^2$ | $\sum (\sqrt{p} - \sqrt{q})^2$ |
| $\chi^2$ divergence | $(t-1)^2$ | $\sum \frac{(p(x) - q(x))^2}{q(x)}$ |

All f-divergences share key properties: non-negativity, $D_f(P \| Q) = 0$ iff $P = Q$, and invariance under sufficient statistics. They are also jointly convex in $(P, Q)$.

**Total variation** $\delta(P, Q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ is the only f-divergence that is a true metric. It has the operational interpretation: $\delta(P,Q) = \max_A |P(A) - Q(A)|$ — the maximum difference in probability assigned to any event.

The Hellinger distance $H(P,Q) = \sqrt{H^2(P,Q)}$ is also a metric and satisfies useful bounds with total variation:

$$H^2(P,Q) \leq \delta(P,Q) \leq H(P,Q)\sqrt{2}$$

### Why f-Divergences Matter for GANs

The f-GAN framework (Nowozin et al., 2016) showed that any f-divergence can be estimated from samples via its variational (Fenchel conjugate) lower bound:

$$D_f(P \| Q) \geq \sup_T \left[\mathbb{E}_P[T(x)] - \mathbb{E}_Q[f^*(T(x))]\right]$$

where $f^*$ is the convex conjugate of $f$. This unifies many GAN variants: the original GAN approximately minimizes Jensen-Shannon divergence (an f-divergence), while other choices of $f$ yield different training dynamics. This equivalence holds only when the discriminator is **optimal** (i.e., at the global maximum of the GAN objective). In practice, with a suboptimal discriminator, the generator minimizes a different divergence — a subtlety that motivates the various GAN training instabilities.

## Jensen-Shannon Divergence: The Symmetric Alternative

The **Jensen-Shannon divergence** symmetrizes KL by averaging both directions through a mixture:

$$\text{JSD}(P \| Q) = \frac{1}{2}D_{\text{KL}}(P \| M) + \frac{1}{2}D_{\text{KL}}(Q \| M), \quad M = \frac{P+Q}{2}$$

Key properties:
- **Symmetric:** $\text{JSD}(P\|Q) = \text{JSD}(Q\|P)$
- **Bounded:** $0 \leq \text{JSD}(P\|Q) \leq \log 2$ (in nats), unlike KL which is unbounded
- **Square root is a metric:** $\sqrt{\text{JSD}(P\|Q)}$ satisfies the triangle inequality

The JSD is related to the original GAN objective: the optimal discriminator in a standard GAN minimizes $2\,\text{JSD}(P_{\text{data}} \| P_G) - \log 4$. This neat connection explains why GAN training tries to "match distributions" — it is literally minimizing a divergence between the data and generator distributions.

> **Key insight:** JSD is well-defined even when $P$ and $Q$ have non-overlapping support (unlike KL, which diverges to infinity). This makes it better-behaved for comparing distributions on different manifolds — though Wasserstein distance (Semester 5) handles this even more gracefully.

## The Donsker-Varadhan Representation

For KL divergence specifically, a tighter variational bound exists. The **Donsker-Varadhan representation**:

$$D_{\text{KL}}(P \| Q) = \sup_{T: \mathbb{E}_Q[e^T] < \infty} \left[\mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T]\right]$$

where the supremum is over all measurable functions $T$. The optimal $T^*(x) = \log \frac{p(x)}{q(x)} + C$ for any constant $C$. This representation is the foundation of neural KL estimation and connects to the MINE estimator (covered in Lesson 3).

> **Key insight:** The Donsker-Varadhan bound turns KL estimation into an optimization problem over function space — exactly the kind of problem neural networks can approximate. This is why we can estimate divergences between complex, high-dimensional distributions using learned discriminators.

## KL Divergence in VAEs: The ELBO

The variational autoencoder provides the most important application of KL divergence in modern deep learning. For a latent variable model $p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$, the log-evidence decomposes as:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{ELBO}(x; \theta, \phi)} + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$$

Since the last term is non-negative, the **Evidence Lower Bound (ELBO)** is a lower bound on $\log p_\theta(x)$. Maximizing the ELBO simultaneously:

1. **Maximizes reconstruction** — the $\mathbb{E}[\log p_\theta(x|z)]$ term pushes the decoder to reconstruct $x$ well.
2. **Minimizes rate** — the $D_{\text{KL}}(q_\phi(z|x) \| p(z))$ term regularizes the encoder toward the prior.

The KL term acts as a **rate penalty**: it measures how many nats of information the encoder transmits about $x$ through the bottleneck $z$. This is a direct connection to rate-distortion theory (Lesson 4).

:::quiz
question: "In the VAE ELBO, what happens if $D_{\text{KL}}(q(z|x) \\| p(z)) = 0$ for all $x$?"
options:
  - "The model achieves perfect reconstruction"
  - "The encoder ignores the input entirely (posterior collapse)"
  - "The ELBO equals the log-evidence exactly"
  - "The decoder becomes deterministic"
correct: 1
explanation: "If $D_{\text{KL}}(q(z|x) \\| p(z)) = 0$, then $q(z|x) = p(z)$ for all $x$ — the encoder's output doesn't depend on $x$. This is posterior collapse: the latent code carries zero information about the input, and the decoder must generate outputs from the prior alone."
:::

## Python Example: Forward vs Reverse KL

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Target: bimodal mixture of Gaussians
def p_pdf(x):
    return 0.5 * norm.pdf(x, -3, 0.8) + 0.5 * norm.pdf(x, 3, 0.8)

# Approximation: single Gaussian q(x; mu, sigma)
x_grid = np.linspace(-8, 8, 2000)
dx = x_grid[1] - x_grid[0]
p_vals = p_pdf(x_grid)
p_vals /= (p_vals.sum() * dx)  # normalize on grid

def forward_kl(params):
    """D_KL(p || q): mode-covering — forces q to spread over both modes."""
    mu, log_sigma = params
    q_vals = norm.pdf(x_grid, mu, np.exp(log_sigma))
    q_vals = np.clip(q_vals, 1e-10, None)
    return np.sum(p_vals * np.log(p_vals / q_vals)) * dx

def reverse_kl(params):
    """D_KL(q || p): mode-seeking — q collapses onto one mode."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q_vals = norm.pdf(x_grid, mu, sigma)
    q_vals = np.clip(q_vals, 1e-10, None)
    p_clipped = np.clip(p_vals, 1e-10, None)
    return np.sum(q_vals * np.log(q_vals / p_clipped)) * dx

# Forward KL: expect mu ≈ 0, large sigma (covers both modes)
res_fwd = minimize(forward_kl, [0.0, 1.0], method='Nelder-Mead')
print(f"Forward KL → mu={res_fwd.x[0]:.2f}, sigma={np.exp(res_fwd.x[1]):.2f}")

# Reverse KL: expect mu ≈ ±3, small sigma (locks onto one mode)
res_rev = minimize(reverse_kl, [2.0, 0.5], method='Nelder-Mead')
print(f"Reverse KL → mu={res_rev.x[0]:.2f}, sigma={np.exp(res_rev.x[1]):.2f}")
```

Running this confirms: forward KL places $q$ centered near $\mu \approx 0$ with large $\sigma$ (covering both modes), while reverse KL collapses $q$ onto one mode near $\mu \approx 3$ with $\sigma \approx 0.8$.

:::quiz
question: "The Donsker-Varadhan representation $D_{\text{KL}}(P \\| Q) = \\sup_T [\\mathbb{E}_P[T] - \\log \\mathbb{E}_Q[e^T]]$ is useful because:"
options:
  - "It makes KL divergence symmetric"
  - "It allows estimating KL from samples without knowing the density ratio"
  - "It proves KL divergence satisfies the triangle inequality"
  - "It converts KL divergence to total variation distance"
correct: 1
explanation: "The Donsker-Varadhan bound converts KL computation into an optimization over functions $T$, which can be parameterized by neural networks and optimized with samples from $P$ and $Q$. This avoids needing explicit density ratio $p/q$, which is typically unavailable for complex distributions."
:::

## Summary

KL divergence is the central divergence measure in ML: it appears in cross-entropy losses, variational inference, the ELBO, and as the theoretical justification for maximum likelihood. The forward/reverse KL distinction explains why variational posteriors are mode-seeking, and the f-divergence framework generalizes KL to a family of divergences that underpin GAN training. The Donsker-Varadhan representation bridges the gap between theory and neural estimation.
