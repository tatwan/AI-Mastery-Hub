---
title: "Stochastic Differential Equations & Diffusion"
estimatedMinutes: 35
tags: ["SDE", "Ito-calculus", "Brownian-motion", "Fokker-Planck", "diffusion-models", "score-matching"]
prerequisites: ["l1-measure-theory", "l4-martingales"]
---

# Stochastic Differential Equations & Diffusion

## Brownian Motion

**Brownian motion** (or the Wiener process) $W_t$ is the continuous-time analogue of a random walk and the fundamental building block of stochastic calculus. It is defined by four properties:

1. $W_0 = 0$
2. **Independent increments**: $W_t - W_s$ is independent of $\mathcal{F}_s$ for $s < t$
3. **Gaussian increments**: $W_t - W_s \sim \mathcal{N}(0, t - s)$
4. **Continuous paths**: $t \mapsto W_t(\omega)$ is continuous for almost every $\omega$

These innocuous-looking properties create an object with extraordinary roughness. The key fact is that the **quadratic variation** of Brownian motion is deterministic:

$$\langle W \rangle_t = \lim_{n \to \infty} \sum_{i} (W_{t_{i+1}} - W_{t_i})^2 = t$$

Informally, $(dW_t)^2 = dt$. This means Brownian paths have infinite total variation on any interval — they are far too irregular for classical calculus. A smooth function has zero quadratic variation, so the nonzero quadratic variation of $W_t$ forces us to develop a new calculus.

> **Key insight:** The identity $(dW_t)^2 = dt$ is the single equation that distinguishes stochastic calculus from ordinary calculus. Every unusual feature of Itô calculus — the correction term in Itô's lemma, the difference between Itô and Stratonovich integrals — traces back to this fact.

## Stochastic Differential Equations

A **stochastic differential equation** (SDE) takes the form:

$$dX_t = f(X_t, t) \, dt + g(X_t, t) \, dW_t$$

where $f$ is the **drift** (deterministic tendency) and $g$ is the **diffusion** (noise intensity). The SDE is shorthand for the integral equation:

$$X_t = X_0 + \int_0^t f(X_s, s) \, ds + \int_0^t g(X_s, s) \, dW_s$$

The first integral is an ordinary Riemann/Lebesgue integral. The second is a **stochastic integral** that requires careful definition because $W_s$ has infinite variation.

SDEs define not just a single trajectory but a probability distribution over trajectories — a **stochastic process** $\{X_t\}_{t \geq 0}$. This distribution-over-paths viewpoint is central to diffusion models, which use SDEs to define both the noising and denoising processes.

## The Itô Integral

The **Itô integral** $\int_0^T g(t) \, dW_t$ is constructed as the $L^2$ limit of left-endpoint Riemann sums:

$$\int_0^T g(t) \, dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} g(t_i)(W_{t_{i+1}} - W_{t_i})$$

The crucial choice is evaluating $g$ at the **left** endpoint $t_i$, not the midpoint or right endpoint. This choice — the **Itô convention** — ensures two critical properties:

1. **Martingale property**: $\mathbb{E}\left[\int_0^T g(t) \, dW_t\right] = 0$ (the integral is a martingale)
2. **Itô isometry**: $\mathbb{E}\left[\left(\int_0^T g(t) \, dW_t\right)^2\right] = \int_0^T \mathbb{E}[g(t)^2] \, dt$

The martingale property means stochastic integrals have zero mean — the "noise" integrates to zero on average. The Itô isometry converts the variance of a stochastic integral into an ordinary integral, making calculations tractable.

The alternative Stratonovich convention (midpoint evaluation) gives different results — specifically, it obeys the classical chain rule but loses the martingale property. Itô's convention is standard in probability and ML; Stratonovich is preferred in some physics contexts. The two are related by a drift correction: a Stratonovich SDE can always be converted to an Itô SDE and vice versa.

## Itô's Lemma

The **Itô formula** is the chain rule of stochastic calculus. For a twice continuously differentiable function $f(x, t)$ and a process $X_t$ satisfying $dX_t = \mu \, dt + \sigma \, dW_t$:

$$df(X_t, t) = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} \, dW_t$$

The term $\frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}$ is the **Itô correction**. It has no analogue in ordinary calculus and arises directly from $(dW_t)^2 = dt$. In a Taylor expansion, the second-order term $\frac{1}{2}f''(dX)^2$ normally vanishes as $dt \to 0$, but the $\sigma^2 (dW)^2 = \sigma^2 dt$ part survives.

**Example.** Let $f(x) = x^2$ and $X_t = W_t$. By Itô's lemma:

$$d(W_t^2) = 2W_t \, dW_t + \frac{1}{2} \cdot 2 \cdot dt = 2W_t \, dW_t + dt$$

Taking expectations: $\mathbb{E}[W_t^2] = t$, consistent with $W_t \sim \mathcal{N}(0, t)$. The extra $dt$ term — absent in ordinary calculus — is what makes $\mathbb{E}[W_t^2]$ grow linearly rather than remaining zero.

### Multivariate Itô's Lemma

For the practically important case of a vector process $\mathbf{X}_t \in \mathbb{R}^d$ satisfying $d\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,dt + G(\mathbf{X}_t, t)\,d\mathbf{W}_t$ (where $\mathbf{W}_t \in \mathbb{R}^m$ is an $m$-dimensional Brownian motion and $G \in \mathbb{R}^{d \times m}$), the Itô formula for $\phi(\mathbf{X}_t, t)$ is:

$$d\phi = \left(\frac{\partial \phi}{\partial t} + (\nabla_x \phi)^T \mathbf{f} + \frac{1}{2}\text{tr}\!\left(G G^T \nabla_x^2 \phi\right)\right)dt + (\nabla_x \phi)^T G \, d\mathbf{W}_t$$

The Itô correction term $\frac{1}{2}\text{tr}(GG^T H_\phi)$ involves the **Hessian** of $\phi$ contracted with the diffusion covariance $GG^T$. This is exactly what appears in the Fokker-Planck equation for multivariate diffusion processes — and what makes the score-matching objective in diffusion models tractable when $GG^T = \sigma_t^2 I$ (isotropic noise).

> **Key insight:** Itô's lemma is the workhorse of SDE computations. Any time you need to find the SDE for a transformed process — log-prices in finance, log-densities in diffusion models — you apply Itô's lemma and collect the correction term.

## The Fokker-Planck Equation

While the SDE describes the evolution of a single sample path, the **Fokker-Planck equation** (also called the Kolmogorov forward equation) describes the evolution of the probability density $p(x, t)$ of $X_t$:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[f(x,t) \, p] + \frac{1}{2}\frac{\partial^2}{\partial x^2}[g(x,t)^2 \, p]$$

The first term is the **advection** (transport by the drift); the second is **diffusion** (spreading by the noise). The Fokker-Planck equation is a deterministic PDE — all stochasticity has been "integrated out" into the density.

This equation is the bridge between the sample-level view (individual trajectories of the SDE) and the distribution-level view (how the density evolves). In diffusion models, the forward process gradually transforms the data distribution into noise; the Fokker-Planck equation tracks this transformation analytically.

For constant coefficients $f = \mu$, $g = \sigma$, the Fokker-Planck equation is the heat equation with drift, and its solution starting from $\delta(x - x_0)$ is the Gaussian $\mathcal{N}(x_0 + \mu t, \sigma^2 t)$.

## The Ornstein-Uhlenbeck Process

The **Ornstein-Uhlenbeck (OU) process** is the prototypical mean-reverting SDE:

$$dX_t = -\theta X_t \, dt + \sigma \, dW_t$$

The drift $-\theta X_t$ pulls the process toward zero (when $\theta > 0$), while the diffusion $\sigma \, dW_t$ adds noise. The balance between mean-reversion and noise injection creates a stationary distribution:

$$X_\infty \sim \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$$

The transition kernel is also Gaussian: given $X_s = x$,

$$X_t | X_s = x \sim \mathcal{N}\left(x e^{-\theta(t-s)}, \frac{\sigma^2}{2\theta}(1 - e^{-2\theta(t-s)})\right)$$

As $t - s \to \infty$, the conditional distribution converges to the stationary distribution regardless of the starting point. This is precisely the noising process used in many diffusion models: start with data $X_0 \sim p_{\text{data}}$ and run the OU process forward until $X_T \approx \mathcal{N}(0, \sigma^2/2\theta)$.

## Diffusion Models as SDEs

The framework of Song et al. (2020) unifies score-based generative models through the SDE lens. The **forward process** progressively corrupts data:

$$dX_t = f(X_t, t) \, dt + g(t) \, dW_t$$

starting from $X_0 \sim p_{\text{data}}$. Common choices include the Variance Preserving (VP) SDE, which corresponds to the DDPM noising schedule, and the Variance Exploding (VE) SDE.

The key theoretical result is that the **reverse-time SDE** exists and takes the form:

$$dX_t = \left[f(X_t, t) - g(t)^2 \nabla_x \log p_t(X_t)\right] dt + g(t) \, d\bar{W}_t$$

where $\bar{W}_t$ is a reverse-time Brownian motion and $p_t$ is the marginal density of $X_t$ under the forward process. Running this SDE backward in time — from $T$ to $0$ — transforms noise back into data.

The critical unknown is the **score function** $\nabla_x \log p_t(x)$ — the gradient of the log-density at each time $t$. This quantity is not available in closed form (if we knew $p_t$ analytically, we would not need a generative model). Instead, we train a neural network $s_\theta(x, t)$ to approximate it.

> **Key insight:** The reverse SDE reveals why diffusion models work: denoising is itself a stochastic process with a well-defined SDE. The drift of the reverse SDE has two components — the original forward drift $f$ (known) and a correction $-g^2 \nabla_x \log p_t$ (learned). The score function is the only thing we need to learn; everything else is specified by the forward process.

## Score Matching

How do we train $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ without access to $p_t$? The answer is **denoising score matching**. The forward process gives us the conditional density $p(x_t | x_0)$ in closed form (it is Gaussian for affine SDEs). The training objective is:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{x_t \sim p(x_t|x_0)} \left[\left\| s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0) \right\|^2\right]$$

Since $p(x_t | x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)$, the conditional score is:

$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2}$$

This is computable — it only requires knowing the forward process parameters $\alpha_t$ and $\sigma_t$, plus a sample $x_0$ from the data. The remarkable fact (proved by Vincent, 2011) is that minimizing this denoising objective is equivalent to minimizing the "true" score matching loss $\mathbb{E}[\|s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t)\|^2]$ up to a constant independent of $\theta$.

## Connecting the Pieces

The full pipeline of a diffusion model, viewed through the SDE lens:

1. **Forward SDE** corrupts data into noise, governed by chosen drift $f$ and diffusion $g$
2. **Fokker-Planck equation** describes how $p_t$ evolves (analytically for affine SDEs)
3. **Score network** $s_\theta$ is trained via denoising score matching to approximate $\nabla_x \log p_t$
4. **Reverse SDE** generates samples by integrating backward from $X_T \sim \mathcal{N}(0, \sigma^2 I)$, using $s_\theta$ in the drift
5. **Itô's lemma** and **Girsanov's theorem** provide the theoretical tools to analyze sampling quality, probability flow ODEs, and likelihood computation

The probability flow ODE — an alternative to the reverse SDE with the same marginals but no stochasticity — is obtained by removing the diffusion term and adjusting the drift: $dX_t = [f - \frac{1}{2}g^2 \nabla_x \log p_t] dt$. This deterministic trajectory provides exact likelihood computation via the instantaneous change of variables formula (a consequence of the Fokker-Planck equation). DDIM is a discrete approximation to this ODE.

## Python: Forward Noising and Score Estimation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Ornstein-Uhlenbeck forward process: dX = -theta*X dt + sigma dW
theta, sigma = 1.0, 1.5
dt = 0.01
T = 5.0
n_steps = int(T / dt)
n_paths = 5000

# Sample x0 from a bimodal data distribution
x0 = np.where(np.random.rand(n_paths) < 0.5,
               np.random.randn(n_paths) - 3,
               np.random.randn(n_paths) + 3)

# Simulate forward SDE via Euler-Maruyama
X = np.zeros((n_paths, n_steps + 1))
X[:, 0] = x0
for i in range(n_steps):
    dW = np.sqrt(dt) * np.random.randn(n_paths)
    X[:, i+1] = X[:, i] - theta * X[:, i] * dt + sigma * dW

# Estimate score at t=T/2 via finite differences on kernel density
t_idx = n_steps // 2
x_mid = X[:, t_idx]
x_grid = np.linspace(-6, 6, 200)
bw = 0.3  # bandwidth
log_kde = np.array([
    np.log(np.mean(np.exp(-0.5 * ((x - x_mid) / bw)**2)) + 1e-30)
    for x in x_grid
])
# Score = d/dx log p(x) via central differences
dx = x_grid[1] - x_grid[0]
score_est = np.gradient(log_kde, dx)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(x0, bins=60, density=True, alpha=0.5, label='$t=0$ (data)')
plt.hist(X[:, -1], bins=60, density=True, alpha=0.5, label=f'$t={T}$ (noise)')
plt.legend()
plt.title('Forward noising: data → noise')

plt.subplot(1, 2, 2)
plt.plot(x_grid, score_est, label='Estimated $\\nabla_x \\log p_{t}(x)$')
plt.axhline(0, color='gray', ls='--', lw=0.5)
plt.xlabel('$x$')
plt.title(f'Score function at $t={T/2:.1f}$')
plt.legend()
plt.tight_layout()
plt.show()
```

The left panel shows the bimodal data distribution at $t=0$ being transformed into an approximately Gaussian distribution at $t=T$ by the OU forward process. The right panel shows the estimated score function at the midpoint $t = T/2$: it points toward the modes of the distribution (negative slope between modes, positive at the tails), which is exactly the "denoising direction" that the reverse SDE follows.

:::quiz
question: "In Itô's lemma, the 'correction term' ½σ²f'' arises because:"
options:
  - "Brownian motion has non-differentiable paths, requiring regularization"
  - "The quadratic variation of Brownian motion satisfies (dW)² = dt, so second-order terms in the Taylor expansion survive"
  - "The Stratonovich integral convention requires this correction"
  - "The drift term f(X_t, t) contributes a second-order effect"
correct: 1
explanation: "In a Taylor expansion of f(X_t), the term ½f''(dX)² normally vanishes as dt→0. But (dW)² = dt ≠ 0, so the σ²(dW)² = σ²dt piece of (dX)² persists, giving the ½σ²f'' correction. Option A is vague and not the mechanism; option C reverses the relationship (Stratonovich has no correction but loses the martingale property); option D is incorrect since the drift contributes only first-order terms."
:::

:::quiz
question: "The reverse-time SDE for diffusion models contains the term -g(t)²∇ₓ log p_t(x). This score function is estimated via denoising score matching because:"
options:
  - "The score function is always Gaussian, so only the mean and variance need to be estimated"
  - "The marginal density p_t(x) is intractable, but the conditional density p(x_t|x_0) is Gaussian and known, and minimizing the denoising loss is equivalent to matching the true score"
  - "Direct score matching requires computing the Hessian of the log-density, which is too expensive"
  - "The Fokker-Planck equation provides p_t(x) in closed form for any forward SDE"
correct: 1
explanation: "The marginal p_t(x) = ∫p(x_t|x_0)p_data(x_0)dx_0 is intractable (it involves integrating over the data distribution). However, p(x_t|x_0) is Gaussian for affine SDEs, and Vincent (2011) showed that the denoising score matching objective — matching ∇log p(x_t|x_0) — is equivalent to the true score matching objective up to θ-independent constants. Option A is false (the score is only Gaussian for Gaussian p_t); option C describes a practical concern but not the fundamental reason; option D is false for nonlinear SDEs or non-Gaussian data."
:::

:::quiz
question: "The Ornstein-Uhlenbeck process dX = -θX dt + σ dW has stationary distribution N(0, σ²/2θ). If we double the mean-reversion rate θ while keeping σ fixed, the stationary variance:"
options:
  - "Doubles, because stronger mean-reversion increases fluctuations"
  - "Halves, because stronger mean-reversion pulls X toward zero more aggressively"
  - "Stays the same, because variance depends only on σ"
  - "Quadruples, because variance scales as θ²"
correct: 1
explanation: "The stationary variance is σ²/(2θ). Doubling θ halves the variance: stronger mean-reversion confines the process closer to zero, reducing the spread. The noise intensity σ injects variance, while the mean-reversion θ dissipates it — the stationary distribution reflects the balance between these two forces."
:::
