---
title: "Proximal Methods & Constrained Optimization"
estimatedMinutes: 35
tags: ["proximal", "ADMM", "LASSO", "Frank-Wolfe", "projected-gradient", "sparsity", "constrained"]
prerequisites: ["l1-convex-analysis", "l2-gradient-methods"]
---

## The Proximal Operator

Many important ML objectives take the form $f(x) = g(x) + h(x)$, where $g$ is smooth (differentiable with Lipschitz gradient) but $h$ is non-smooth — for example, $\ell_1$ regularization ($h(x) = \lambda\|x\|_1$) or indicator functions of convex constraints. Subgradient descent can handle this but converges slowly ($O(1/\sqrt{T})$) because it ignores the smooth-non-smooth structure.

**Proximal methods** exploit this structure by handling the non-smooth part exactly at each step, achieving the faster $O(1/T)$ rate of smooth gradient descent.

**Definition.** The **proximal operator** of a function $h$ with parameter $\eta > 0$ is:

$$\text{prox}_{\eta h}(v) = \arg\min_{x} \left\{ h(x) + \frac{1}{2\eta}\|x - v\|^2 \right\}$$

> **Intuition:** $\text{prox}_h(v)$ is the point that minimizes $h$ while staying close to $v$ — a "proximal step" for the non-smooth part. For $h = \lambda\|\cdot\|_1$, it is soft thresholding: shrink each coordinate toward zero by $\lambda$ (and set it to zero if it was already small). For $h = \delta_\mathcal{C}$ (indicator of a convex set $\mathcal{C}$), it is projection onto $\mathcal{C}$. The proximal operator handles non-smoothness exactly in closed form, which is why proximal methods recover the fast $O(1/T)$ convergence rate.

This is a regularized minimization of $h$: find the point that minimizes $h$ while staying close to $v$, with $\eta$ controlling the trade-off. The proximal operator always exists and is unique when $h$ is closed and convex.

**Geometric interpretation.** Think of the proximal operator as a "gradient step" for the non-smooth function $h$. For smooth $h$, the gradient step $v - \eta \nabla h(v)$ minimizes the linearization $h(v) + \nabla h(v)^\top(x-v) + \frac{1}{2\eta}\|x-v\|^2$, which is exactly the proximal problem with $h$ replaced by its linearization. The proximal operator solves the exact problem instead — more expensive but more accurate.

**Key examples:**

**L1 norm (soft thresholding).** For $h(x) = \lambda\|x\|_1$:

$$[\text{prox}_{\eta\lambda\|\cdot\|_1}(v)]_i = \text{sign}(v_i)\max(|v_i| - \eta\lambda, 0) =: \mathcal{S}_{\eta\lambda}(v_i)$$

> **Remember:** Soft thresholding: $(\text{prox}_{\lambda\|\cdot\|_1}(v))_i = \text{sign}(v_i) \cdot \max(|v_i| - \lambda, 0)$. Shrink each coordinate toward zero by $\lambda$; set it to exactly zero if $|v_i| \leq \lambda$. This is the core operation in LASSO, sparse coding, and sparse autoencoders — the exact sparsity (hard zero) arises from the non-smooth kink in $|x|$ at zero, not from any post-hoc rounding.

The soft-thresholding operator $\mathcal{S}_{\tau}(v)$ shrinks $v$ toward zero by $\tau$: large components are shrunk, components with $|v_i| \leq \tau$ are set to zero. This is the exact solution because the $\ell_1$ proximal problem separates across coordinates.

**Indicator function (projection).** For $h(x) = \delta_\mathcal{C}(x)$ (0 if $x \in \mathcal{C}$, $+\infty$ otherwise):

$$\text{prox}_{\eta\delta_\mathcal{C}}(v) = \arg\min_{x \in \mathcal{C}} \frac{1}{2}\|x - v\|^2 = \Pi_\mathcal{C}(v)$$

The proximal operator of an indicator function is the **projection** onto the set $\mathcal{C}$. Thus, projected gradient descent is a special case of the proximal gradient method.

**Nuclear norm.** For $h(X) = \lambda\|X\|_*$ (sum of singular values), $\text{prox}_{\eta h}(M) = U\,\text{diag}(\mathcal{S}_{\eta\lambda}(\sigma))\,V^\top$ where $M = U\,\text{diag}(\sigma)\,V^\top$ is the SVD of the input. This is used in matrix completion and low-rank learning.

> **Key insight:** Proximal methods extend gradient descent to non-smooth objectives without sacrificing convergence guarantees — the proximal operator handles the non-smooth part exactly, recovering the $O(1/T)$ convergence rate of smooth gradient descent.

## Proximal Gradient Descent (ISTA)

For $f(x) = g(x) + h(x)$ with $g$ smooth ($L$-smooth) and $h$ convex (proximal-friendly), the **proximal gradient method** (also called ISTA — Iterative Shrinkage-Thresholding Algorithm — when $h = \ell_1$ norm) is:

$$x_{t+1} = \text{prox}_{\eta h}(x_t - \eta \nabla g(x_t))$$

**Algorithm interpretation.** First take a gradient step with respect to $g$ (the smooth part), then apply the proximal operator of $h$ (the non-smooth part). This is the "forward-backward splitting": forward step for $g$, backward (implicit) step for $h$.

> **Refresher:** ISTA = gradient step on the smooth part $g$ + proximal step on the non-smooth part $h$. If you tried plain gradient descent on $f = g + h$ with non-smooth $h$, you would need subgradients of $h$, losing the benefit of $L$-smoothness of $g$ and falling back to the slower $O(1/\sqrt{T})$ subgradient rate. By handling $h$ exactly via the proximal operator, ISTA restores the $O(1/T)$ convergence rate that $L$-smoothness of $g$ alone would give.

**Derivation.** Proximal gradient minimizes the following upper bound on $f$ at each step:

$$Q(x; x_t) = g(x_t) + \nabla g(x_t)^\top(x - x_t) + \frac{1}{2\eta}\|x - x_t\|^2 + h(x)$$

Minimizing $Q$ over $x$: the gradient term and quadratic combine into a proximal problem, giving exactly the proximal gradient update.

**Convergence.** With step size $\eta = 1/L$:
- $h$ convex, $g$ convex + $L$-smooth: $f(x_T) - f(x^*) = O(1/T)$
- $h$ convex, $g$ $\mu$-strongly convex + $L$-smooth: linear convergence $O(\rho^T)$ with $\rho = 1 - \mu/L$

**ISTA for LASSO.** Set $g(x) = \frac{1}{2}\|Ax - b\|^2$ (smooth, $L = \|A\|^2$) and $h(x) = \lambda\|x\|_1$ (non-smooth). The ISTA update is:

$$x_{t+1} = \mathcal{S}_{\eta\lambda}(x_t - \eta A^\top(Ax_t - b))$$

This is extremely efficient: each iteration requires one matrix-vector product and one soft-thresholding operation.

## FISTA: Momentum-Accelerated Proximal Gradient

**FISTA** (Beck & Teboulle, 2009) adds Nesterov momentum to ISTA, achieving $O(1/T^2)$ convergence — a quadratic improvement over ISTA's $O(1/T)$.

**Algorithm.**

$$y_{t+1} = x_t + \frac{t-1}{t+2}(x_t - x_{t-1})$$
$$x_{t+1} = \text{prox}_{\eta h}(y_{t+1} - \eta \nabla g(y_{t+1}))$$

The momentum term $\frac{t-1}{t+2} \to 1$ as $t \to \infty$, with specific scheduling ensuring the accelerated rate. In practice, Beck and Teboulle use the exact recurrence $t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2$ with momentum $\gamma_k = (t_k - 1)/t_{k+1}$, as implemented in the code below. The simplified $\frac{t-1}{t+2}$ formula is an approximation that gives the same $O(1/T^2)$ rate but slightly worse constants.

**Convergence.** With step size $\eta = 1/L$:

$$f(x_T) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{(T+1)^2} = O(1/T^2)$$

This is the **optimal** rate for first-order methods on smooth convex functions (matching Nesterov's lower bound). FISTA is the go-to algorithm for LASSO and similar composite optimization problems in signal processing and ML.

> **Key insight:** FISTA's $O(1/T^2)$ rate is optimal for first-order methods on composite convex problems — Nesterov's acceleration doubles the convergence exponent from $O(1/T)$ to $O(1/T^2)$ without increasing per-iteration cost.

## ADMM: Alternating Direction Method of Multipliers

**ADMM** (Boyd et al., 2011) solves problems of the form:

$$\min_{x, z} \; f(x) + g(z) \quad \text{subject to} \quad Ax + Bz = c$$

by splitting variables and alternating minimization. The augmented Lagrangian is:

$$L_\rho(x, z, \lambda) = f(x) + g(z) + \lambda^\top(Ax + Bz - c) + \frac{\rho}{2}\|Ax + Bz - c\|^2$$

**ADMM iteration:**

1. **x-update:** $x^{k+1} = \arg\min_x L_\rho(x, z^k, \lambda^k)$
2. **z-update:** $z^{k+1} = \arg\min_z L_\rho(x^{k+1}, z, \lambda^k)$
3. **Dual update:** $\lambda^{k+1} = \lambda^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$

The key advantage: when $f$ and $g$ are separable or have structure that makes their individual minimization tractable, ADMM provides a practical algorithm for the coupled problem.

> **Intuition:** ADMM splits "minimize $f(x) + g(z)$ subject to $Ax + Bz = c$" into alternating minimizations over $x$ and $z$, then a dual variable update. Each subproblem is often trivial in isolation — for LASSO, the $x$-update is ridge regression (closed-form linear solve) and the $z$-update is soft thresholding. ADMM coordinates them via the dual variable $\lambda$, which accumulates the constraint violation and penalizes it. The joint problem, which would be hard to solve directly, becomes easy when split this way.

**Convergence.** ADMM converges (for convex $f, g$ and appropriate $\rho > 0$) at $O(1/T)$ rate for the primal and dual residuals. The $O(1/T)$ rate requires at least one of $f$ or $g$ to be strongly convex. For general convex $f, g$, ADMM is guaranteed to converge to a primal-dual optimal pair, but the rate may be slower and depends on the penalty parameter $\rho$. The penalty parameter controls the trade-off between convergence speed and solution quality; adaptive $\rho$ schemes often work better in practice.

**LASSO via ADMM.** Reformulate LASSO $\min_x \frac{1}{2}\|Ax-b\|^2 + \lambda\|x\|_1$ as:

$$\min_{x,z} \; \frac{1}{2}\|Ax-b\|^2 + \lambda\|z\|_1 \quad \text{s.t.} \quad x = z$$

- x-update: $x \leftarrow (A^\top A + \rho I)^{-1}(A^\top b + \rho z - \lambda^k)$ — a ridge regression problem with closed form
- z-update: $z \leftarrow \mathcal{S}_{\lambda/\rho}(x + \lambda^k/\rho)$ — soft thresholding

**Distributed optimization.** ADMM naturally decomposes into parallel subproblems. For large-scale ML problems (e.g., training across many machines), each node handles a subset of data/parameters for the x-update, and ADMM's structure allows coordination via the shared dual variable $\lambda$.

## Projected Gradient Descent

For constrained optimization $\min f(x)$ subject to $x \in \mathcal{C}$ (convex $\mathcal{C}$), **projected gradient descent** performs:

$$x_{t+1} = \Pi_\mathcal{C}(x_t - \eta \nabla f(x_t))$$

where $\Pi_\mathcal{C}(v) = \arg\min_{x \in \mathcal{C}} \|x - v\|^2$ is the Euclidean projection. This is the proximal gradient method with $h = \delta_\mathcal{C}$.

**Projection formulas for common constraint sets:**

- **$\ell_2$ ball** $\mathcal{C} = \{x : \|x\| \leq r\}$: $\Pi_\mathcal{C}(v) = r \cdot v / \max(\|v\|, r)$
- **$\ell_\infty$ ball** $\mathcal{C} = \{x : \|x\|_\infty \leq r\}$: $[\Pi_\mathcal{C}(v)]_i = \text{clip}(v_i, -r, r)$
- **Probability simplex** $\mathcal{C} = \{x : x \geq 0, \mathbf{1}^\top x = 1\}$: sort $v$ in descending order, find threshold $\tau$, output $\max(v_i - \tau, 0)$. Costs $O(d \log d)$

**ML applications.** Spectral normalization for GANs constrains weight matrices to have spectral norm at most 1 — implemented by projecting the weight matrix to the unit spectral norm ball via rescaling. Fair learning with demographic parity constraints uses projected gradient to stay feasible.

## Frank-Wolfe (Conditional Gradient)

**Frank-Wolfe** (1956) is a projection-free method for $\min_{x \in \mathcal{C}} f(x)$. Instead of projecting after a gradient step, it solves a **linear program** over $\mathcal{C}$ at each iteration:

1. **Linear minimization oracle (LMO):** $s_t = \arg\min_{s \in \mathcal{C}} \nabla f(x_t)^\top s$
2. **Convex combination step:** $x_{t+1} = (1-\gamma_t) x_t + \gamma_t s_t$ with $\gamma_t = 2/(t+2)$

> **Intuition:** Instead of projecting onto the feasible set $\mathcal{C}$ after a gradient step (which can require an expensive computation like a full SVD), Frank-Wolfe linearizes $f$ at the current point and solves a linear program over $\mathcal{C}$. For the nuclear-norm ball, the LP oracle only needs the leading singular vector pair — a single power iteration — instead of a full SVD projection. Frank-Wolfe also naturally produces low-rank or sparse iterates because it moves toward extreme points of $\mathcal{C}$.

**Why avoid projection?** For many constraint sets, projection is expensive ($O(d^3)$ for matrix constraints) while solving the linear program is cheap. Examples:
- **Nuclear norm ball** $\{X : \|X\|_* \leq \tau\}$: LMO is $s = -\tau u_1 v_1^\top$ where $(u_1, v_1)$ is the leading singular vector pair — one power iteration instead of a full SVD projection
- **Spectral norm ball** $\{W : \|W\|_2 \leq 1\}$: LMO requires only the leading singular vectors
- **Flow polytope** for optimal transport: LMO is a shortest-path computation

**Convergence.** Frank-Wolfe achieves $O(1/T)$ convergence for smooth convex objectives on bounded domains. The convergence is in terms of the **Frank-Wolfe gap**: $\langle \nabla f(x_t), x_t - s_t \rangle$, which is always an upper bound on $f(x_t) - f(x^*)$ and is efficiently computable at each step.

**Sparsity property.** After $T$ iterations, the Frank-Wolfe iterate is a convex combination of $T$ extreme points of $\mathcal{C}$. For the nuclear norm ball, this means the solution has rank at most $T$ — Frank-Wolfe naturally produces low-rank solutions.

## ML Connections

**LASSO and feature selection.** The LASSO problem $\min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda\|\beta\|_1$ is the cornerstone of sparse learning. ISTA/FISTA solve it efficiently, and the soft-thresholding structure directly connects to the proximal operator. In transformers, sparse autoencoders used for mechanistic interpretability are trained with $\ell_1$ losses using these methods.

**Constrained fine-tuning.** Modern fine-tuning methods (parameter-efficient fine-tuning, LoRA) can be formulated as constrained optimization: $\min_{\Delta W} L(W_0 + \Delta W)$ subject to $\text{rank}(\Delta W) \leq r$ or $\|\Delta W\|_F \leq \rho$. Frank-Wolfe and projected gradient methods provide efficient algorithms for these constraints.

**Spectral normalization.** For GANs, the discriminator is constrained to be 1-Lipschitz by normalizing weight matrices: $W \leftarrow W / \sigma_{\max}(W)$. This is exactly the projection onto the unit spectral norm ball — a projected gradient step for the constrained training problem.

**Attention with simplex constraints.** Sparsemax (Martins & Astudillo, 2016) replaces softmax in attention with a projection onto the probability simplex, producing sparse attention distributions. This uses the $O(d \log d)$ simplex projection algorithm.

## Python: ISTA and FISTA for LASSO

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

np.random.seed(42)

# Generate LASSO problem: sparse signal recovery
m, n = 50, 100  # underdetermined: more unknowns than measurements
A = np.random.randn(m, n) / np.sqrt(m)
x_sparse = np.zeros(n)
x_sparse[:10] = np.random.randn(10) * 3  # 10 non-zero entries
b = A @ x_sparse + 0.05 * np.random.randn(m)
lam = 0.1

# Lipschitz constant L = ||A||^2
L = np.linalg.norm(A, ord=2)**2

def soft_threshold(v, tau):
    """Soft thresholding operator."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def lasso_obj(x):
    return 0.5 * np.sum((A @ x - b)**2) + lam * np.sum(np.abs(x))

def gradient_g(x):
    """Gradient of smooth part g(x) = 0.5*||Ax - b||^2."""
    return A.T @ (A @ x - b)

# ---- ISTA ----
def ista(x_init, T=500):
    x = x_init.copy()
    losses = [lasso_obj(x)]
    eta = 1.0 / L

    for _ in range(T):
        grad = gradient_g(x)
        x = soft_threshold(x - eta * grad, eta * lam)
        losses.append(lasso_obj(x))

    return x, losses

# ---- FISTA (Beck-Teboulle) ----
def fista(x_init, T=500):
    x = x_init.copy()
    y = x.copy()
    t_k = 1.0
    losses = [lasso_obj(x)]
    eta = 1.0 / L

    for _ in range(T):
        # Gradient step at momentum point y
        x_new = soft_threshold(y - eta * gradient_g(y), eta * lam)
        # Nesterov momentum update
        t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        gamma = (t_k - 1) / t_k_new
        y = x_new + gamma * (x_new - x)
        x = x_new
        t_k = t_k_new
        losses.append(lasso_obj(x))

    return x, losses

# ---- Subgradient descent for comparison ----
def subgradient(x_init, T=500, c=0.5):
    x = x_init.copy()
    x_best = x.copy()
    f_best = lasso_obj(x)
    losses = [f_best]

    for t in range(1, T + 1):
        eta = c / np.sqrt(t)
        g = gradient_g(x) + lam * np.sign(x)
        x = x - eta * g
        f_val = lasso_obj(x)
        if f_val < f_best:
            f_best = f_val
            x_best = x.copy()
        losses.append(f_best)

    return x_best, losses

x0 = np.zeros(n)
x_ista, losses_ista = ista(x0, T=500)
x_fista, losses_fista = fista(x0, T=500)
x_sgd_prox, losses_sgd_prox = subgradient(x0, T=500, c=0.3)

f_opt = min(losses_fista[-1], losses_ista[-1])  # FISTA typically finds optimal

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convergence comparison
gap_ista = np.array(losses_ista) - f_opt + 1e-12
gap_fista = np.array(losses_fista) - f_opt + 1e-12
gap_sgd = np.array(losses_sgd_prox) - f_opt + 1e-12

t_arr = np.arange(1, 502)
axes[0].loglog(t_arr, gap_ista, label='ISTA O(1/t)', linewidth=2)
axes[0].loglog(t_arr, gap_fista, label='FISTA O(1/t²)', linewidth=2)
axes[0].loglog(t_arr, gap_sgd, label='Subgradient O(1/√t)', linewidth=2, linestyle='--')
# Reference lines
axes[0].loglog(t_arr, 1.0/t_arr, 'k--', alpha=0.4, label='O(1/t)')
axes[0].loglog(t_arr, 1.0/t_arr**2, 'k:', alpha=0.4, label='O(1/t²)')
axes[0].set_xlabel('Iteration t')
axes[0].set_ylabel('f(x_t) − f* + ε')
axes[0].set_title('ISTA vs FISTA vs Subgradient\n(log-log scale, theory slopes shown)')
axes[0].legend(fontsize=8)
axes[0].grid(True)

# Sparsity pattern evolution for FISTA
sparsity_history = []
x = np.zeros(n)
y = x.copy()
t_k = 1.0
eta = 1.0 / L
for step in range(200):
    x_new = soft_threshold(y - eta * gradient_g(y), eta * lam)
    t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
    gamma = (t_k - 1) / t_k_new
    y = x_new + gamma * (x_new - x)
    x = x_new
    t_k = t_k_new
    if step % 20 == 0:
        sparsity_history.append((step, np.sum(np.abs(x) > 0.01), x.copy()))

# Show sparsity at start and end
axes[1].stem(range(n), x_sparse, linefmt='C0-', markerfmt='C0o', basefmt='k-',
             label='True signal (10 non-zeros)')
axes[1].stem(range(n), x_fista, linefmt='C1-', markerfmt='C1x', basefmt='k-',
             label=f'FISTA recovery ({np.sum(np.abs(x_fista) > 0.01)} non-zeros)')
axes[1].set_xlabel('Coefficient index')
axes[1].set_ylabel('Value')
axes[1].set_title('Sparse Signal Recovery via FISTA')
axes[1].legend(fontsize=8)
axes[1].grid(True)

# Evolution of number of non-zeros during FISTA
x_track = np.zeros(n)
y_track = x_track.copy()
t_k = 1.0
nnz_history = []
for step in range(500):
    x_new = soft_threshold(y_track - eta * gradient_g(y_track), eta * lam)
    t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
    gamma = (t_k - 1) / t_k_new
    y_track = x_new + gamma * (x_new - x_track)
    x_track = x_new
    t_k = t_k_new
    nnz_history.append(np.sum(np.abs(x_track) > 0.01))

axes[2].plot(nnz_history, linewidth=2)
axes[2].axhline(10, color='r', linestyle='--', label='True sparsity (10)')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Number of non-zeros')
axes[2].set_title('Sparsity Pattern Evolution During FISTA')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('proximal_methods.png', dpi=150)
plt.show()

# Recovery quality
print(f"True support: {np.where(np.abs(x_sparse) > 0.01)[0].tolist()}")
print(f"ISTA support: {np.where(np.abs(x_ista) > 0.01)[0].tolist()}")
print(f"FISTA support: {np.where(np.abs(x_fista) > 0.01)[0].tolist()}")
print(f"\nReconstruction errors:")
print(f"  ISTA:  ||x_ista  - x_true||₂ = {np.linalg.norm(x_ista - x_sparse):.4f}")
print(f"  FISTA: ||x_fista - x_true||₂ = {np.linalg.norm(x_fista - x_sparse):.4f}")
print(f"  FISTA objective: {lasso_obj(x_fista):.6f}")
print(f"  ISTA  objective: {lasso_obj(x_ista):.6f}")
```

:::quiz
question: "The proximal operator prox_{ηh}(v) = argmin_x {h(x) + (1/2η)||x − v||²} for h(x) = λ||x||₁ yields soft thresholding: sign(v_i)·max(|v_i| − ηλ, 0). Why do small-magnitude coordinates become exactly zero (hard sparsity) rather than being merely shrunk toward zero?"
options:
  - "Soft thresholding sets small coordinates to zero to improve numerical stability in floating point arithmetic"
  - "The ℓ₁ norm has non-differentiable kinks at zero; the subdifferential ∂|x| = [-1,1] at x=0 allows the optimality condition 0 ∈ ∂(h + quadratic) to hold at x=0 for |v_i| ≤ ηλ"
  - "The quadratic penalty (1/2η)||x−v||² is stronger than the ℓ₁ term for small x, forcing coordinates below the threshold to zero"
  - "ISTA sets small coordinates to zero by convention to produce sparse solutions, not as a consequence of the optimization"
correct: 1
explanation: "The proximal problem for coordinate i is: min_{x_i} {λ|x_i| + (1/2η)(x_i − v_i)²}. Optimality requires 0 ∈ ∂_x{λ|x_i| + (1/2η)(x_i − v_i)²}. For x_i = 0: the subdifferential is λ[-1,1] + (1/η)(0 − v_i) = [-λ − v_i/η, λ − v_i/η]. This contains 0 iff −λ ≤ v_i/η ≤ λ, i.e., |v_i| ≤ ηλ. So x_i = 0 is optimal exactly when |v_i| ≤ ηλ — hard sparsity is a direct consequence of the subdifferential of |x| containing an interval at zero, not a design choice."
:::

:::quiz
question: "FISTA achieves O(1/T²) convergence while ISTA achieves O(1/T). What is the role of the Nesterov momentum coefficient γ_k = (t_{k-1} − 1)/t_k in FISTA, and why is it critical that t_k follows the specific update t_{k+1} = (1 + √(1 + 4t_k²))/2?"
options:
  - "The momentum coefficient controls the step size; the specific update ensures the step size decreases at the rate 1/T²"
  - "The momentum performs an extrapolation step; the specific t_k sequence ensures the quantity t_k²(f(x_k) − f*) is non-increasing, which is needed to prove the O(1/T²) bound via a Lyapunov argument"
  - "The momentum prevents the algorithm from cycling; the t_k update ensures strong convexity is satisfied at each step"
  - "The momentum averages consecutive iterates; the specific t_k sequence minimizes the condition number of the proximal subproblem"
correct: 1
explanation: "The key to FISTA's proof is a Lyapunov function of the form Φ_k = t_k²(f(x_k) − f*) + C·||z_k − x*||². Beck & Teboulle show that Φ_{k+1} ≤ Φ_k when t_k satisfies t_{k+1}² − t_{k+1} ≤ t_k², which is guaranteed by the specific update rule. This non-increasing Lyapunov function telescopes to give t_T² · (f(x_T) − f*) ≤ Φ_0 = O(||x_0 − x*||²), and since t_T grows as O(T), we get f(x_T) − f* = O(1/T²). The specific t_k formula is not magic — it is chosen to make the Lyapunov decrease condition hold."
:::

:::quiz
question: "Frank-Wolfe (Conditional Gradient) avoids projection by solving a linear minimization oracle (LMO) over the feasible set at each step. For the nuclear norm ball {X : ||X||_* ≤ τ}, why is the LMO cheaper than projection, and what sparsity structure do Frank-Wolfe iterates naturally have?"
options:
  - "The LMO for the nuclear norm ball requires a full SVD (O(min(m,n)·mn) cost), which is equal to projection cost; iterates are dense matrices"
  - "The LMO requires only the leading singular vectors (s = −τu₁v₁ᵀ, O(mn) with power iteration), much cheaper than projection which requires a full SVD; iterates are rank-T matrices after T steps"
  - "The LMO for nuclear norm is a convex program, solved by interior-point methods in O(n³); iterates are low-rank due to the nuclear norm constraint"
  - "Frank-Wolfe and projected gradient have the same computational cost for the nuclear norm ball; the difference is only in convergence speed"
correct: 1
explanation: "For the nuclear norm ball, the LMO is s_t = argmin_{||S||_* ≤ τ} tr(∇f(X_t)ᵀ S) = −τ u₁ v₁ᵀ, where u₁, v₁ are the leading left and right singular vectors of ∇f(X_t). Computing only the leading singular vector pair costs O(mn) with power iteration (versus O(min(m,n)·mn) for a full SVD needed for projection). Each Frank-Wolfe step adds one rank-1 matrix s_t to the iterate; after T steps, X_T = Σ γ_k s_k is a sum of T rank-1 matrices, so rank(X_T) ≤ T. This automatic low-rank structure is valuable for matrix completion and low-rank factorization problems where the optimal solution is known to be low-rank."
:::
