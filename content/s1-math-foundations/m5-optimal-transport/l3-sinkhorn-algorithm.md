---
title: "Sinkhorn Algorithm & Entropic Regularization"
estimatedMinutes: 35
tags: ["Sinkhorn", "entropic-OT", "regularization", "scaling-algorithm", "log-sum-exp"]
prerequisites: ["l1-kantorovich-problem", "l2-dual-theory"]
---

## The Computational Bottleneck of Exact OT

The LP formulation of discrete OT is elegant but expensive. For $n$-point discrete measures, the LP has $n^2$ variables and $2n$ constraints. The best general LP solvers run in $O(n^3)$ time. The network simplex algorithm — specialized for transportation problems — runs in $O(n^3 \log n)$ worst case (though faster in practice). For the typical ML use case of $n = 10^4$ or $n = 10^5$ mini-batch samples, this is completely infeasible.

**Why mini-batch OT matters.** In generative modeling, we compare mini-batches of samples from the model distribution $p_\theta$ and the data distribution $p_{\text{data}}$. At each gradient step, we have $n = 64$ or $n = 256$ samples and need to compute $W_2$ (or a proxy) and its gradient with respect to $\theta$. The LP must be solved thousands of times per training run, and must be differentiable for backpropagation.

Exact LP solvers are not differentiable, not GPU-parallelizable, and too slow. We need a different approach.

## Entropic Regularization

Cuturi (2013) proposed adding an entropic regularization term to the Kantorovich problem:

$$\text{OT}_\varepsilon(\mu, \nu) = \min_{\pi \in \mathbf{U}(r,c)} \left[ \langle C, \pi \rangle - \varepsilon H(\pi) \right]$$

where $H(\pi) = -\sum_{i,j} \pi_{ij} \log \pi_{ij}$ is the **entropy** of the transport plan, and $\varepsilon > 0$ is the regularization strength.

> **Intuition:** Entropy $H(\pi)$ is maximized by the uniform distribution and penalized by concentration. Subtracting $\varepsilon H(\pi)$ from the objective adds a preference for "spread out" transport plans. When $\varepsilon \to 0$, the regularization vanishes and we recover exact OT. When $\varepsilon \to \infty$, the entropy term dominates and the solution approaches the product measure $\pi = r c^\top$ (independent coupling) — transport every grain to every hole uniformly.

**Properties of the regularized problem:**
- *Strict convexity:* The objective $\langle C, \pi \rangle - \varepsilon H(\pi)$ is strictly convex in $\pi$ (since $-H$ is strictly convex). This guarantees a **unique** solution $\pi_\varepsilon^*$ for every $\varepsilon > 0$.
- *Smoothness:* $\text{OT}_\varepsilon(\mu,\nu)$ is smooth (infinitely differentiable) in $\mu$ and $\nu$, unlike exact OT which is only Lipschitz.
- *Convergence:* As $\varepsilon \to 0$, $\pi_\varepsilon^* \to \pi^*$ (optimal coupling of exact OT) and $\text{OT}_\varepsilon(\mu,\nu) \to \text{OT}(\mu,\nu)$.

> **Intuition:** Entropic regularization replaces the "hard" linear program (sharp, sparse solution) with a "soft" strictly convex problem (smooth, dense solution). As ε → 0 the soft solution approaches the hard one. As ε → ∞ the solution approaches independent coupling — total spreading. The practical sweet spot is where the solution is close enough to true OT but numerically stable: typically ε ≈ median(C)/20.

> **Remember:** $\text{OT}_\varepsilon(\mu,\nu) = \min_{\pi \in \mathbf{U}(r,c)} \langle C, \pi \rangle - \varepsilon H(\pi)$, where $H(\pi) = -\sum_{ij} \pi_{ij} \log \pi_{ij}$. The regularization parameter $\varepsilon$ trades off transport cost (fidelity) against smoothness (spread).

## The Gibbs Kernel and Scaling Structure

The key structural insight is that the solution of the entropic OT problem has a special product form.

**Theorem (Gibbs kernel structure).** The unique optimal solution of $\text{OT}_\varepsilon$ is:

$$\pi_{ij}^\varepsilon = a_i \, K_{ij} \, b_j$$

where $K_{ij} = e^{-C_{ij}/\varepsilon}$ is the **Gibbs kernel** and $a \in \mathbb{R}^m_{>0}$, $b \in \mathbb{R}^n_{>0}$ are positive scaling vectors.

**Proof.** Writing the KKT conditions for the strictly convex problem: the Lagrangian is $\mathcal{L}(\pi, u, v) = \langle C, \pi \rangle - \varepsilon H(\pi) - u^\top(P\mathbf{1} - r) - v^\top(P^\top\mathbf{1} - c)$. Setting $\frac{\partial \mathcal{L}}{\partial \pi_{ij}} = 0$:

$$C_{ij} + \varepsilon(\log \pi_{ij} + 1) - u_i - v_j = 0$$

$$\Rightarrow \pi_{ij} = e^{(u_i - C_{ij}/\varepsilon + v_j)/\varepsilon - 1} = e^{u_i/\varepsilon} \cdot e^{-C_{ij}/\varepsilon} \cdot e^{v_j/\varepsilon} \cdot e^{-1}$$

Setting $a_i = e^{u_i/\varepsilon - 1/2}$ and $b_j = e^{v_j/\varepsilon - 1/2}$ gives $\pi_{ij} = a_i K_{ij} b_j$. $\square$

**The marginal constraints.** From $\sum_j \pi_{ij} = r_i$ and $\sum_i \pi_{ij} = c_j$:

$$a_i \sum_j K_{ij} b_j = r_i \quad \Rightarrow \quad a = r \oslash (Kb)$$

$$b_j \sum_i K_{ij} a_i = c_j \quad \Rightarrow \quad b = c \oslash (K^\top a)$$

where $\oslash$ denotes element-wise division. These two equations must hold simultaneously — this is the key tension that the Sinkhorn algorithm resolves.

## The Sinkhorn Algorithm

The Sinkhorn-Knopp algorithm (also called matrix scaling or iterative proportional fitting) solves the marginal constraints by alternating between them.

**Algorithm (Sinkhorn):**
1. Initialize $b^{(0)} = \mathbf{1}_n$ (or any positive vector).
2. For $t = 0, 1, 2, \ldots$:
   - $a^{(t+1)} = r \oslash (K b^{(t)})$
   - $b^{(t+1)} = c \oslash (K^\top a^{(t+1)})$
3. Output transport plan: $\pi^* = \text{diag}(a) K \text{diag}(b)$.

Each iteration requires two matrix-vector products $Kb$ and $K^\top a$, each costing $O(mn)$. For $n = m$, this is $O(n^2)$ per iteration — a dramatic improvement over the $O(n^3 \log n)$ exact LP.

> **Key insight:** Sinkhorn is matrix scaling: we want to scale the rows and columns of $K$ to achieve target marginals. Row normalization and column normalization alternate. The critical observation is that the Gibbs kernel $K$ is a fixed positive matrix, and only the diagonal scalings $a$ and $b$ change. The entire algorithm is two matrix-vector products per iteration — easily parallelizable on GPUs.

**Convergence.** The Sinkhorn iterates converge linearly to the optimal $(a^*, b^*)$. The convergence rate is $\kappa = \tanh(\frac{1}{4} \log \frac{\lambda_{\max}}{\lambda_{\min}})$ where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest entries of $K$ (Franklin and Lorenz, 1989, via Birkhoff-Hopf contraction on the Hilbert projective metric). For small $\varepsilon$, $K$ is very peaked (near-zero entries everywhere except near the OT support), and convergence slows. This is the fundamental tension: small $\varepsilon$ gives better approximation to true OT but slower Sinkhorn convergence.

**Practical guideline.** Set $\varepsilon \approx \text{median}(C) / 20$ for a good balance of accuracy and convergence speed. For most ML applications, 20–100 Sinkhorn iterations suffice.

> **Refresher:** Linear convergence means the error shrinks by a constant factor each iteration: $\|e^{(t)}\| \leq \kappa^t \|e^{(0)}\|$. The rate $\kappa < 1$ depends on the "peakedness" of the Gibbs kernel $K$. For large ε (diffuse $K$), $\kappa \approx 0$ and convergence is fast (a few iterations). For small ε (peaked $K$, near exact OT), $\kappa \to 1$ and many iterations are needed — the fundamental tradeoff.

## Log-Domain Sinkhorn

For small $\varepsilon$, the Gibbs kernel entries $K_{ij} = e^{-C_{ij}/\varepsilon}$ underflow to zero in floating point (since $C_{ij}/\varepsilon \gg 1$). The standard algorithm collapses numerically.

**Solution: work in log-domain.** Define the log-scaling vectors (dual potentials):

$$f_i = \varepsilon \log a_i, \qquad g_j = \varepsilon \log b_j$$

Then $\log \pi_{ij} = (f_i - C_{ij} + g_j)/\varepsilon$ and the transport plan is:

$$\pi_{ij} = \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)$$

The marginal constraints become:

$$f_i = \varepsilon \log r_i - \varepsilon \log \sum_j \exp\left(\frac{g_j - C_{ij}}{\varepsilon}\right) = \varepsilon \log r_i + \varepsilon \, \text{LSE}_{j}\!\left(\frac{g_j - C_{ij}}{\varepsilon}\right) \cdot (-1)$$

More cleanly, using the **softmin** operator $\text{smin}_\varepsilon(h)_i = -\varepsilon \log \sum_j e^{-h_{ij}/\varepsilon}$:

$$f_i \leftarrow \varepsilon \log r_i + \text{smin}_\varepsilon(C - g^\top)_i$$

$$g_j \leftarrow \varepsilon \log c_j - \text{smin}_\varepsilon(C^\top - f^\top)_j$$

These updates use log-sum-exp, which is numerically stable for any value of $C_{ij}/\varepsilon$.

> **Intuition:** The log-domain Sinkhorn computes the dual potentials $f, g$ directly instead of the scaling vectors $a = e^{f/\varepsilon}$, $b = e^{g/\varepsilon}$. Since $f$ and $g$ stay in a reasonable range even when $e^{f/\varepsilon}$ would overflow or underflow, the algorithm is numerically stable.

## Sinkhorn Divergence

A subtle problem with $\text{OT}_\varepsilon$ as a loss function: **it is not zero when $\mu = \nu$**. Even for identical distributions, the regularized OT cost $\text{OT}_\varepsilon(\mu, \mu) > 0$ because the entropy penalty forces the transport plan away from the identity map.

**Definition (Sinkhorn divergence).** The Sinkhorn divergence debiases $\text{OT}_\varepsilon$:

$$S_\varepsilon(\mu, \nu) = \text{OT}_\varepsilon(\mu, \nu) - \frac{1}{2}\text{OT}_\varepsilon(\mu, \mu) - \frac{1}{2}\text{OT}_\varepsilon(\nu, \nu)$$

**Properties:**
- *Positive definiteness:* $S_\varepsilon(\mu, \nu) \geq 0$ with equality iff $\mu = \nu$.
- *Symmetry:* $S_\varepsilon(\mu, \nu) = S_\varepsilon(\nu, \mu)$.
- *Convergence:* As $\varepsilon \to 0$, $S_\varepsilon(\mu,\nu) \to \text{OT}(\mu,\nu)$ (Wasserstein distance).
- *Metrization:* $S_\varepsilon$ metrizes the same weak convergence topology as $W_p$ for all $\varepsilon > 0$.

> **Remember:** The Sinkhorn divergence is the bias-corrected version of $\text{OT}_\varepsilon$: $S_\varepsilon(\mu,\nu) = \text{OT}_\varepsilon(\mu,\nu) - \frac{1}{2}\text{OT}_\varepsilon(\mu,\mu) - \frac{1}{2}\text{OT}_\varepsilon(\nu,\nu) \geq 0$, with equality iff $\mu = \nu$.

**Why the correction works.** The bias $\text{OT}_\varepsilon(\mu,\mu) > 0$ comes from the regularization penalizing concentration. Both $\text{OT}_\varepsilon(\mu,\mu)$ and $\text{OT}_\varepsilon(\nu,\nu)$ contribute the same bias, and the cross term $\text{OT}_\varepsilon(\mu,\nu)$ combines both biases. The debiasing formula cancels them.

> **Key insight:** The Sinkhorn divergence $S_\varepsilon(\mu,\nu) = \text{OT}_\varepsilon(\mu,\nu) - \frac{1}{2}\text{OT}_\varepsilon(\mu,\mu) - \frac{1}{2}\text{OT}_\varepsilon(\nu,\nu)$ is positive definite and converges to $W_p$ as ε → 0. It combines the best of both worlds: the smoothness and GPU-friendliness of entropic OT, and the correct zero-when-equal property needed as a training loss. The three Sinkhorn runs (one cross-term, two self-terms) only triple the computational cost.

**Differentiability.** Both $\text{OT}_\varepsilon(\mu, \nu)$ and $S_\varepsilon(\mu, \nu)$ are differentiable with respect to the support points $\{x_i\}$ and weights $r$. The gradient $\nabla_{x_i} S_\varepsilon$ can be computed via the dual potentials:

$$\frac{\partial \text{OT}_\varepsilon}{\partial x_i} = r_i \nabla_{x_i} f_i^*$$

where $f_i^*$ is the converged log-domain Sinkhorn potential at the fixed point. This formula applies when Sinkhorn has converged; in practice, backpropagating through $T$ Sinkhorn iterations approximates this gradient, and the approximation quality improves with $T$.

## ML Connections

Sinkhorn and entropic OT are the key algorithmic enablers for using optimal transport in large-scale deep learning — they turn a cubic algorithm into a quadratic one that runs on GPUs.

- **Point Cloud Matching and Generation:** 3D point cloud generative models (PointFlow, ShapeGLO) use Sinkhorn distance as a training objective. The Sinkhorn loss between generated and real point clouds is differentiable (via the implicit function theorem on the scaling iterations), providing stable gradients for shape generation.
- **Sequence-to-Sequence Alignment:** Cross-lingual word alignment and machine translation evaluation (BERTScore, MoverScore) use Sinkhorn to compute soft alignment between token embeddings. The transport plan $\pi^\varepsilon$ provides a probabilistic word-to-word matching that accounts for semantic similarity.
- **Cell Trajectory Inference (scRNA-seq):** Waddington-OT uses Sinkhorn to interpolate between cell distributions at different time points in single-cell RNA sequencing data. The entropic regularization smooths the transport plan, preventing overfitting to the noisy observed cell distributions. This is a major ML application in computational biology.
- **Domain Adaptation with Sinkhorn:** The Sinkhorn distance provides a differentiable objective for domain adaptation: minimize $S_\varepsilon(\mu_\text{source}, \mu_\text{target})$ over the feature extractor parameters. Automatic Differentiation through Sinkhorn iterations (using `torch.autograd`) enables end-to-end training.
- **Fair Allocation in Federated Learning:** Entropic OT is used to fairly distribute data across federated clients: the transport plan $\pi^\varepsilon$ between the global data distribution and each client's local distribution determines how to balance the data. The entropy regularization ensures no client receives too homogeneous a subset.

> **Key insight:** Sinkhorn's two key properties — GPU parallelism (matrix operations on the Gibbs kernel $K_{ij} = e^{-C_{ij}/\varepsilon}$) and automatic differentiability (through the scaling iterations) — are what make OT practical in deep learning. Without entropic regularization, OT would remain a theoretical tool; with it, it becomes a trainable loss function.

## Python: Sinkhorn in Standard and Log-Domain

```python
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import linprog

# ──────────────────────────────────────────────────────────────────────────────
# 1. Sinkhorn algorithm (standard and log-domain)
# ──────────────────────────────────────────────────────────────────────────────

def sinkhorn_standard(r, c, C, eps, n_iter=100, return_history=False):
    """
    Standard Sinkhorn algorithm.
    r, c : marginals (arrays summing to 1)
    C    : cost matrix (m x n)
    eps  : regularization parameter
    Returns: transport plan P (m x n), cost, optional convergence history.
    """
    K = np.exp(-C / eps)
    b = np.ones(len(c))
    errs = []
    for _ in range(n_iter):
        a = r / (K @ b)
        b = c / (K.T @ a)
        if return_history:
            P_curr = a[:, None] * K * b[None, :]
            err = np.max(np.abs(P_curr.sum(axis=1) - r))
            errs.append(err)
    P = a[:, None] * K * b[None, :]
    cost = np.sum(C * P)
    return (P, cost, errs) if return_history else (P, cost)


def sinkhorn_log(r, c, C, eps, n_iter=100, return_history=False):
    """
    Log-domain Sinkhorn for numerical stability (for small eps).
    Uses log-sum-exp (softmin) updates on dual potentials f, g.
    """
    m, n = C.shape
    f = np.zeros(m)   # dual potentials (log-domain)
    g = np.zeros(n)
    log_r = np.log(r + 1e-300)
    log_c = np.log(c + 1e-300)
    errs = []

    for _ in range(n_iter):
        # f_i = eps * log(r_i) + softmin_j(C_ij - g_j) * eps
        # softmin_eps(h)_i = -eps * log sum_j exp(-h_j / eps)
        M_f = (g[None, :] - C) / eps           # shape (m, n)
        # Log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        # This prevents overflow/underflow when entries of log_K are very negative (large C/eps)
        f = eps * log_r + eps * np.log(np.sum(np.exp(M_f - M_f.max(axis=1, keepdims=True)), axis=1)) + M_f.max(axis=1)

        M_g = (f[:, None] - C) / eps           # shape (m, n)
        g = eps * log_c + eps * np.log(np.sum(np.exp(M_g - M_g.max(axis=0, keepdims=True)), axis=0)) + M_g.max(axis=0)

        if return_history:
            log_P = (f[:, None] + g[None, :] - C) / eps
            P_curr = np.exp(log_P)
            err = np.max(np.abs(P_curr.sum(axis=1) - r))
            errs.append(err)

    log_P = (f[:, None] + g[None, :] - C) / eps
    P = np.exp(log_P)
    cost = np.sum(C * P)
    return (P, cost, errs) if return_history else (P, cost)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Setup: two 1D distributions
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(7)
n = 50
# Source: bimodal mixture; Target: unimodal Gaussian
x_src = np.concatenate([rng.normal(-1.5, 0.4, n // 2),
                         rng.normal(+1.5, 0.4, n // 2)])
x_tgt = rng.normal(0.0, 1.0, n)
x_src.sort(); x_tgt.sort()

r = np.ones(n) / n
c = np.ones(n) / n
C = np.abs(x_src[:, None] - x_tgt[None, :])  # L1 cost for W1

# ──────────────────────────────────────────────────────────────────────────────
# 3. Convergence comparison: different eps values
# ──────────────────────────────────────────────────────────────────────────────

eps_values = [0.5, 0.1, 0.02]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
for eps, col in zip(eps_values, colors):
    _, _, errs = sinkhorn_log(r, c, C, eps=eps, n_iter=80, return_history=True)
    ax.semilogy(errs, color=col, label=f'$\\varepsilon = {eps}$', lw=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Marginal error (max abs)', fontsize=10)
ax.set_title('Sinkhorn Convergence (log-domain)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Transport plans for different eps
# ──────────────────────────────────────────────────────────────────────────────

ax = axes[1]
eps_plot = [1.0, 0.1, 0.01]
colors_plot = ['#9ecae1', '#3182bd', '#08306b']
for eps, col in zip(eps_plot, colors_plot):
    P, cost = sinkhorn_log(r, c, C, eps=eps, n_iter=200)
    # Show the "soft assignment" as an image row
    ax.imshow(P * n, cmap='Blues', aspect='auto', alpha=0.5)
ax.set_title(f'Transport Plan ($\\varepsilon = 0.1$, n={n})', fontsize=11)
P_show, _ = sinkhorn_log(r, c, C, eps=0.1, n_iter=200)
ax2 = axes[1]
ax2.imshow(P_show * n, cmap='Blues', aspect='auto')
ax2.set_xlabel('Target index $j$')
ax2.set_ylabel('Source index $i$')
ax2.set_title('Entropic Transport Plan ($\\varepsilon=0.1$)', fontsize=11)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Sinkhorn divergence: debiasing demonstration
# ──────────────────────────────────────────────────────────────────────────────

def sinkhorn_cost(r_a, x_a, r_b, x_b, eps, n_iter=200):
    C_ab = np.abs(x_a[:, None] - x_b[None, :])
    _, cost = sinkhorn_log(r_a, r_b, C_ab, eps=eps, n_iter=n_iter)
    return cost

# Vary eps and compute OT_eps, OT_eps(mu,mu), and S_eps
eps_range = np.logspace(-2, 1, 30)
ot_cross, ot_self_src, ot_self_tgt, sinkhorn_div = [], [], [], []

for eps in eps_range:
    ot_xy = sinkhorn_cost(r, x_src, c, x_tgt, eps)
    ot_xx = sinkhorn_cost(r, x_src, r, x_src, eps)
    ot_yy = sinkhorn_cost(c, x_tgt, c, x_tgt, eps)
    s_div = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
    ot_cross.append(ot_xy)
    ot_self_src.append(ot_xx)
    sinkhorn_div.append(s_div)

ax = axes[2]
ax.semilogx(eps_range, ot_cross, 'b-o', ms=4, label='$\\mathrm{OT}_\\varepsilon(\\mu,\\nu)$', lw=2)
ax.semilogx(eps_range, ot_self_src, 'r--s', ms=4, label='$\\mathrm{OT}_\\varepsilon(\\mu,\\mu)$', lw=2)
ax.semilogx(eps_range, sinkhorn_div, 'g-^', ms=4, label='$S_\\varepsilon(\\mu,\\nu)$', lw=2)
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel('Regularization $\\varepsilon$', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Sinkhorn Divergence Debiasing', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sinkhorn_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Speed comparison: Sinkhorn vs exact LP (small n)
# ──────────────────────────────────────────────────────────────────────────────

def exact_ot_lp(r, c, C):
    m, n = C.shape
    c_obj = C.flatten()
    A_eq = np.zeros((m + n, m * n))
    for i in range(m):
        A_eq[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([r, c])
    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (m * n),
                     method='highs')
    return result.fun

n_small = 30
x_s = np.sort(rng.normal(-1, 1, n_small))
x_t = np.sort(rng.normal(1, 1, n_small))
r_s = np.ones(n_small) / n_small
c_t = np.ones(n_small) / n_small
C_small = np.abs(x_s[:, None] - x_t[None, :])

t0 = time.perf_counter()
lp_cost = exact_ot_lp(r_s, c_t, C_small)
t_lp = time.perf_counter() - t0

t0 = time.perf_counter()
sink_cost, _ = sinkhorn_log(r_s, c_t, C_small, eps=0.05, n_iter=500)
t_sink = time.perf_counter() - t0

print(f"n = {n_small}")
print(f"Exact LP cost:       {lp_cost:.5f}  (time: {t_lp*1000:.1f} ms)")
print(f"Sinkhorn cost:       {sink_cost:.5f}  (time: {t_sink*1000:.1f} ms, eps=0.05)")
print(f"Relative error: {abs(lp_cost - sink_cost)/lp_cost * 100:.3f}%")
```

> **Note:** The `ot` library ([Python Optimal Transport](https://pythonot.github.io/)) provides production-quality implementations of Sinkhorn, Sinkhorn divergence, Gromov-Wasserstein, and sliced Wasserstein: `pip install POT`. In practice, use `ot.sinkhorn(a, b, M, reg)` and `ot.sliced_wasserstein_distance()` rather than the from-scratch implementations above. The from-scratch code here is for understanding the algorithm; POT is for building pipelines.

:::quiz
question: "The entropic OT problem adds $-\\varepsilon H(\\pi)$ to the Kantorovich objective. What is the closed-form structure of the unique optimal solution $\\pi^\\varepsilon$?"
options:
  - "$\\pi^\\varepsilon_{ij} = \\frac{r_i c_j}{n}$ — the product of marginals, independent of the cost."
  - "$\\pi^\\varepsilon_{ij} = a_i \\exp(-C_{ij}/\\varepsilon) b_j$ — a Gibbs kernel scaled by two positive diagonal vectors."
  - "$\\pi^\\varepsilon_{ij} = \\frac{e^{-C_{ij}/\\varepsilon}}{\\sum_{k,l} e^{-C_{kl}/\\varepsilon}}$ — the Gibbs distribution over all $(i,j)$ pairs."
  - "$\\pi^\\varepsilon_{ij} = \\mathbf{1}[i = \\arg\\min_k C_{kj}] \\cdot c_j$ — each target point receives mass from its nearest source."
correct: 1
explanation: "The KKT conditions of the strictly convex regularized problem yield $\\log \\pi_{ij} = (u_i - C_{ij} + v_j)/\\varepsilon - 1$. Setting $a_i = \\exp(u_i/\\varepsilon - 1/2)$ and $b_j = \\exp(v_j/\\varepsilon - 1/2)$, we get $\\pi^\\varepsilon_{ij} = a_i K_{ij} b_j$ where $K_{ij} = \\exp(-C_{ij}/\\varepsilon)$ is the Gibbs kernel. The vectors $a$ and $b$ are determined by the marginal constraints, and the Sinkhorn algorithm finds them by alternating normalization."
:::

:::quiz
question: "The standard Sinkhorn algorithm (operating on $K = e^{-C/\\varepsilon}$ directly) fails numerically for small $\\varepsilon$. Why, and how does the log-domain algorithm fix this?"
options:
  - "For small $\\varepsilon$, the Sinkhorn iterations do not converge; the log-domain algorithm uses a smaller step size."
  - "For small $\\varepsilon$, entries $K_{ij} = e^{-C_{ij}/\\varepsilon}$ underflow to zero in floating point, making $Kb = 0$ and causing division by zero; the log-domain algorithm maintains log-scaling potentials $f = \\varepsilon \\log a$, performing log-sum-exp operations that remain numerically stable."
  - "For small $\\varepsilon$, the transport plan is no longer unique, so the standard algorithm oscillates; the log-domain algorithm adds a small perturbation to ensure uniqueness."
  - "For small $\\varepsilon$, the Gibbs kernel $K$ is ill-conditioned; the log-domain algorithm uses a preconditioner to improve conditioning."
correct: 1
explanation: "When $\\varepsilon$ is small, $C_{ij}/\\varepsilon$ is large, causing $K_{ij} = e^{-C_{ij}/\\varepsilon}$ to underflow to machine zero (float64 underflows around $10^{-308}$, so $C_{ij}/\\varepsilon > 709$ causes underflow). The matrix $K$ becomes numerically zero, and $Kb = 0$, making $a = r \\oslash (Kb)$ undefined. The log-domain algorithm works with $f_i = \\varepsilon \\log a_i$ and $g_j = \\varepsilon \\log b_j$ directly, using log-sum-exp: $f_i \\leftarrow \\varepsilon \\log r_i + \\text{max}_j(g_j - C_{ij}) + \\varepsilon \\log \\sum_j \\exp((g_j - C_{ij} - \\text{max})/\\varepsilon)$. The log-sum-exp trick subtracts the maximum before exponentiating, avoiding overflow/underflow."
:::

:::quiz
question: "The Sinkhorn divergence is defined as $S_\\varepsilon(\\mu,\\nu) = \\mathrm{OT}_\\varepsilon(\\mu,\\nu) - \\frac{1}{2}\\mathrm{OT}_\\varepsilon(\\mu,\\mu) - \\frac{1}{2}\\mathrm{OT}_\\varepsilon(\\nu,\\nu)$. What property does $S_\\varepsilon$ have that $\\mathrm{OT}_\\varepsilon$ alone lacks, and why is this important for training generative models?"
options:
  - "$S_\\varepsilon$ is faster to compute than $\\mathrm{OT}_\\varepsilon$ because two of the three terms cancel analytically."
  - "$S_\\varepsilon(\\mu,\\nu) = 0$ if and only if $\\mu = \\nu$ (positive definiteness), whereas $\\mathrm{OT}_\\varepsilon(\\mu,\\mu) > 0$ for any $\\mu$. This ensures that $S_\\varepsilon = 0$ is a valid training target: a model achieving zero Sinkhorn divergence exactly matches the data distribution."
  - "$S_\\varepsilon$ is always equal to $\\mathrm{OT}(\\mu,\\nu)$ (the unregularized distance) regardless of $\\varepsilon$, making it an exact computation."
  - "$S_\\varepsilon$ has a closed-form gradient with respect to support points, whereas $\\mathrm{OT}_\\varepsilon$ requires backpropagating through the Sinkhorn loop."
correct: 1
explanation: "The bias $\\mathrm{OT}_\\varepsilon(\\mu,\\mu) > 0$ arises because entropic regularization spreads the transport plan away from the identity coupling even when transporting $\\mu$ to itself. This means minimizing $\\mathrm{OT}_\\varepsilon(p_\\theta, p_{\\text{data}})$ does not drive $p_\\theta$ to $p_{\\text{data}}$ — the minimum is achieved at some $p_\\theta \\neq p_{\\text{data}}$. The Sinkhorn divergence subtracts the self-transport biases, ensuring $S_\\varepsilon(p_\\theta, p_{\\text{data}}) = 0 \\Leftrightarrow p_\\theta = p_{\\text{data}}$. This makes it a valid (positive definite) loss function for generative modeling."
:::
