---
title: "The Kantorovich Problem & Wasserstein Distances"
estimatedMinutes: 35
tags: ["optimal-transport", "Wasserstein", "Kantorovich", "couplings", "earth-mover"]
prerequisites: ["l1-convex-analysis from m4"]
---

## The Monge Problem: Moving Mass Optimally

The story of optimal transport begins with an eighteenth-century problem posed by Gaspard Monge (1781): given a distribution of sand $\mu$ on $\mathcal{X}$ and a distribution of holes $\nu$ on $\mathcal{Y}$, find the most efficient way to move the sand into the holes. Formally, let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$ be a cost function — say $c(x,y) = \|x - y\|^2$ for transporting a grain from location $x$ to location $y$. A **transport map** is a measurable function $T : \mathcal{X} \to \mathcal{Y}$ such that pushing $\mu$ forward through $T$ recovers $\nu$:

$$T_\# \mu = \nu \qquad \text{meaning} \quad \nu(B) = \mu(T^{-1}(B)) \text{ for all measurable } B$$

The Monge problem asks for the map minimizing total transport cost:

$$\inf_{T : T_\# \mu = \nu} \int_{\mathcal{X}} c(x, T(x)) \, d\mu(x)$$

> **Intuition:** Think of $\mu$ as a probability distribution over source locations and $\nu$ as a probability distribution over target locations. The map $T$ assigns to each source point $x$ a destination $T(x)$. The constraint $T_\# \mu = \nu$ says that mass is conserved: the amount of source mass mapping into any region equals the target mass of that region. We minimize the average cost of transport.

**The existence problem.** The Monge problem is seriously ill-posed. Transport maps need not exist. Consider $\mu = \delta_{x_0}$ (a single point mass) and $\nu = \frac{1}{2}\delta_{y_0} + \frac{1}{2}\delta_{y_1}$ (mass split between two points). No function $T$ can push $\delta_{x_0}$ to $\nu$: a function maps one point to one point and cannot split mass. This is a fundamental obstruction: maps cannot split mass, but optimal transport might need to.

Even when transport maps exist, finding the infimum is a hard non-convex problem. Monge's formulation languished for nearly two centuries before Kantorovich's decisive reformulation.

## The Kantorovich Relaxation

Leonid Kantorovich (1942) solved both problems simultaneously by replacing transport maps with transport **plans** — joint probability distributions over source-target pairs.

**Definition (coupling).** A coupling of $\mu$ and $\nu$ is a probability measure $\pi$ on $\mathcal{X} \times \mathcal{Y}$ whose marginals are $\mu$ and $\nu$:

$$\pi(A \times \mathcal{Y}) = \mu(A), \qquad \pi(\mathcal{X} \times B) = \nu(B)$$

for all measurable $A \subseteq \mathcal{X}$ and $B \subseteq \mathcal{Y}$. Denote the set of all couplings $\Pi(\mu, \nu)$.

The **Kantorovich problem** minimizes expected transport cost over all couplings:

$$\text{OT}(\mu, \nu) = \min_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) \, d\pi(x, y)$$

> **Key insight:** The Kantorovich relaxation turns the hard non-convex Monge problem into a convex optimization problem. The set $\Pi(\mu,\nu)$ is convex (a linear constraint set), and the objective is linear in $\pi$. This is why optimal transport is tractable: it is, at its core, a linear program.

**Why couplings generalize maps.** Any transport map $T$ induces a coupling $\pi_T = (\text{Id}, T)_\# \mu$, the joint distribution of $(x, T(x))$ when $x \sim \mu$. Couplings that arise from maps are special: they are supported on the graph of $T$ and assign zero probability to "split" mass. The Kantorovich problem optimizes over all couplings — including those that split mass — so it always has a solution (by weak compactness of the set of probability measures with fixed marginals).

**Existence and uniqueness.** Under mild continuity conditions on $c$, the minimum in the Kantorovich problem is attained. The minimizer $\pi^*$ is called the **optimal coupling** or **optimal transport plan**. In general it is not unique, but for the quadratic cost and absolutely continuous $\mu$, it is (Brenier's theorem, Lesson 2).

## Wasserstein-$p$ Distances

For the special case $c(x,y) = \|x-y\|^p$ with $p \geq 1$, the Kantorovich problem defines the **Wasserstein-$p$ distance**:

$$W_p(\mu, \nu) = \left( \min_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} \|x - y\|^p \, d\pi(x,y) \right)^{1/p}$$

**Theorem (Wasserstein metric).** $W_p$ is a metric on the space $\mathcal{P}_p(\mathcal{X})$ of probability measures with finite $p$-th moments:

$$\mathcal{P}_p(\mathcal{X}) = \left\{ \mu : \int \|x\|^p \, d\mu(x) < \infty \right\}$$

The metric properties are:
- *Non-negativity:* $W_p(\mu,\nu) \geq 0$ with equality iff $\mu = \nu$.
- *Symmetry:* $W_p(\mu,\nu) = W_p(\nu,\mu)$ (take $\pi^*(x,y) \mapsto \pi^*(y,x)$).
- *Triangle inequality:* $W_p(\mu,\rho) \leq W_p(\mu,\nu) + W_p(\nu,\rho)$ (gluing lemma).

**$W_2$ and Riemannian structure.** The Wasserstein-2 distance is the most important in machine learning. It metrizes weak convergence plus convergence of second moments — a strictly weaker topology than total variation (TV convergence implies $W_p$ convergence, but not vice versa; $W_p$ permits sequences with shifting supports to converge). The space $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ admits a formal Riemannian structure (Otto calculus) with geodesics, gradients, and curvature that makes it the natural setting for analyzing diffusion models, generative flows, and distribution evolution.

> **Remember:** $W_p(\mu,\nu) = \left(\min_{\pi \in \Pi(\mu,\nu)} \mathbb{E}_{(x,y)\sim\pi}[\|x-y\|^p]\right)^{1/p}$ — the $p$-th root of the minimum expected $p$-th power displacement under the optimal coupling.

**Curse of dimensionality.** The empirical Wasserstein distance between two distributions with $n$ i.i.d. samples converges at rate $O(n^{-1/d})$ in $d$ dimensions — exponentially slow in high dimensions. In $d = 2$, you need $O(n)$ samples; in $d = 100$, you need $O(n^{-0.01})$ which is essentially useless. This explains why entropic regularization (Sinkhorn), sliced Wasserstein, and other approximations are not mere computational shortcuts — they are necessities for high-dimensional ML applications.

> **Intuition:** The $O(n^{-1/d})$ convergence rate means doubling the samples only improves the estimate by a factor of $2^{1/d}$. In $d = 100$, you would need $10^{30}$ samples to match the accuracy of $10^3$ samples in 1D. This is why Sinkhorn, sliced Wasserstein, and other approximations are essential for high-dimensional ML — they are not computational shortcuts, they are survival tools.

**1D closed form.** For distributions on $\mathbb{R}$, the Wasserstein-$p$ distance admits a closed form via quantile functions. Let $F$ and $G$ be the CDFs of $\mu$ and $\nu$ with quantile (inverse CDF) functions $F^{-1}$ and $G^{-1}$. Then:

$$W_p^p(\mu, \nu) = \int_0^1 |F^{-1}(t) - G^{-1}(t)|^p \, dt$$

This follows because the optimal coupling in 1D always sorts the mass monotonically: the $t$-th quantile of $\mu$ is matched to the $t$-th quantile of $\nu$. In higher dimensions, no such simple formula exists, and we need the full LP machinery.

## $W_1$ = Earth Mover's Distance and the Kantorovich-Rubinstein Dual

The Wasserstein-1 distance has the most intuitive interpretation. Think of $\mu$ as a pile of dirt and $\nu$ as the target configuration. $W_1(\mu,\nu)$ is the minimum total work (mass times distance) needed to rearrange $\mu$ into $\nu$ — hence the name **earth mover's distance** (EMD).

**Kantorovich-Rubinstein duality.** For $W_1$, the dual problem has a particularly clean form:

$$W_1(\mu, \nu) = \sup_{\|f\|_{\text{Lip}} \leq 1} \left( \mathbb{E}_{x \sim \mu}[f(x)] - \mathbb{E}_{y \sim \nu}[f(y)] \right)$$

where $\|f\|_{\text{Lip}} = \sup_{x \neq y} |f(x) - f(y)| / \|x-y\|$ is the Lipschitz constant.

> **Key insight:** The dual formulation of $W_1$ makes it estimable from samples using a neural network as the witness function $f$. Instead of solving the LP, parameterize $f$ as a neural network, enforce the Lipschitz constraint, and maximize the empirical expectation difference. This is precisely the Wasserstein GAN.

**Proof sketch of duality.** By LP strong duality (Fenchel-Rockafellar), the dual of minimizing $\langle C, \pi \rangle$ over the marginal constraints is a maximization over dual variables $(u,v)$ with $u(x) + v(y) \leq c(x,y) = \|x-y\|$. Setting $v(y) = -u(y)$ and using $\|u\|_{\text{Lip}} \leq 1$ recovers the Kantorovich-Rubinstein form. The function $u^c(y) = \inf_x \{ \|x-y\| - u(x) \}$ that appears in Lesson 2's c-transform theory is exactly the optimal $v$ here: the dual constraint $u(x) + v(y) \leq \|x-y\|$ is tightest when $v = u^c$.

## Discrete Optimal Transport as a Linear Program

When $\mu = \sum_{i=1}^m r_i \delta_{x_i}$ and $\nu = \sum_{j=1}^n c_j \delta_{y_j}$ are discrete measures with weights $r \in \Delta^m$ and $c \in \Delta^n$ (simplices), the Kantorovich problem reduces to a finite-dimensional LP.

**Setup.** Define the $m \times n$ cost matrix $C_{ij} = c(x_i, y_j)$ and the transport matrix $P \in \mathbb{R}^{m \times n}_{\geq 0}$ where $P_{ij}$ is the mass transported from $x_i$ to $y_j$. The problem is:

$$\min_{P \geq 0} \sum_{i,j} C_{ij} P_{ij} \quad \text{subject to} \quad \sum_j P_{ij} = r_i \; \forall i, \quad \sum_i P_{ij} = c_j \; \forall j$$

In compact notation: $\min_{P \in \mathbf{U}(r,c)} \langle C, P \rangle$ where the transport polytope is:

$$\mathbf{U}(r,c) = \{ P \in \mathbb{R}^{m \times n}_{\geq 0} : P\mathbf{1}_n = r, \; P^\top \mathbf{1}_m = c \}$$

> **Refresher:** This is a standard linear program: linear objective, linear equality/inequality constraints. The feasible set $\mathbf{U}(r,c)$ is a polytope (bounded, since all entries are bounded by $\min(r_i,c_j)$). The LP always has a solution at a vertex of the polytope. The network simplex algorithm solves it in $O(n^3 \log n)$ time.

**Connection to the assignment problem.** When $r = c = \mathbf{1}_n / n$ (uniform distributions over $n$ points each), the transport matrix $P$ is a doubly stochastic matrix scaled by $1/n$. By Birkhoff's theorem, the vertices of the doubly stochastic polytope are permutation matrices. The optimal transport plan at a vertex is a deterministic assignment — an $n$-to-$n$ matching. This is the linear assignment problem, solvable by the Hungarian algorithm in $O(n^3)$.

> **Remember:** The Kantorovich LP always has a solution (the transport polytope is compact). The solution is sparse: at most $m+n-1$ nonzero entries despite the full $m \times n$ matrix. This sparsity is the LP vertex structure — exactly what makes network simplex fast in practice.

**Sparsity.** An optimal LP solution (vertex of $\mathbf{U}(r,c)$) has at most $m+n-1$ nonzero entries. For $n = 1000$ points, the transport plan is nearly empty — most mass moves along only $\sim 2000$ edges — even though the full matrix has $10^6$ entries.

## ML Connection: Wasserstein GAN

Standard GANs train the generator $G$ to minimize the Jensen-Shannon divergence between the generated distribution $p_G$ and the data distribution $p_{\text{data}}$. The JS divergence has a catastrophic failure mode: when the two distributions have disjoint supports (common early in training), $D_{\text{KL}}(p_G \| p_{\text{data}}) = \infty$ and $\text{JS}(p_G, p_{\text{data}}) = \log 2$ — constant! The generator receives zero gradient.

**WGAN** (Arjovsky et al., 2017) replaces JS divergence with $W_1$ using the Kantorovich-Rubinstein dual. The training objective is:

$$\min_G \max_{D : \|D\|_{\text{Lip}} \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

The discriminator $D$ (called a "critic") approximates the 1-Lipschitz witness function $f$ from the dual. Its output is no longer a probability — it is an unbounded real number estimating how much more "real" a point is than "fake" in the Wasserstein sense.

**Why it works.** For non-overlapping distributions, $W_1$ is still finite and informative — it measures the distance you would have to move one distribution to reach the other. The gradient of $W_1$ with respect to $G$'s parameters is always well-defined (under mild conditions), eliminating the vanishing gradient problem.

**Lipschitz enforcement.** The original WGAN used weight clipping ($|w| \leq 0.01$) to enforce Lipschitz constraint — crude but functional. WGAN-GP (Gulrajani et al., 2017) uses a gradient penalty:

$$\lambda \cdot \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]$$

where $\hat{x}$ is interpolated between real and generated samples: $\hat{x} = \alpha x + (1-\alpha) G(z)$ with $\alpha \sim \text{Uniform}[0,1]$. Spectral normalization (Miyato et al., 2018) provides an efficient alternative by normalizing each weight matrix by its largest singular value.

## Python: Discrete OT via Linear Programming

```python
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# 1. Discrete OT as an LP using scipy.optimize.linprog
# ──────────────────────────────────────────────────────────────────────────────

def discrete_ot(r, c, C):
    """
    Solve discrete OT: min_{P in U(r,c)} <C, P>
    via scipy.optimize.linprog.

    Parameters
    ----------
    r : array of shape (m,) — source marginal (sums to 1)
    c : array of shape (n,) — target marginal (sums to 1)
    C : array of shape (m, n) — cost matrix

    Returns
    -------
    P : array of shape (m, n) — optimal transport plan
    cost : float — optimal transport cost
    """
    m, n = C.shape
    # Flatten P into a vector of length m*n
    c_obj = C.flatten()  # objective coefficients

    # Equality constraints: row sums = r, column sums = c
    # A_eq @ p = b_eq
    A_eq = np.zeros((m + n, m * n))
    # Row constraints: sum_j P_ij = r_i for each i
    for i in range(m):
        A_eq[i, i * n:(i + 1) * n] = 1.0
    # Column constraints: sum_i P_ij = c_j for each j
    for j in range(n):
        A_eq[m + j, j::n] = 1.0

    b_eq = np.concatenate([r, c])

    # Bounds: P_ij >= 0
    bounds = [(0, None)] * (m * n)

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    P = result.x.reshape(m, n)
    return P, result.fun


# ──────────────────────────────────────────────────────────────────────────────
# 2. Example: transport between two 1D distributions
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

# Source: mixture of two Gaussians; Target: single Gaussian
n_points = 8
x_src = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
x_tgt = np.array([-1.8, -0.9, -0.1,  0.4, 0.8, 1.2, 1.7, 2.2])

# Uniform weights
r = np.ones(n_points) / n_points
c = np.ones(n_points) / n_points

# L1 cost matrix (for W1)
C_l1 = np.abs(x_src[:, None] - x_tgt[None, :])

P_opt, ot_cost = discrete_ot(r, c, C_l1)

print(f"Optimal W1 cost (discrete): {ot_cost:.4f}")

# Compare to L2 distance between means (does not account for distribution shape)
l2_mean = np.abs(x_src.mean() - x_tgt.mean())
print(f"L2 distance between means:  {l2_mean:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. W1 via quantile function (1D closed form, for verification)
# ──────────────────────────────────────────────────────────────────────────────

def w1_1d(samples_mu, samples_nu):
    """W1 in 1D = integral |F^{-1}(t) - G^{-1}(t)| dt = mean |sorted_mu - sorted_nu|."""
    return np.mean(np.abs(np.sort(samples_mu) - np.sort(samples_nu)))

# Dense 1D comparison
n_dense = 500
mu_samples = rng.normal(-1.0, 0.5, n_dense)
nu_samples = rng.normal(+1.0, 0.5, n_dense)

w1_quantile = w1_1d(mu_samples, nu_samples)
print(f"W1 via quantile formula (1D): {w1_quantile:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Visualize the transport plan
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: transport plan matrix
ax = axes[0]
im = ax.imshow(P_opt * n_points, cmap='Blues', aspect='auto',
               vmin=0, vmax=1.5)
ax.set_xlabel('Target index $j$', fontsize=12)
ax.set_ylabel('Source index $i$', fontsize=12)
ax.set_title('Optimal Transport Plan $P^*$ (scaled by $n$)', fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Right: transport arrows on the real line
ax = axes[1]
ax.scatter(x_src, np.zeros(n_points) - 0.05, s=80, color='steelblue',
           zorder=5, label='Source $\\mu$')
ax.scatter(x_tgt, np.zeros(n_points) + 0.05, s=80, color='coral',
           zorder=5, label='Target $\\nu$')

# Draw arrows weighted by transport mass
for i in range(n_points):
    for j in range(n_points):
        if P_opt[i, j] > 1e-6:
            ax.annotate('', xy=(x_tgt[j], 0.05),
                        xytext=(x_src[i], -0.05),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=P_opt[i, j] * n_points * 2))

ax.set_xlim(-3, 3)
ax.set_ylim(-0.25, 0.25)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_yticks([])
ax.set_xlabel('Position on $\\mathbb{R}$', fontsize=12)
ax.set_title('Transport Plan: Arrows = Mass Flow', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('ot_transport_plan.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nTransport plan sparsity: {np.sum(P_opt > 1e-6)} nonzero entries "
      f"(max possible: {n_points + n_points - 1})")
```

:::quiz
question: "Why does the Monge transport problem fail to have a solution when $\\mu = \\delta_{x_0}$ and $\\nu = \\frac{1}{2}\\delta_{y_0} + \\frac{1}{2}\\delta_{y_1}$?"
options:
  - "The cost function is not continuous at point masses."
  - "A map $T$ cannot split the mass at $x_0$ into two destinations, but $\\nu$ requires mass at two points."
  - "The optimal cost is infinite for any choice of $T$."
  - "The feasible set $\\{T : T_\\# \\mu = \\nu\\}$ is not compact."
correct: 1
explanation: "A function $T : \\mathcal{X} \\to \\mathcal{Y}$ assigns exactly one output to each input, so $T(x_0)$ must be a single point. The pushforward $T_\\# \\delta_{x_0} = \\delta_{T(x_0)}$ is always a Dirac mass — it cannot equal the two-point mixture $\\frac{1}{2}\\delta_{y_0} + \\frac{1}{2}\\delta_{y_1}$. The Kantorovich relaxation resolves this by allowing the coupling $\\pi$ to split mass: we can have $\\pi(\\{x_0\\} \\times \\{y_0\\}) = \\frac{1}{2}$ and $\\pi(\\{x_0\\} \\times \\{y_1\\}) = \\frac{1}{2}$."
:::

:::quiz
question: "The Kantorovich-Rubinstein duality states $W_1(\\mu,\\nu) = \\sup_{\\|f\\|_{\\text{Lip}} \\leq 1} (\\mathbb{E}_\\mu[f] - \\mathbb{E}_\\nu[f])$. How does this formula underlie the Wasserstein GAN discriminator?"
options:
  - "The discriminator minimizes the Lipschitz constant of its gradient to enforce the constraint."
  - "The discriminator is trained to be a 1-Lipschitz function that maximizes the difference in expected values between real and generated distributions, thereby estimating $W_1$."
  - "The formula shows that $W_1$ is a lower bound on the JS divergence, motivating a switch of loss functions."
  - "The generator minimizes $\\sup_f (\\mathbb{E}_\\mu[f] - \\mathbb{E}_\\nu[f])$ directly by back-propagating through the sup."
correct: 1
explanation: "The Kantorovich-Rubinstein dual recasts $W_1$ as the maximum difference in expected value under a 1-Lipschitz function. In WGAN, the discriminator (critic) $D_\\phi$ parameterizes this function: training it to maximize $\\mathbb{E}_{x\\sim p_{\\text{data}}}[D_\\phi(x)] - \\mathbb{E}_{z}[D_\\phi(G(z))]$ subject to $\\|D_\\phi\\|_{\\text{Lip}} \\leq 1$ provides an estimate of $W_1(p_{\\text{data}}, p_G)$. The generator then minimizes this estimate."
:::

:::quiz
question: "For discrete measures $\\mu = \\sum_i r_i \\delta_{x_i}$ and $\\nu = \\sum_j c_j \\delta_{y_j}$ with $m = n = 1000$ support points each, the optimal transport plan $P^*$ is a vertex of the transport polytope $\\mathbf{U}(r,c)$. What is the maximum number of nonzero entries in $P^*$?"
options:
  - "Exactly $n^2 = 10^6$, since the full matrix must be specified."
  - "Exactly $n = 1000$, one per source point."
  - "At most $m + n - 1 = 1999$, by LP vertex structure of the transport polytope."
  - "At most $\\min(m,n) = 1000$, since the plan must be a permutation matrix when $r = c$."
correct: 2
explanation: "A basic feasible solution of an LP with $mn$ variables and $m+n$ equality constraints (minus 1 for redundancy) has at most $m+n-1$ nonzero variables. The transport polytope $\\mathbf{U}(r,c)$ has exactly $m+n-1$ linearly independent equality constraints (the sum of all row constraints equals 1, same for column constraints, so there is one redundancy). Therefore every vertex — and hence every optimal solution of a non-degenerate instance — has at most $m+n-1 = 1999$ nonzero entries, despite the full matrix having $10^6$ entries."
:::
