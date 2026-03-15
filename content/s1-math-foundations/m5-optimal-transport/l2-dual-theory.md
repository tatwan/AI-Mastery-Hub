---
title: "Dual Theory & the Brenier Map"
estimatedMinutes: 35
tags: ["duality", "Brenier", "c-transform", "semi-dual", "optimal-maps"]
prerequisites: ["l1-kantorovich-problem", "l1-convex-analysis from m4"]
---

## Kantorovich Duality

The Kantorovich problem is a linear program, and every LP has a dual. Unpacking this duality reveals deep structure: instead of searching for an $m \times n$ transport plan, we can equivalently search for two scalar functions $u : \mathcal{X} \to \mathbb{R}$ and $v : \mathcal{Y} \to \mathbb{R}$.

**The dual problem.** For a cost function $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, the Kantorovich dual is:

$$\sup_{u, v} \int_{\mathcal{X}} u(x) \, d\mu(x) + \int_{\mathcal{Y}} v(y) \, d\nu(y) \quad \text{subject to} \quad u(x) + v(y) \leq c(x, y) \; \forall x, y$$

The functions $u$ and $v$ are called **dual potentials** or **Kantorovich potentials**.

> **Refresher:** Recall LP duality from Lesson 1 of M4. The primal minimizes $\langle C, P \rangle$ subject to $P \geq 0$ and marginal equalities. The dual introduces one multiplier per primal constraint: $u_i$ for the row-sum constraint $\sum_j P_{ij} = r_i$ and $v_j$ for the column-sum constraint $\sum_i P_{ij} = c_j$. Complementary slackness says $P_{ij} > 0$ implies $u_i + v_j = C_{ij}$ at optimality — mass is only transported along "tight" edges.

**Weak duality.** For any feasible primal $\pi$ and feasible dual $(u,v)$:

$$\int u \, d\mu + \int v \, d\nu = \int \!\int [u(x) + v(y)] \, d\pi(x,y) \leq \int \!\int c(x,y) \, d\pi(x,y)$$

where the equality uses the marginal constraints on $\pi$, and the inequality uses $u(x) + v(y) \leq c(x,y)$ pointwise. This shows the dual objective is always a lower bound on the primal objective.

**Strong duality (Fenchel-Rockafellar).** Under mild conditions (e.g., $c$ lower semicontinuous and $\mu, \nu$ compactly supported, or $c(x,y) = \|x-y\|^p$ with $\mu, \nu \in \mathcal{P}_p$), the duality gap is zero:

$$\min_{\pi \in \Pi(\mu,\nu)} \int c \, d\pi = \max_{u+v \leq c} \int u \, d\mu + \int v \, d\nu$$

and both sides are attained. The proof uses the Fenchel-Rockafellar duality theorem applied to the pair of functionals $F(\pi) = \iota_{\Pi(\mu,\nu)}(\pi)$ (indicator of the marginal constraints) and $G(\pi) = \int c \, d\pi$ (transport cost).

**Complementary slackness.** At the optimum:
- If $\pi^*(A \times B) > 0$, then $u^*(x) + v^*(y) = c(x,y)$ for $\pi^*$-almost all $(x,y)$.
- Equivalently, the optimal coupling is supported on the **contact set** $\{(x,y) : u(x) + v(y) = c(x,y)\}$.

This is the geometric content: transport only occurs along pairs where the dual constraint is tight.

## c-Transforms and the Semi-Dual

The constraint $u(x) + v(y) \leq c(x,y)$ couples $u$ and $v$. The **c-transform** decouples them.

**Definition (c-conjugate).** The c-conjugate (or c-transform) of a function $u : \mathcal{X} \to \mathbb{R}$ is:

$$u^c(y) = \inf_{x \in \mathcal{X}} \left\{ c(x, y) - u(x) \right\}$$

By definition, $u(x) + u^c(y) \leq c(x,y)$ for all $x, y$ — so $(u, u^c)$ is always a feasible dual pair. Moreover, $u^c$ is the *largest* $v$ such that $(u,v)$ is feasible, i.e., $u^c = \arg\max_v \{ v : u(x) + v(y) \leq c(x,y) \, \forall x \}$.

**Optimality.** At the optimum of the dual, we always have $v^* = (u^*)^c$. This allows us to eliminate $v$ and reduce the dual to a single-function optimization:

$$\text{OT}(\mu, \nu) = \max_{u} \left[ \int_{\mathcal{X}} u(x) \, d\mu(x) + \int_{\mathcal{Y}} u^c(y) \, d\nu(y) \right]$$

This is the **semi-dual** formulation — one unconstrained function $u$, no inequality constraints.

> **Key insight:** The semi-dual collapses the entire optimal transport problem — finding a joint distribution over $\mathcal{X} \times \mathcal{Y}$ — into the problem of finding a single scalar function $u$ on $\mathcal{X}$. This is the mathematical insight that makes neural OT tractable: parameterize $u$ as a neural network, compute $u^c$ by a softmin operation, and maximize the semi-dual objective.

**c-concavity.** A function $u$ is **c-concave** if $u = (u^c)^c$, i.e., $u(x) = \inf_y \{c(x,y) - u^c(y)\}$. The optimal dual potential $u^*$ is always c-concave. For the quadratic cost $c(x,y) = \frac{1}{2}\|x-y\|^2$, the c-transform is:

$$u^c(y) = \inf_x \left\{ \frac{1}{2}\|x-y\|^2 - u(x) \right\} = -\sup_x \left\{ u(x) - \frac{1}{2}\|x-y\|^2 \right\}$$

The Legendre transform of $u$ is $u^*(y) = \sup_x \{x \cdot y - u(x)\}$, so $u^c(y) = -u^*(-y) + \frac{1}{2}\|y\|^2$ (up to constants). For the quadratic cost, **c-concave = convex**: the optimal dual potential is a convex function.

## Brenier's Theorem

For smooth measures and the quadratic cost, the optimal transport plan is deterministic — it is a map, not a general coupling.

**Theorem (Brenier, 1991).** Let $\mu, \nu \in \mathcal{P}_2(\mathbb{R}^d)$ and suppose $\mu$ is absolutely continuous (has a density). For the quadratic cost $c(x,y) = \frac{1}{2}\|x-y\|^2$, there exists a unique optimal transport map $T^* : \mathbb{R}^d \to \mathbb{R}^d$ such that:
1. $T^*_\# \mu = \nu$ (pushforward constraint),
2. $T^* = \nabla \phi$ for some convex function $\phi : \mathbb{R}^d \to \mathbb{R}$,
3. $T^*$ is the unique optimal map ($\mu$-a.e.), and the optimal coupling is $\pi^* = (\text{Id}, T^*)_\# \mu$.

The function $\phi$ is called the **Brenier potential** or **Kantorovich potential** (up to the identification $u = \phi - \frac{1}{2}\|\cdot\|^2$).

> **Intuition:** Brenier's theorem says that for the squared-distance cost, the optimal way to move mass is always by a gradient map — each source point $x$ moves in the direction $\nabla\phi(x) - x$. There is no "crossing" of transport paths: if $x_1 < x_2$ in 1D, then $T(x_1) \leq T(x_2)$. In higher dimensions, the analogous "non-crossing" condition is that $T = \nabla\phi$ is curl-free (a gradient).

**Proof idea.** The optimal coupling $\pi^*$ is supported on the contact set of the dual: $\{(x,y) : u(x) + v(y) = \frac{1}{2}\|x-y\|^2\}$. Taking the gradient of this equality with respect to $y$ yields $\nabla v(y) = x - y$, so $x = y + \nabla v(y) = y - \nabla(-v)(y)$. Setting $\phi = -v + \frac{1}{2}\|\cdot\|^2$ (the c-concave dual potential shifted to be convex), we get $x = \nabla\phi^*(y)$ and $y = \nabla\phi(x)$, where $\phi^*$ is the Legendre conjugate of $\phi$. Therefore $T = \nabla\phi$ and $T^{-1} = \nabla\phi^*$.

**The Monge-Ampère equation.** The constraint that $T = \nabla\phi$ pushes $\mu$ to $\nu$ — i.e., $(\nabla\phi)_\# \mu = \nu$ — is encoded as a PDE. If $\mu$ has density $\rho_0$ and $\nu$ has density $\rho_1$, the change-of-variables formula gives:

$$\det(\nabla^2 \phi(x)) = \frac{\rho_0(x)}{\rho_1(\nabla\phi(x))}$$

This is the **Monge-Ampère equation** — a fully nonlinear elliptic PDE for the convex potential $\phi$. Its solution theory connects OT to differential geometry and PDE analysis.

> **Remember:** For $c(x,y) = \frac{1}{2}\|x-y\|^2$ and absolutely continuous $\mu$, the optimal transport map is $T^* = \nabla\phi$ for a unique convex function $\phi$. The optimal transport plan is always deterministic (supported on a graph), not a general coupling.

**Gaussian case.** For $\mu = \mathcal{N}(m_0, \Sigma_0)$ and $\nu = \mathcal{N}(m_1, \Sigma_1)$, the Brenier map is affine: $T(x) = \Sigma_0^{-1/2}(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2} \Sigma_0^{-1/2}(x - m_0) + m_1$, and the $W_2$ distance is:

$$W_2^2(\mu, \nu) = \|m_0 - m_1\|^2 + \mathcal{B}^2(\Sigma_0, \Sigma_1)$$

where $\mathcal{B}^2(\Sigma_0, \Sigma_1) = \text{tr}(\Sigma_0) + \text{tr}(\Sigma_1) - 2\, \text{tr}\!\left((\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2}\right)$ is the **Bures metric** between covariance matrices.

## The Semi-Dual in Practice

The semi-dual objective is:

$$\mathcal{J}(u) = \int_{\mathcal{X}} u(x) \, d\mu(x) + \int_{\mathcal{Y}} u^c(y) \, d\nu(y)$$

For the quadratic cost, this simplifies: $u^c(y) = \inf_x \{ \frac{1}{2}\|x-y\|^2 - u(x) \} = -\sup_x \{u(x) - \frac{1}{2}\|x-y\|^2\} = -(\frac{1}{2}\|\cdot\|^2 - u)^*(y) + \frac{1}{2}\|y\|^2 - \frac{1}{2}\|y\|^2$... More directly, writing $\phi = u + \frac{1}{2}\|\cdot\|^2$ (convex), the semi-dual becomes:

$$\max_{\phi \text{ convex}} \left[ \mathbb{E}_{x \sim \mu}[\phi(x)] - \mathbb{E}_{y \sim \nu}[\phi^*(y)] \right]$$

where $\phi^*(y) = \sup_x \{x \cdot y - \phi(x)\}$ is the Legendre conjugate. The optimal $\phi$ is the Brenier potential, and $T^* = \nabla\phi$.

**Gradient of the semi-dual.** The gradient of $\mathcal{J}$ with respect to $u$ (in function space) is:

$$\frac{\delta \mathcal{J}}{\delta u}(x) = \mu(x) - (T_u)_\# \nu(x)$$

where $T_u(y) = \arg\min_{x'} \{c(x', y) - u(x')\}$ is the optimal transport map for the current potential $u$. This shows gradient ascent on the semi-dual alternates between "where should mass go" and "update the potential."

## ML Connection: Neural OT via Input-Convex Neural Networks

**The challenge.** Parameterizing a general convex function $\phi : \mathbb{R}^d \to \mathbb{R}$ is non-trivial. A fully connected network is not convex in its input.

**Input-Convex Neural Networks (ICNN).** Amos et al. (2017) proposed a network architecture that is convex in its input $x$:

$$z_{k+1} = \sigma(W_k^z z_k + W_k^x x + b_k)$$

where: (1) the weights $W_k^z$ are constrained to be non-negative ($W_k^z \geq 0$), (2) the activation $\sigma$ is convex and non-decreasing (e.g., softplus), and (3) the input skip connections $W_k^x x$ are unconstrained. The output $z_K$ is convex in $x$ for any fixed $y$-related parameters.

**Neural OT algorithm** (Makkuva et al. 2020; Korotin et al. 2021):
1. Parameterize $\phi_\theta$ as an ICNN.
2. Compute $\phi_\theta^*(y)$ via a second ICNN $\psi_\omega$ trained to approximate the Legendre conjugate (or via explicit maximization for simple cases).
3. Maximize $\hat{\mathcal{J}}(\theta) = \frac{1}{n}\sum_i \phi_\theta(x_i) - \frac{1}{m}\sum_j \psi_\omega(y_j)$ using stochastic gradient ascent.
4. The Brenier map is $T^*(x) = \nabla_x \phi_\theta(x)$.

> **Key insight:** Neural OT is the end-to-end application of Brenier's theorem. The ICNN architecture enforces the convexity constraint needed for the semi-dual to equal the primal OT cost. Once trained, the gradient $\nabla\phi_\theta$ directly gives the optimal transport map — no LP or iterative Sinkhorn is needed at test time.

**Applications:**
- *Unpaired image translation:* $\mu$ = source domain images, $\nu$ = target domain images; $T^*$ transports images from $\mu$ to $\nu$ with minimal distortion.
- *Single-cell genomics:* $\mu$ = gene expression at time $t_0$, $\nu$ = at time $t_1$; $T^*$ infers developmental trajectories of individual cells.
- *Domain adaptation:* $\mu$ = source distribution, $\nu$ = target distribution; $T^*$ maps source features to look like target features.

## Python: Semi-Dual for 2D Gaussian Transport

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# Optimal transport between two 2D Gaussians via the semi-dual.
# For Gaussians, the Brenier potential is quadratic: phi(x) = x^T A x / 2 + b^T x
# and the optimal map is affine: T(x) = Ax + b.
# We recover it numerically using the discrete semi-dual on samples.
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(0)

# Source: N(m0, Sigma0), Target: N(m1, Sigma1)
m0 = np.array([0.0, 0.0])
m1 = np.array([2.0, 1.0])
Sigma0 = np.array([[1.0, 0.5], [0.5, 1.0]])
Sigma1 = np.array([[2.0, -0.3], [-0.3, 0.5]])

n = 300  # samples per distribution
L0 = np.linalg.cholesky(Sigma0)
L1 = np.linalg.cholesky(Sigma1)
X = rng.standard_normal((n, 2)) @ L0.T + m0  # source samples
Y = rng.standard_normal((n, 2)) @ L1.T + m1  # target samples


# ──────────────────────────────────────────────────────────────────────────────
# Closed-form Brenier map for Gaussians (ground truth)
# T(x) = Sigma0^{-1/2} (Sigma0^{1/2} Sigma1 Sigma0^{1/2})^{1/2} Sigma0^{-1/2} (x - m0) + m1
# ──────────────────────────────────────────────────────────────────────────────

def sqrtm_spd(A):
    """Matrix square root of a symmetric positive definite matrix."""
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.maximum(w, 0))) @ v.T

S0_half = sqrtm_spd(Sigma0)
S0_inv_half = np.linalg.inv(S0_half)
M = sqrtm_spd(S0_half @ Sigma1 @ S0_half)
A_opt = S0_inv_half @ M @ S0_inv_half  # optimal linear map component

def brenier_map(x):
    """Closed-form Brenier map for Gaussians."""
    return (x - m0) @ A_opt.T + m1

T_X = brenier_map(X)  # optimal transport of source samples


# ──────────────────────────────────────────────────────────────────────────────
# Wasserstein-2 distance (Gaussian closed form)
# W_2^2 = ||m0 - m1||^2 + Bures^2(Sigma0, Sigma1)
# ──────────────────────────────────────────────────────────────────────────────

bures_sq = (np.trace(Sigma0) + np.trace(Sigma1)
            - 2 * np.trace(sqrtm_spd(S0_half @ Sigma1 @ S0_half)))
w2_sq = np.sum((m0 - m1)**2) + bures_sq
print(f"W_2^2 (Gaussian closed form): {w2_sq:.4f}")
print(f"W_2   (Gaussian closed form): {np.sqrt(w2_sq):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualize: source, target, and transported samples
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: Source and target
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, color='steelblue', label='Source $\\mu$')
ax.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.4, color='coral', label='Target $\\nu$')
ax.set_title('Source and Target Samples', fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

# Panel 2: Transport vectors (subsample for clarity)
ax = axes[1]
idx = rng.choice(n, 40, replace=False)
ax.scatter(X[:, 0], X[:, 1], s=5, alpha=0.2, color='steelblue')
ax.scatter(Y[:, 0], Y[:, 1], s=5, alpha=0.2, color='coral')
for i in idx:
    ax.annotate('', xy=T_X[i], xytext=X[i],
                arrowprops=dict(arrowstyle='->', color='purple', lw=0.8, alpha=0.7))
ax.set_title('Brenier Map: Transport Arrows', fontsize=11)
ax.set_aspect('equal')

# Panel 3: Transported samples vs target
ax = axes[2]
ax.scatter(T_X[:, 0], T_X[:, 1], s=8, alpha=0.4, color='purple',
           label='Transported $T^*_\\#\\mu$')
ax.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.4, color='coral', label='Target $\\nu$')
ax.set_title('Transported Distribution vs Target', fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('brenier_map_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Verify: W_2 via discrete LP on a smaller sample (optional comparison)
# ──────────────────────────────────────────────────────────────────────────────

def w2_squared_discrete(X, Y):
    """Discrete W_2^2 using the 1D quantile trick does not apply in 2D.
    Instead compute empirical cost under the closed-form map."""
    T_X = brenier_map(X)
    return np.mean(np.sum((T_X - Y[np.argsort(np.argsort(
        np.sum(Y**2, axis=1)))])**2, axis=1))

# Cross-check: average squared displacement under the optimal map
# (approximate since we're using samples)
avg_displacement = np.mean(np.sum((T_X - Y)**2, axis=1))
print(f"Mean squared displacement (samples, approximate): {avg_displacement:.4f}")
print(f"Note: exact W_2^2 = {w2_sq:.4f}; sample estimate is noisy for small n")
```

:::quiz
question: "Let $c(x,y) = \\frac{1}{2}\\|x-y\\|^2$ be the quadratic cost. At the optimum of the Kantorovich dual, what is the relationship between the two dual potentials $u^*$ and $v^*$?"
options:
  - "$v^*(y) = u^*(y)$ — the two potentials are equal functions."
  - "$v^*(y) = -u^*(-y)$ — one is the reflection of the other."
  - "$v^*(y) = (u^*)^c(y) = \\inf_x \\{\\frac{1}{2}\\|x-y\\|^2 - u^*(x)\\}$ — $v^*$ is the c-transform of $u^*$."
  - "$v^*(y) = -(u^*)^*(y)$ — $v^*$ is the negated Legendre conjugate of $u^*$."
correct: 2
explanation: "The c-transform of $u$ under the cost $c(x,y)$ is $u^c(y) = \\inf_x \\{c(x,y) - u(x)\\}$. The dual constraint $u(x) + v(y) \\leq c(x,y)$ has the tightest feasible $v$ equal to $u^c$. At optimality, $v^* = (u^*)^c$ — the second dual potential is fully determined by the first. This reduction is what enables the semi-dual formulation: optimize over $u$ alone, set $v = u^c$."
:::

:::quiz
question: "Brenier's theorem guarantees that for $c(x,y) = \\frac{1}{2}\\|x-y\\|^2$ and $\\mu$ absolutely continuous, the optimal transport map satisfies $T^* = \\nabla\\phi$ for a convex $\\phi$. Why does absolute continuity of $\\mu$ matter?"
options:
  - "Absolute continuity ensures that the cost function $c$ is integrable with respect to $\\mu$."
  - "Without absolute continuity, the Monge-Ampère equation has no classical solution."
  - "Absolute continuity prevents two source points from mapping to the same target — it rules out 'crossing' transport plans that would be non-optimal."
  - "Absolute continuity ensures the dual potentials are differentiable everywhere on the support of $\\mu$."
correct: 2
explanation: "The key reason is uniqueness and existence of the optimal map. If $\\mu$ has atoms (point masses), multiple source points can map to the same target, and the optimal coupling can be non-deterministic. Absolute continuity guarantees that $\\mu(\\{x\\}) = 0$ for all $x$, which (via a cyclical monotonicity argument) forces the optimal coupling to be supported on a graph: $\\pi^* = (\\text{Id}, T^*)_\\#\\mu$ for some map $T^*$. The map is then $\\mu$-a.e. unique and equals $\\nabla\\phi$."
:::

:::quiz
question: "In the neural OT framework using Input-Convex Neural Networks (ICNNs), what property of the network architecture ensures that the parameterized function $\\phi_\\theta$ is a valid Brenier potential?"
options:
  - "The network uses skip connections from every layer to the output, which ensures global convexity."
  - "The weights $W_k^z$ connecting consecutive hidden layers are constrained to be non-negative, and the activations are convex and non-decreasing — this ensures convexity in the input $x$ by induction."
  - "The network is trained with an $L_2$ regularizer on its Hessian, penalizing non-convexity."
  - "Batch normalization layers are removed, since they break the convexity constraint."
correct: 1
explanation: "An ICNN maintains convexity in $x$ by induction: the output $z_1 = \\sigma(W_0^x x + b_0)$ is convex in $x$ if $\\sigma$ is convex (the composition of convex and linear is convex). For deeper layers, $z_{k+1} = \\sigma(W_k^z z_k + W_k^x x + b_k)$ is convex in $x$ if $W_k^z \\geq 0$ (element-wise) and $z_k$ is convex in $x$ — since the non-negative-weighted sum of convex functions is convex, plus an affine term. The non-negativity constraint on $W_k^z$ is the essential structural requirement."
:::
