---
title: "Gradient Descent & Convergence Theory"
estimatedMinutes: 30
tags: ["gradient-descent", "convergence", "SGD", "Lipschitz", "strong-convexity", "PL-condition"]
prerequisites: ["l1-convex-analysis"]
---

## Gradient Descent: Algorithm and Geometry

**Gradient descent** is the workhorse of ML optimization. For an unconstrained minimization problem $\min_x f(x)$, the iteration is:

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

where $\eta > 0$ is the **step size** (learning rate). The negative gradient $-\nabla f(x_t)$ is the direction of steepest descent locally, but this local linearity is only a good approximation within a small neighborhood.

**Geometric interpretation.** Gradient descent minimizes a local linear model of $f$ subject to a quadratic (Euclidean) trust region:

$$x_{t+1} = \arg\min_{x} \left\{ f(x_t) + \nabla f(x_t)^\top(x - x_t) + \frac{1}{2\eta}\|x - x_t\|^2 \right\}$$

This proximal viewpoint is central: the $\frac{1}{2\eta}\|x - x_t\|^2$ term prevents moving too far from $x_t$ where the linear approximation is unreliable. Different choices of the "distance" metric lead to natural gradient descent, mirror descent, and proximal methods.

## L-Smoothness and the Quadratic Upper Bound

**Definition.** A differentiable function $f$ has $L$-**Lipschitz gradient** (is $L$-smooth) if:

$$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|, \quad \forall x, y$$

> **Intuition:** $L$-smoothness means the gradient cannot change too fast — the function is bounded above by a quadratic with curvature $L$. Practically, this gives you a "safe" step size: taking $\eta = 1/L$ guarantees every gradient step decreases the objective. If $L$ is large (steep curvature), you must use small steps; if $L$ is small (gentle curvature), you can take large steps safely.

This bounds how fast the gradient can change. Equivalently (for differentiable $f$):

$$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2 \quad \text{(Quadratic Upper Bound)}$$

The Hessian characterization: $f$ is $L$-smooth iff $\nabla^2 f(x) \preceq LI$ for all $x$.

**Descent lemma.** Setting $y = x_{t+1} = x_t - \eta \nabla f(x_t)$ in the quadratic upper bound:

$$f(x_{t+1}) \leq f(x_t) - \eta\|\nabla f(x_t)\|^2 + \frac{L\eta^2}{2}\|\nabla f(x_t)\|^2 = f(x_t) - \eta\!\left(1 - \frac{L\eta}{2}\right)\|\nabla f(x_t)\|^2$$

For $\eta \leq 1/L$, the factor $(1 - L\eta/2) \geq 1/2 > 0$, guaranteeing a decrease at every step. The optimal fixed step size is $\eta = 1/L$, giving:

$$f(x_{t+1}) \leq f(x_t) - \frac{1}{2L}\|\nabla f(x_t)\|^2$$

> **Key insight:** Step size $\eta = 1/L$ is the theoretically optimal fixed step size for $L$-smooth functions — it is exactly the reciprocal of the curvature bound. Using $\eta > 1/L$ risks divergence; using $\eta \ll 1/L$ is unnecessarily slow.

## Convergence Rates

### Convex + L-Smooth: O(1/T) Rate

> **Remember:** For convex $L$-smooth functions, gradient descent with step size $\eta = 1/L$ gives $f(x_T) - f^* \leq \frac{L\|x_0 - x^*\|^2}{2T}$ — convergence rate $O(1/T)$. To go from suboptimality $\varepsilon$ to $\varepsilon/2$ takes twice as many steps. Linear (geometric) convergence — error halving every fixed number of steps — requires strong convexity.

**Theorem.** Let $f$ be convex and $L$-smooth, with minimizer $x^*$. With step size $\eta = 1/L$, gradient descent satisfies:

$$f(x_T) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2T}$$

**Proof.** From the descent lemma with $\eta = 1/L$:

$$f(x_{t+1}) \leq f(x_t) - \frac{1}{2L}\|\nabla f(x_t)\|^2$$

From the first-order convexity condition: $f(x_t) - f(x^*) \leq \nabla f(x_t)^\top(x_t - x^*)$. By Cauchy-Schwarz and AM-GM:

$$\nabla f(x_t)^\top(x_t - x^*) \leq \frac{1}{2}\left[\frac{\|\nabla f(x_t)\|^2}{1/(2L)} - 0\right]$$

More precisely, letting $\delta_t = f(x_t) - f(x^*)$:

$$\frac{1}{2L}\|\nabla f(x_t)\|^2 \geq \frac{1}{L} \cdot \frac{(f(x_t)-f(x^*))^2}{2\|x_t - x^*\|^2 / (1/(L))}$$

The cleanest proof uses $\|\nabla f(x_t)\|^2 \geq 2L(f(x_t) - f(x^*))$ (from smoothness + convexity), but the telescoping argument is more direct:

From the descent lemma: $\delta_{t+1} \leq \delta_t - \frac{1}{2L}\|\nabla f(x_t)\|^2$.
From the co-coercivity of $L$-smooth convex functions: $\frac{1}{2L}\|\nabla f(x_t)\|^2 \geq \frac{\delta_t^2}{L\|x_0 - x^*\|^2}$.

Alternatively, summing the descent lemma and using convexity directly: $\sum_{t=0}^{T-1} \|\nabla f(x_t)\|^2 \leq 2L(f(x_0) - f(x^*))$, so the running average satisfies $\min_t \|\nabla f(x_t)\|^2 = O(1/T)$.

### Strongly Convex + L-Smooth: Linear Convergence

A function is **$\mu$-strongly convex** if $f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$ for all $x,y$.

**Theorem.** Let $f$ be $\mu$-strongly convex and $L$-smooth, with minimizer $x^*$. With step size $\eta = 1/L$:

$$\|x_{t+1} - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^t \|x_0 - x^*\|^2$$

$$f(x_t) - f(x^*) \leq \frac{L}{2}\left(1 - \frac{\mu}{L}\right)^t \|x_0 - x^*\|^2$$

The convergence ratio $\rho = 1 - \mu/L < 1$ gives geometric (linear) convergence — the error decreases by a constant factor at each step.

**Proof sketch.** Using strong convexity: $\langle \nabla f(x), x - x^*\rangle \geq \frac{\mu}{2}\|x - x^*\|^2 + (f(x) - f(x^*))$. After one gradient step:

$$\|x_{t+1} - x^*\|^2 = \|x_t - \eta\nabla f(x_t) - x^*\|^2 = \|x_t - x^*\|^2 - 2\eta\langle\nabla f(x_t), x_t - x^*\rangle + \eta^2\|\nabla f(x_t)\|^2$$

Using strong convexity to bound the middle term and $L$-smoothness to bound the last term, one obtains the contraction factor $1 - \mu/L$.

### Non-Convex: O(1/√T) to Stationary Points

For non-convex functions (e.g., neural network losses), global convergence is not guaranteed, but gradient descent still converges to **stationary points** where $\nabla f(x) = 0$.

**Theorem.** Let $f$ be $L$-smooth (not necessarily convex), bounded below by $f^*$. With step size $\eta = 1/L$:

$$\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2 \leq \frac{2L(f(x_0) - f^*)}{T}$$

Thus $\min_t \|\nabla f(x_t)\| \leq \sqrt{2L(f(x_0)-f^*)/T} = O(1/\sqrt{T})$.

This is the best possible rate for first-order methods on non-convex smooth functions. Reaching an $\varepsilon$-stationary point (where $\|\nabla f(x)\| \leq \varepsilon$) requires $O(1/\varepsilon^2)$ gradient evaluations.

## Nesterov Acceleration

Gradient descent achieves $O(1/T)$ for convex functions and $O(\rho^T)$ for strongly convex — but are these rates optimal? Nesterov (1983) proved a **lower bound**: no first-order method can achieve better than $O(1/T^2)$ for general smooth convex functions. And he also showed this bound is achievable:

**Nesterov's accelerated gradient descent** maintains two sequences:
$$y_{t+1} = x_t + \frac{t-1}{t+2}(x_t - x_{t-1})$$
$$x_{t+1} = y_{t+1} - \frac{1}{L}\nabla f(y_{t+1})$$

The first step computes a **momentum extrapolation** at a lookahead point $y_{t+1}$, then takes a gradient step from that lookahead. This achieves:
$$f(x_T) - f^* \leq \frac{2L\|x_0 - x^*\|^2}{(T+1)^2} = O(1/T^2)$$

This is optimal — matching Nesterov's lower bound.

> **Key insight:** The critical distinction: **heavy-ball momentum** $x_{t+1} = x_t - \eta\nabla f(x_t) + \beta(x_t - x_{t-1})$ evaluates the gradient at the *current* point $x_t$ and achieves $O(1/T^2)$ only for quadratics. **Nesterov's method** evaluates at the *lookahead* point $y_{t+1}$ and achieves $O(1/T^2)$ universally. FISTA (Lesson 5) uses Nesterov's momentum — not heavy-ball — which is why it achieves the optimal rate.

## The Condition Number

> **Intuition:** The condition number $\kappa = L/\mu$ measures how "elongated" the level sets of $f$ are — for a quadratic, it is the ratio of the largest to smallest eigenvalue of the Hessian. Large $\kappa$ means the loss landscape is shaped like a narrow ravine: gradient descent oscillates across the ravine while making slow progress along it. This is why feature normalization and batch normalization help so much — they reshape the landscape toward $\kappa \approx 1$.

The **condition number** $\kappa = L/\mu$ captures the difficulty of optimization for strongly convex functions. The convergence rate is $\rho = 1 - 1/\kappa$, so the number of iterations to reduce the error by factor $1/e$ is $\approx \kappa$.

For ill-conditioned problems ($\kappa \gg 1$), gradient descent moves very slowly — in directions of low curvature (small eigenvalues of the Hessian), progress is slow because the step size is limited by the large-curvature directions.

**Why batch normalization and weight normalization help.** These techniques reduce the condition number of the optimization problem by normalizing the scale of different directions. Specifically, they reduce the ratio of the largest to smallest curvature, accelerating gradient descent. This is not just empirical observation — it is a direct consequence of the condition number $\kappa$ governing convergence speed.

> **Key insight:** Condition number $\kappa = L/\mu$ determines convergence speed — this is why normalization (batch norm, layer norm, weight decay) dramatically accelerates training by making the loss landscape more isotropic.

## Stochastic Gradient Descent

In ML, we minimize the empirical risk $f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$. Computing the full gradient $\nabla f(x) = \frac{1}{n}\sum_i \nabla f_i(x)$ costs $O(n)$ per step — prohibitive for large $n$.

**SGD** uses a single random sample (or small batch):

$$x_{t+1} = x_t - \eta_t g_t, \quad g_t = \nabla f_{i_t}(x_t)$$

where $i_t$ is drawn uniformly at random. This is an **unbiased estimator**: $\mathbb{E}[g_t \mid x_t] = \nabla f(x_t)$.

**Convergence in expectation.** Let $\sigma^2 = \mathbb{E}[\|g_t - \nabla f(x_t)\|^2]$ be the gradient variance. For convex $f$, with decreasing step sizes $\eta_t = c/\sqrt{t}$:

$$\mathbb{E}[f(\bar{x}_T) - f(x^*)] \leq \frac{\|x_0 - x^*\|^2 + c^2 \sigma^2 \log T}{2c\sqrt{T}} = O\!\left(\frac{\sigma}{\sqrt{T}}\right)$$

where $\bar{x}_T = \frac{1}{T}\sum_{t=0}^{T-1} x_t$ is the **Polyak-Ruppert average**.

> **Intuition:** SGD with a constant step size does not converge to the exact minimum — it oscillates in a neighborhood of size $O(\eta \sigma^2 / \mu)$, where $\sigma^2$ is the gradient variance and $\mu$ is the strong convexity constant. Smaller step sizes shrink this noise floor but also slow down early progress. This tension is why learning rate schedules (start high, decay later) work: high learning rate for fast early progress, decayed learning rate to settle into a tight neighborhood near the end.

Key observations:
- SGD with constant step size $\eta$ converges to a neighborhood of $x^*$ of radius $O(\eta\sigma^2/\mu)$ — but does not converge exactly
- Decreasing step sizes $\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$ ensure convergence to $x^*$ but slow down late-stage convergence
- The $O(1/\sqrt{T})$ rate matches the full-gradient rate in wall-clock time when $n$ is large (each SGD step is $n\times$ cheaper)

**Noise as regularization.** The gradient noise in SGD is not just an approximation artifact — it provides implicit regularization by preventing overfitting to individual samples and, in certain regimes, biasing toward flat minima.

## The Polyak-Łojasiewicz Condition

> **Refresher:** The PL condition $\frac{1}{2}\|\nabla f(x)\|^2 \geq \mu(f(x) - f^*)$ says the gradient is large whenever you are far from the optimum — it is a lower bound on gradient magnitude in terms of suboptimality. Unlike strong convexity, PL does not require a unique minimizer or a bowl-shaped landscape; multiple global minima are allowed. Remarkably, PL gives the same linear convergence rate as strong convexity, and many overparameterized neural networks satisfy PL locally.

The $\mu$-**Polyak-Łojasiewicz (PL) condition** is:

$$\frac{1}{2}\|\nabla f(x)\|^2 \geq \mu(f(x) - f^*), \quad \forall x$$

This is strictly weaker than strong convexity: it requires only that the gradient is large whenever the function value is above the minimum — not that the function itself has a unique shape. Many non-convex functions satisfy the PL condition.

**Theorem (PL linear convergence).** If $f$ is $L$-smooth and satisfies the $\mu$-PL condition, then gradient descent with step size $\eta = 1/L$ achieves:

$$f(x_t) - f^* \leq \left(1 - \frac{\mu}{L}\right)^t (f(x_0) - f^*)$$

*Proof.* From the descent lemma: $f(x_{t+1}) - f^* \leq f(x_t) - \frac{1}{2L}\|\nabla f(x_t)\|^2 - f^* \leq (f(x_t) - f^*) - \frac{\mu}{L}(f(x_t) - f^*)$.

**Why does this matter for neural networks?** While neural network losses are globally non-convex, they often satisfy the PL condition locally — especially near the initialization and in overparameterized regimes. The PL condition is the theoretical mechanism explaining why gradient descent converges to good solutions in neural network training despite the absence of global convexity.

> **Key insight:** The PL condition explains why neural networks, despite non-convexity, often converge to globally good solutions — whenever PL holds locally, gradient descent achieves the same linear convergence rate as strongly convex optimization.

## Mini-Batch Gradient Descent

**Mini-batch SGD** uses a batch of $B$ samples:

$$g_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla f_i(x_t), \quad \text{Var}(g_t) = \frac{\sigma^2}{B}$$

The variance is reduced by a factor of $B$ compared to single-sample SGD. The trade-off:

| Batch size $B$ | Gradient variance | Cost per step | Steps to converge |
|---|---|---|---|
| 1 | $\sigma^2$ | $O(1)$ | $O(1/\varepsilon^2)$ |
| $B$ | $\sigma^2/B$ | $O(B)$ | $O(1/(B\varepsilon^2)) \cdot B = O(1/\varepsilon^2)$ |
| $n$ | 0 | $O(n)$ | $O(\log 1/\varepsilon)$ (if strongly convex) |

For strongly convex problems, mini-batch SGD can match full-batch GD's iteration complexity while being parallelizable. For non-convex problems (e.g., neural nets), large batches $B \gg 1$ reduce gradient noise, which can actually *hurt* generalization — the "large batch generalization gap" is an empirical phenomenon explained by the implicit regularization of gradient noise. This degradation can be mitigated with the **linear scaling rule** (multiply learning rate proportionally to batch size) and linear **warmup** — large-batch training can match small-batch generalization with careful hyperparameter tuning.

## ML Connections

Gradient descent convergence theory explains why every design choice in deep learning training — learning rate, batch size, normalization, momentum — works the way it does.

- **Learning Rate Scheduling:** The optimal step size $\eta = 1/L$ (L-smooth functions) and warmup schedules derive from convergence theory. Too large $\eta$ diverges; too small gives $O(1/T)$ convergence that's impractically slow. Cosine decay with warmup navigates the sharp early landscape (high L) before settling into the smooth basin.
- **Batch Normalization and Layer Normalization:** These operations reduce the condition number $\kappa = L/\mu$ of the loss Hessian. When $\kappa$ is large, gradient descent oscillates — normalization equalizes curvature across parameter directions, making training faster and more stable. BatchNorm was originally motivated empirically; the $\kappa$-reduction explains it theoretically.
- **SGD vs Adam for Generalization:** SGD's $O(\sigma/\sqrt{T})$ convergence with noise scale $\sigma^2$ is not just a limitation — the noise implicitly regularizes training by preferring flat minima (Langevin SDE interpretation). Adam adapts learning rates per coordinate, converging faster but with less noise, which can lead to sharper minima and worse generalization on some tasks.
- **Gradient Clipping in LLM Training:** The PL condition gives linear convergence when $\|\nabla f\|^2 \geq 2\mu(f - f^*)$. When this fails (near saddle points or cliffs), gradients become very large. Gradient clipping enforces an implicit L-smoothness-like constraint by bounding $\|\nabla f\|$, preventing gradient explosions in transformer training.
- **Neural Network Convergence via PL:** Overparameterized networks satisfy the PL condition in a neighborhood of the initialization, which is why gradient descent reliably finds global minima despite non-convexity. The PL condition explains the empirical observation that training loss reaches near-zero for sufficiently large models.

> **Key insight:** The entire design space of neural network optimizers — learning rates, momentum, normalization, clipping — maps directly onto convergence theory. L-smoothness gives the safe step size; strong convexity/PL gives the convergence rate; condition number explains why normalization helps. Knowing the theory lets you diagnose training failures and tune hyperparameters from first principles rather than grid search.

## Python: GD, SGD, and Mini-Batch GD on Logistic Regression

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate linearly separable data
n, d = 500, 20
X = np.random.randn(n, d)
w_true = np.random.randn(d)
y = (X @ w_true > 0).astype(float) * 2 - 1  # labels in {-1, +1}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))

def logistic_loss(w, X, y, lam=0.01):
    """Logistic regression loss with L2 regularization."""
    scores = X @ w
    loss = np.mean(np.log1p(np.exp(-y * scores))) + 0.5 * lam * np.dot(w, w)
    return loss

def logistic_grad(w, X, y, lam=0.01):
    """Full gradient."""
    scores = X @ w
    probs = sigmoid(-y * scores)
    grad = -X.T @ (y * probs) / len(y) + lam * w
    return grad

def stochastic_grad(w, xi, yi, lam=0.01):
    """Gradient on a single sample."""
    score = xi @ w
    prob = sigmoid(-yi * score)
    grad = -xi * (yi * prob) + lam * w
    return grad

# --- Full Gradient Descent ---
def run_gd(eta=0.1, T=200, lam=0.01):
    w = np.zeros(d)
    losses = []
    for _ in range(T):
        losses.append(logistic_loss(w, X, y, lam))
        g = logistic_grad(w, X, y, lam)
        w = w - eta * g
    return losses

# --- SGD (single sample, decaying step size) ---
def run_sgd(eta0=1.0, T=200, lam=0.01):
    w = np.zeros(d)
    losses = []
    for t in range(T):
        losses.append(logistic_loss(w, X, y, lam))
        eta = eta0 / (1 + t * 0.01)  # slowly decaying step size
        i = np.random.randint(n)
        g = stochastic_grad(w, X[i], y[i], lam)
        w = w - eta * g
    return losses

# --- Mini-batch SGD ---
def run_minibatch_sgd(eta=0.1, B=32, T=200, lam=0.01):
    w = np.zeros(d)
    losses = []
    for t in range(T):
        losses.append(logistic_loss(w, X, y, lam))
        idx = np.random.choice(n, B, replace=False)
        g = logistic_grad(w, X[idx], y[idx], lam)
        w = w - eta * g
    return losses

# Run experiments
losses_gd = run_gd(eta=0.5, T=300)
losses_sgd = run_sgd(eta0=2.0, T=300)
losses_mb = run_minibatch_sgd(eta=0.5, B=32, T=300)

# --- Step size sensitivity for GD ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogy(losses_gd, label='Full GD (η=0.5)', linewidth=2)
axes[0].semilogy(losses_sgd, label='SGD (η₀=2.0, decaying)', alpha=0.7)
axes[0].semilogy(losses_mb, label='Mini-batch (B=32, η=0.5)', alpha=0.7)
axes[0].set_xlabel('Iteration (epoch equivalent)')
axes[0].set_ylabel('Training Loss (log scale)')
axes[0].set_title('GD vs SGD vs Mini-Batch SGD')
axes[0].legend()
axes[0].grid(True)

# Demonstrate effect of step size on GD
for eta in [0.05, 0.2, 0.5, 1.0, 2.0]:
    try:
        l = run_gd(eta=eta, T=100)
        if not any(np.isnan(l)):
            axes[1].semilogy(l, label=f'η={eta}')
    except Exception:
        pass
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Training Loss (log scale)')
axes[1].set_title('Effect of Step Size on GD Convergence')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('gradient_descent_convergence.png', dpi=150)
plt.show()

# Estimate condition number from Hessian
H = X.T @ X / n + 0.01 * np.eye(d)  # Hessian at w=0 approx
eigvals = np.linalg.eigvalsh(H)
kappa = eigvals[-1] / eigvals[0]
print(f"Approx condition number κ = L/μ ≈ {kappa:.1f}")
print(f"Expected convergence ratio ρ = 1 - 1/κ ≈ {1 - 1/kappa:.4f}")
print(f"Steps for 1/e reduction ≈ {int(kappa)}")
```

:::quiz
question: "For a μ-strongly convex, L-smooth function, gradient descent converges as (1 − μ/L)^t. What is the effect of multiplying all eigenvalues of the Hessian by a constant factor c > 1 (increasing curvature uniformly)?"
options:
  - "Convergence becomes faster because both μ and L increase, but the ratio μ/L stays the same"
  - "Convergence becomes faster because the minimum eigenvalue μ increases"
  - "Convergence rate is unchanged since κ = L/μ is scale-invariant, but the optimal step size 1/L decreases"
  - "Convergence becomes slower because larger curvature means more difficult optimization"
correct: 2
explanation: "If all Hessian eigenvalues are multiplied by c, then both μ (smallest eigenvalue) and L (largest eigenvalue) scale by c. The condition number κ = L/μ is invariant, so the convergence rate ρ = 1 − μ/L = 1 − 1/κ is unchanged. However, the optimal step size η = 1/L decreases by factor c. This is why the convergence rate (in iterations) is determined solely by the condition number, not the absolute scale of curvature."
:::

:::quiz
question: "SGD with constant step size η does not converge to the exact minimizer x* of a strongly convex function. What does it converge to?"
options:
  - "The global minimum x*, just more slowly than full-batch gradient descent"
  - "A neighborhood of x* of radius proportional to η·σ/μ, where σ² is the gradient variance"
  - "A local minimum of the individual component functions f_i"
  - "It diverges for any constant step size in the strongly convex case"
correct: 1
explanation: "SGD with constant step size oscillates in a neighborhood of x* rather than converging to it. The radius of this neighborhood is O(ησ²/μ) for μ-strongly convex functions with gradient variance σ². This is the 'noise floor' of SGD: the step size η must be reduced to zero (satisfying Σ η_t = ∞, Σ η_t² < ∞) to achieve exact convergence. In practice, this is handled via learning rate schedules that decay η over training."
:::

:::quiz
question: "The Polyak-Łojasiewicz (PL) condition ½‖∇f(x)‖² ≥ μ(f(x) − f*) is weaker than strong convexity. Which of the following non-convex functions satisfies the PL condition with some μ > 0?"
options:
  - "f(x) = x⁴, which has a flat region near zero where the gradient vanishes quickly"
  - "f(x, y) = (x² + y²)·sin²(x), which has infinitely many global minima along the y-axis"
  - "f(x) = x² + 3 sin²(x) for x near 0, which has a global minimum at 0 and satisfies the gradient lower bound"
  - "Any non-convex function with multiple isolated local minima"
correct: 2
explanation: "f(x) = x² + 3sin²(x) has f* = 0 at x = 0 and satisfies the PL condition in a neighborhood of 0 because ‖∇f(x)‖² = (2x + 6sin(x)cos(x))² grows quadratically relative to f(x) − f*. The key property the PL condition captures is that the gradient cannot be small unless we are near the minimum — it does not require a unique minimum or a bowl-shaped landscape. Functions with multiple isolated local minima at different values do NOT satisfy PL globally, since at a non-global local minimum ∇f = 0 but f > f*."
:::
