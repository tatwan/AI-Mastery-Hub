---
title: "Convex Analysis & Optimality Conditions"
estimatedMinutes: 30
tags: ["convex", "KKT", "duality", "subgradient", "SVM", "constrained-optimization"]
prerequisites: []
---

## Convex Sets and Functions

A set $\mathcal{C} \subseteq \mathbb{R}^n$ is **convex** if for any two points $x, y \in \mathcal{C}$ and any $\theta \in [0,1]$, the convex combination also lies in $\mathcal{C}$:

$$\theta x + (1-\theta)y \in \mathcal{C}$$

Geometrically, the line segment connecting any two points stays inside the set. Examples include affine subspaces, halfspaces $\{x : a^\top x \leq b\}$, Euclidean balls $\{x : \|x\| \leq r\}$, and the positive semidefinite cone $\mathbb{S}^n_+$.

A function $f : \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain is a convex set and for all $x, y$ in the domain:

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y), \quad \forall \theta \in [0,1]$$

> **Intuition:** A function is convex if the chord between any two points on its graph lies above (or on) the graph — the classic "bowl shape." The most important consequence: every local minimum is also a global minimum. This is why convex optimization is tractable: you never need to worry about getting stuck in a suboptimal valley.

The function lies at or below the chord connecting any two of its points. It is **strictly convex** if the inequality is strict for $x \neq y$ and $\theta \in (0,1)$.

**Canonical examples:**

- *Affine functions* $f(x) = a^\top x + b$ — convex and concave simultaneously
- *Quadratic* $f(x) = \frac{1}{2}x^\top Q x + b^\top x$ — convex iff $Q \succeq 0$
- *Norms* $f(x) = \|x\|_p$ for $p \geq 1$ — convex by triangle inequality
- *Log-sum-exp* $f(x) = \log \sum_{i=1}^n e^{x_i}$ — convex; appears in softmax-based losses
- *Negative entropy* $f(p) = \sum_i p_i \log p_i$ — strictly convex over the probability simplex
- *Indicator function* $\delta_\mathcal{C}(x) = 0$ if $x \in \mathcal{C}$, $+\infty$ otherwise — convex iff $\mathcal{C}$ is convex

**Preservation of convexity.** The following operations preserve convexity: non-negative weighted sums, composition with affine maps ($f(Ax+b)$), pointwise maximum of convex functions, and infimal convolution.

### Jensen's Inequality

> **Refresher:** Jensen's inequality says that for a convex $f$, the function of the average is no greater than the average of the function: $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$. Think of it this way: if you average the inputs first and then apply a bowl-shaped function, you land lower than if you apply the function to each input first and then average the results. This directly explains why log-sum-exp is convex and why KL divergence is non-negative.

For a convex function $f$ and a random variable $X$:

$$f\!\left(\mathbb{E}[X]\right) \leq \mathbb{E}[f(X)]$$

This single inequality underlies an enormous number of results in ML: the derivation of the ELBO in VAEs, the expectation-maximization algorithm, log-sum-exp bounds, and concentration inequalities all rely on Jensen's inequality.

**Proof sketch.** By the first-order condition (derived below), $f(y) \geq f(\mathbb{E}[X]) + \nabla f(\mathbb{E}[X])^\top(y - \mathbb{E}[X])$ for all $y$. Taking expectations of both sides over $Y = X$ gives the result.

> **Key insight:** Jensen's inequality is the single inequality underlying the ELBO in variational inference, the EM algorithm, and the non-negativity of KL divergence — it may be the most versatile tool in all of ML analysis.

## First and Second-Order Conditions

**First-order condition.** A differentiable function $f$ is convex if and only if:

$$f(y) \geq f(x) + \nabla f(x)^\top (y - x), \quad \forall x, y$$

The tangent hyperplane at any point is a global lower bound on the function. This is the most useful characterization for optimization: the gradient descent inequality, convergence proofs, and duality theory all follow from it.

*Proof (sufficiency).* Assume the condition holds. For any $\theta \in [0,1]$:

$$f(x) \geq f(z) + \nabla f(z)^\top (x-z)$$
$$f(y) \geq f(z) + \nabla f(z)^\top (y-z)$$

where $z = \theta x + (1-\theta)y$. Multiplying by $\theta$ and $(1-\theta)$ and adding:

$$\theta f(x) + (1-\theta)f(y) \geq f(z) = f(\theta x + (1-\theta)y)$$

*Proof (necessity).* For convex $f$, the directional derivative exists. Taking the limit as $\theta \to 0$ of the convexity inequality $(f(\theta y + (1-\theta)x) \leq \theta f(y) + (1-\theta)f(x))$ yields the gradient condition.

**Optimality from the first-order condition.** For unconstrained minimization of convex $f$, the point $x^*$ is a global minimum if and only if $\nabla f(x^*) = 0$. This is why zero-gradient is a sufficient (not merely necessary) condition for global optimality in convex problems — a property that distinguishes convex from general optimization.

**Second-order condition.** A twice-differentiable function $f$ is convex if and only if its Hessian is positive semidefinite everywhere:

$$\nabla^2 f(x) \succeq 0, \quad \forall x$$

It is strongly convex with parameter $\mu > 0$ if $\nabla^2 f(x) \succeq \mu I$ for all $x$, which implies:

$$f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$$

Strong convexity guarantees a unique minimizer and is the key ingredient in proving linear (geometric) convergence of gradient methods.

## Subdifferentials and Subgradient Descent

Many important ML objectives — $\ell_1$ regularization, ReLU activations, hinge loss — are convex but not differentiable everywhere. Subgradient theory extends the first-order condition to non-smooth functions.

> **Intuition:** At a non-smooth point like the tip of $|x|$ at $x=0$, the derivative doesn't exist — there is no single tangent line. The subdifferential $\partial f(x)$ replaces it with the set of all slopes of valid supporting hyperplanes. For $|x|$ at $x=0$, any slope in $[-1, 1]$ is a valid subgradient. Subgradient descent picks any element of $\partial f(x_t)$ and steps in that direction; the step might not decrease $f$ at every iteration, but convergence is guaranteed in the long run.

**Definition.** A vector $g \in \mathbb{R}^n$ is a **subgradient** of $f$ at $x$ if:

$$f(y) \geq f(x) + g^\top(y - x), \quad \forall y$$

The **subdifferential** $\partial f(x)$ is the set of all subgradients at $x$. When $f$ is differentiable at $x$, the subdifferential contains exactly the gradient: $\partial f(x) = \{\nabla f(x)\}$.

**Examples:**

- Absolute value $f(x) = |x|$: $\partial f(x) = \{1\}$ for $x > 0$, $\{-1\}$ for $x < 0$, $[-1, 1]$ for $x = 0$
- $\ell_1$ norm $f(x) = \|x\|_1$: $\partial f(x)_i = \text{sign}(x_i)$ if $x_i \neq 0$, $[-1,1]$ if $x_i = 0$
- ReLU $f(x) = \max(0,x)$: $\partial f(0) = [0,1]$; in practice deep learning uses a specific choice (0 or 1) at non-differentiable points without causing issues empirically

**Optimality condition.** For a convex function, $x^*$ minimizes $f$ if and only if $0 \in \partial f(x^*)$.

**Subgradient descent.** Pick any $g_t \in \partial f(x_t)$ and update:

$$x_{t+1} = x_t - \eta_t g_t$$

Unlike gradient descent, the subgradient step is not guaranteed to be a descent step at every iteration. We must track the best iterate. With step sizes $\eta_t = c/\sqrt{t}$, subgradient descent achieves $O(1/\sqrt{T})$ convergence for Lipschitz convex functions — slower than gradient descent on smooth functions, which achieves $O(1/T)$.

> **Key insight:** Subgradients extend gradient methods to non-smooth objectives including ReLU networks and $\ell_1$-regularized models without sacrificing convergence guarantees.

## Fenchel Conjugate

The **Fenchel conjugate** (or convex conjugate) of $f: \mathbb{R}^n \to \mathbb{R}$ is:

$$f^*(y) = \sup_{x \in \text{dom}(f)} \{y^\top x - f(x)\}$$

Geometrically, $f^*(y)$ is the maximum gap between the linear function $y^\top x$ and $f(x)$. Key properties:

- $f^*$ is always convex (even if $f$ is not), as a supremum of linear functions
- **Fenchel-Young inequality:** $f(x) + f^*(y) \geq x^\top y$ for all $x, y$
- **Double conjugate:** $(f^*)^* = f$ when $f$ is convex and closed

**Key examples:**
- $f(x) = \frac{1}{2}\|x\|^2$: $f^*(y) = \frac{1}{2}\|y\|^2$ (self-conjugate)
- $f(x) = \|x\|_1$: $f^*(y) = \delta_{\|y\|_\infty \leq 1}$ (indicator of $\ell_\infty$ unit ball)
- $f(x) = \|x\|$: $f^*(y) = \delta_{\|y\| \leq 1}$ (indicator of unit ball)

The **Moreau identity** connects conjugates to proximal operators:
$$\text{prox}_{\eta f}(v) + \eta\,\text{prox}_{f^*/\eta}(v/\eta) = v$$

This identity is essential for ADMM (Lesson 5), where subproblems involving $f^*$ arise naturally. The conjugate also connects to f-divergences (M3): the Donsker-Varadhan representation uses $f^*(t) = \sup_x\{tx - f(x)\}$ as the key variational form.

## Lagrangian Duality

Consider the **primal problem**:

$$p^* = \min_{x} \; f_0(x) \quad \text{subject to} \quad f_i(x) \leq 0, \; i=1,\ldots,m, \quad h_j(x) = 0, \; j=1,\ldots,p$$

The **Lagrangian** combines the objective and constraints with multipliers:

$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

where $\lambda \geq 0$ are the dual variables for inequality constraints and $\nu$ are unrestricted for equality constraints.

**Dual function.** The Lagrange dual function is:

$$g(\lambda, \nu) = \inf_{x} \; L(x, \lambda, \nu)$$

> **Intuition:** The dual problem is obtained by minimizing the Lagrangian over $x$ for fixed multipliers — this gives a lower bound on the primal optimal value for any choice of $\lambda \geq 0$. The dual then maximizes this lower bound. It is always a concave maximization problem (easier structure) even if the primal is non-convex. Strong duality — where the lower bound is tight and equals the primal optimum — holds when Slater's condition is satisfied.

This is always concave in $(\lambda, \nu)$, regardless of the convexity of the original problem, because it is a pointwise infimum of affine functions.

**Weak duality.** For any $\lambda \geq 0$ and any $\nu$:

$$g(\lambda, \nu) \leq p^*$$

*Proof.* For any feasible $\tilde{x}$: $g(\lambda,\nu) \leq L(\tilde{x},\lambda,\nu) = f_0(\tilde{x}) + \sum_i \lambda_i f_i(\tilde{x}) + \sum_j \nu_j h_j(\tilde{x}) \leq f_0(\tilde{x})$ since $\lambda_i \geq 0$, $f_i(\tilde{x}) \leq 0$, and $h_j(\tilde{x}) = 0$. Taking the infimum over feasible $\tilde{x}$ gives $g(\lambda,\nu) \leq p^*$.

Weak duality holds for any optimization problem — convex or not.

**The dual problem** maximizes the lower bound:

$$d^* = \max_{\lambda \geq 0, \nu} \; g(\lambda, \nu)$$

The **duality gap** is $p^* - d^* \geq 0$.

**Strong duality.** Under **Slater's condition** — if the primal is convex and there exists a strictly feasible point $\tilde{x}$ with $f_i(\tilde{x}) < 0$ for all $i$ — then strong duality holds. More precisely, Slater's condition: there exists a **strictly feasible** point $\tilde{x}$ in the relative interior of the domain such that $f_i(\tilde{x}) < 0$ for all inequality constraints $i$ and $h_j(\tilde{x}) = 0$ for all equality constraints $j$.

$$d^* = p^*$$

This means we can solve the (concave) dual problem to find the optimal value of the (convex) primal. The dual is often easier to solve because: (i) the infimum over $x$ may have a closed form, (ii) it may have fewer variables, or (iii) it may expose hidden structure like the kernel trick.

## KKT Conditions

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary and sufficient for optimality in convex problems with constraint qualifications (such as Slater's condition).

> **Remember:** KKT conditions are necessary and sufficient for optimality at a constrained minimum (under Slater's condition): (1) **Stationarity** — gradient of the Lagrangian with respect to $x$ is zero; (2) **Primal feasibility** — all constraints are satisfied; (3) **Dual feasibility** — $\lambda_i \geq 0$ for inequality multipliers; (4) **Complementary slackness** — $\lambda_i g_i(x) = 0$ for each inequality, meaning either the constraint is active or its multiplier is zero. Every constrained ML problem — SVMs, fairness constraints, LoRA bounds — can be analyzed through these four conditions.

For a point $(x^*, \lambda^*, \nu^*)$ to be primal-dual optimal, it must satisfy:

1. **Stationarity:** $\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$
2. **Primal feasibility:** $f_i(x^*) \leq 0$ for all $i$, $h_j(x^*) = 0$ for all $j$
3. **Dual feasibility:** $\lambda_i^* \geq 0$ for all $i$
4. **Complementary slackness:** $\lambda_i^* f_i(x^*) = 0$ for all $i$

**Complementary slackness** is the most interpretable condition: either the dual variable $\lambda_i^* = 0$ (constraint $i$ is not active, it doesn't influence the optimum) or $f_i(x^*) = 0$ (constraint $i$ is active). A constraint that is strictly inactive ($f_i(x^*) < 0$) must have $\lambda_i^* = 0$.

**Sufficiency.** For a convex primal problem, any point satisfying all four KKT conditions is primal-dual optimal (and strong duality holds).

**Necessity.** For differentiable convex problems satisfying Slater's condition, any primal optimal point has dual variables such that all KKT conditions hold. For non-convex problems, KKT conditions are necessary only under additional constraint qualifications (e.g., linear independence constraint qualification, LICQ).

> **Key insight:** KKT conditions are the fundamental language for understanding constrained ML optimization — every constrained learning problem (SVMs, constrained fine-tuning, fairness constraints) can be analyzed through the KKT lens.

## ML Connection: Support Vector Machines

The SVM derives directly from convex duality, and its most elegant properties are only visible through the KKT conditions.

**Primal SVM.** For binary classification with labels $y_i \in \{-1, +1\}$ and features $x_i$, the hard-margin SVM solves:

$$\min_{w, b} \; \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1, \; i=1,\ldots,n$$

This is a quadratic program (QP) with $n$ linear constraints. Using inequality constraints $f_i(w,b) = 1 - y_i(w^\top x_i + b) \leq 0$, the Lagrangian is:

$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i \left[y_i(w^\top x_i + b) - 1\right]$$

**Deriving the dual.** Minimizing over $w$ and $b$ (stationarity conditions):

$$\frac{\partial L}{\partial w} = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i$$

$$\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0$$

Substituting back yields the **dual SVM**:

$$\max_{\alpha \geq 0} \; \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**The kernel trick.** The dual depends on $x_i$ only through inner products $x_i^\top x_j$. If we replace the inner product with a kernel $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$, we effectively perform classification in a (possibly infinite-dimensional) feature space $\phi(\cdot)$ without ever computing $\phi$ explicitly. This is only visible from the dual formulation.

**Support vectors.** By complementary slackness, $\alpha_i (y_i(w^\top x_i + b) - 1) = 0$. Thus $\alpha_i > 0$ only for points on the margin ($y_i(w^\top x_i + b) = 1$) — these are the **support vectors**. The solution $w = \sum_i \alpha_i y_i x_i$ depends only on support vectors, giving the SVM its sparsity and robustness properties.

## Python: Subgradient Descent for L1-Regularized Least Squares

We implement subgradient descent on the LASSO objective $f(x) = \frac{1}{2}\|Ax - b\|^2 + \lambda\|x\|_1$ and compare to `scipy.optimize.minimize`.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# Generate synthetic data: sparse true solution
n, d = 100, 50
A = np.random.randn(n, d)
x_true = np.zeros(d)
x_true[:5] = np.random.randn(5)  # only 5 non-zero coefficients
b = A @ x_true + 0.1 * np.random.randn(n)

lam = 0.5  # regularization strength

def lasso_objective(x):
    return 0.5 * np.sum((A @ x - b)**2) + lam * np.sum(np.abs(x))

def lasso_subgradient(x):
    """Compute a subgradient of the LASSO objective."""
    residual = A @ x - b
    grad_smooth = A.T @ residual  # gradient of smooth part
    # subgradient of L1: sign(x_i) if x_i != 0, any value in [-1,1] if x_i == 0
    subgrad_l1 = np.sign(x)  # convention: sign(0) = 0 is a valid subgradient
    return grad_smooth + lam * subgrad_l1

def subgradient_descent(x_init, T=2000, c=0.1):
    """Subgradient descent with step size eta_t = c / sqrt(t)."""
    x = x_init.copy()
    x_best = x.copy()
    f_best = lasso_objective(x)
    history = [f_best]

    for t in range(1, T + 1):
        eta = c / np.sqrt(t)
        g = lasso_subgradient(x)
        x = x - eta * g
        f_val = lasso_objective(x)
        if f_val < f_best:
            f_best = f_val
            x_best = x.copy()
        history.append(f_best)

    return x_best, history

# Run subgradient descent
x_init = np.zeros(d)
x_sgd, history_sgd = subgradient_descent(x_init, T=2000, c=0.5)

# Reference solution via scipy
result = minimize(
    lasso_objective,
    x_init,
    method='L-BFGS-B',
    jac=lasso_subgradient,
    options={'maxiter': 5000, 'ftol': 1e-15}
)
f_opt = result.fun

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].semilogy(np.array(history_sgd) - f_opt + 1e-12)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Suboptimality $f(x_t) - f^*$')
axes[0].set_title('Subgradient Descent Convergence (log scale)')
axes[0].grid(True)

axes[1].stem(range(d), x_true, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='True x')
axes[1].stem(range(d), x_sgd, linefmt='C1--', markerfmt='C1x', basefmt='k-', label='SGD solution')
axes[1].set_xlabel('Coordinate')
axes[1].set_ylabel('Value')
axes[1].set_title('Sparsity Pattern Recovery')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('subgradient_lasso.png', dpi=150)
plt.show()

print(f"Scipy optimum: {f_opt:.6f}")
print(f"Subgradient best: {history_sgd[-1]:.6f}")
print(f"Non-zeros (thresh 0.01): SGD={np.sum(np.abs(x_sgd) > 0.01)}, True={np.sum(np.abs(x_true) > 0.01)}")
```

The subgradient descent achieves $O(1/\sqrt{T})$ convergence on this non-smooth problem. Notice that the method recovers the sparse structure of the true solution — this is the power of $\ell_1$ regularization as a convex relaxation of the $\ell_0$ sparsity constraint.

:::quiz
question: "Which of the following is NOT preserved under the subdifferential calculus? That is, which operation can produce a non-convex subdifferential?"
options:
  - "Taking the non-negative weighted sum of convex functions"
  - "Composing a convex function with a linear map"
  - "Pointwise maximum of finitely many convex functions"
  - "Pointwise minimum of two convex functions"
correct: 3
explanation: "The pointwise minimum of two convex functions is generally not convex (e.g., min(x², (x−2)²) has a non-convex region between the two parabolas). The other three operations — weighted sums, affine composition, and pointwise maximum — all preserve convexity. This is why 'max' operations (ReLU, softmax, hinge loss) appear everywhere in ML but 'min' operations over function classes require different treatment."
:::

:::quiz
question: "In the SVM dual derivation, why do support vectors have α_i > 0 while non-support-vector points have α_i = 0?"
options:
  - "Because the optimizer prefers sparser solutions to avoid overfitting"
  - "Because complementary slackness requires α_i · (margin constraint) = 0, and points not on the margin satisfy the constraint strictly"
  - "Because gradient descent converges to zero for non-active constraints"
  - "Because the kernel matrix is singular at non-support vectors"
correct: 1
explanation: "Complementary slackness (a KKT condition) states α_i · [y_i(w⊤x_i + b) − 1] = 0. If a point is strictly inside the margin (not on the boundary), then y_i(w⊤x_i + b) − 1 > 0, so complementary slackness forces α_i = 0. Conversely, if α_i > 0, the point must lie exactly on the margin boundary — these are the support vectors. The solution w = Σ α_i y_i x_i thus depends only on support vectors."
:::

:::quiz
question: "Slater's condition requires a strictly feasible point for a convex program. What does 'strictly feasible' mean and why is it needed?"
options:
  - "All constraint gradients must be linearly independent at the optimum"
  - "The objective function must be strictly convex (positive definite Hessian)"
  - "There exists a primal point satisfying all inequality constraints with strict inequality, ensuring the duality gap is zero"
  - "The dual variables must be strictly positive at the optimum"
correct: 2
explanation: "Slater's condition requires the existence of a point x̃ such that f_i(x̃) < 0 for all i (strict inequality for inequalities) and h_j(x̃) = 0 for equalities. This interior-point condition guarantees strong duality (zero duality gap) for convex programs. Without it, strong duality may fail — for example, an infeasible dual or a positive duality gap. It is a constraint qualification: it ensures the constraints do not create degenerate geometry that prevents the dual from reaching the primal optimal value."
:::
