---
title: "Non-Convex Landscapes & Implicit Regularization"
estimatedMinutes: 35
tags: ["non-convex", "saddle-points", "loss-landscape", "SAM", "implicit-regularization", "flatness", "double-descent"]
prerequisites: ["l2-gradient-methods", "l3-adaptive-optimizers"]
---

## Non-Convex Challenges

Real neural network loss landscapes are globally non-convex — they have multiple local minima, saddle points, and flat plateaus. Classical optimization theory offers few guarantees in this regime, yet gradient descent on deep networks reliably finds good solutions. Understanding why requires revisiting what the loss landscape actually looks like in high dimensions.

**Local minima.** A local minimum $x^*$ satisfies $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \succeq 0$. In low dimensions, local minima pose a serious challenge — gradient descent can become trapped at a suboptimal value. The classical worry about training neural networks was that they would get stuck in poor local minima.

**Saddle points.** A saddle point $x^*$ satisfies $\nabla f(x^*) = 0$ but $\nabla^2 f(x^*)$ is indefinite — it has both positive and negative eigenvalues. Near a saddle point, there exist directions of both ascent and descent.

**Flat regions.** Regions where $\|\nabla f(x)\|$ is small but $f(x) > f^*$, causing gradient-based methods to stagnate. These are common in early training before the network has learned useful representations.

## Saddle Points Dominate in High Dimensions

A key result from random matrix theory explains the loss landscape of overparameterized networks: **in high dimensions, saddle points dominate, but local minima are rare and near-global**.

> **Intuition:** In $n$-dimensional space, a saddle point has negative curvature in $k$ directions and positive curvature in the remaining $n-k$ directions. A true local minimum requires all $n$ directions to have positive curvature — an exponentially rare event for large $n$ under random-matrix statistics. For large overparameterized networks, almost all critical points are saddle points, not bad local minima. Having a negative-curvature escape direction is the typical case, not the exception.

**Random matrix theory argument.** Consider a function $f : \mathbb{R}^d \to \mathbb{R}$ near a critical point. The Hessian $\nabla^2 f(x^*)$ has $d$ eigenvalues. For a random (Wigner) matrix, the fraction of negative eigenvalues is determined by the index (fraction of negative eigenvalues). A **local minimum** requires all $d$ eigenvalues to be non-negative — an event with exponentially small probability in $d$ for random matrices. A saddle point with index $k$ (fraction $k/d$ negative eigenvalues) is exponentially more common.

Dauphin et al. (2014) and Choromanska et al. (2015) argued that for neural network losses:
1. Most critical points are saddle points, not local minima
2. Critical points with high loss tend to have many escape directions (large negative eigenvalue count)
3. Critical points with low loss (near the global minimum) tend to be local minima — but they are all at approximately the same loss value

Note: the Choromanska et al. result requires strong assumptions — Gaussian activations, independence, and that the loss can be modeled as a spin-glass Hamiltonian — none of which hold for real networks. It provides heuristic motivation, not a rigorous theorem.

This means gradient descent in a high-dimensional network landscape does not typically get "stuck" in a bad local minimum. The exponential number of near-equivalent local minima means almost all of them generalize similarly.

**Strict saddle property.** A saddle point is **strict** if the minimum Hessian eigenvalue is strictly negative: $\lambda_{\min}(\nabla^2 f(x^*)) < 0$. For functions satisfying the strict saddle property, perturbed gradient descent (adding small noise to gradients) escapes all strict saddle points in polynomial time (Jin et al., 2017). Many neural network architectures satisfy this property locally.

> **Key insight:** In high dimensions, the "curse of non-convexity" is not as bad as feared — saddle points dominate but can be escaped via gradient noise, and most local minima in overparameterized networks are near-global.

## Escape from Saddle Points

**Via noise (SGD).** The gradient noise in stochastic gradient descent provides implicit perturbations that help escape saddle points. Near a saddle point, the gradient is small but not exactly zero. The stochastic noise has components in escape directions (negative curvature directions), which over time push the iterate away from the saddle. The noise scale for mini-batch SGD is proportional to $\eta/B$ (step size divided by batch size).

**Via second-order information.** Newton's method uses the Hessian inverse: $x_{t+1} = x_t - [\nabla^2 f(x_t)]^{-1} \nabla f(x_t)$. Near a saddle point with negative eigenvalue, the Newton step in the negative-curvature direction has the wrong sign — it actually moves *toward* the saddle. Saddle-free Newton modifies the Hessian to $|\nabla^2 f(x_t)|$ (taking absolute values of eigenvalues), which correctly escapes saddle points in all directions. However, this costs $O(d^3)$ per step and is impractical for large networks.

**Noisy gradient dynamics.** The continuous-time limit of SGD with noise scale $\sigma^2$ is the Langevin SDE:

$$dx_t = -\nabla f(x_t)\,dt + \sqrt{2\sigma^2}\,dW_t$$

where $W_t$ is a standard Brownian motion. The stationary distribution of this SDE is the Gibbs distribution $p(x) \propto \exp(-f(x)/\sigma^2)$, which concentrates on minima but assigns positive probability to all of $\mathbb{R}^d$. This means the continuous-time SGD process will eventually escape any finite-depth local minimum or saddle point.

## Loss Landscape Geometry: Flat vs. Sharp Minima

Two neural networks achieving the same training loss may generalize very differently depending on the **geometry** of the minimum they found.

> **Refresher:** A minimum is "flat" if the loss doesn't change much when you perturb the weights — the Hessian has small eigenvalues. "Sharpness" is measured by the largest Hessian eigenvalue $\lambda_{\max}(H)$: a large value means small weight perturbations cause large loss increases. Flat minima often generalize better, possibly because they correspond to solutions that are robust to natural distribution shift between training and test. Note that raw sharpness is not reparametrization-invariant, so filter-normalized measures are used in practice.

**Sharp minimum.** A minimum where the Hessian has large eigenvalues — the loss increases steeply when parameters are perturbed. Formally, for a minimum $x^*$ with Hessian $H$, the loss at a perturbed point $x^* + \delta$ is approximately $f(x^*) + \frac{1}{2}\delta^\top H \delta$. If $\lambda_{\max}(H)$ is large, small perturbations cause large loss increases.

**Flat minimum.** A minimum where $\lambda_{\max}(H)$ is small — the loss landscape is nearly flat in a large neighborhood. Flat minima are robust to parameter perturbations.

**The flat minima hypothesis** (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017): flat minima generalize better than sharp minima. Intuition: if the training loss landscape is flat at the minimum, then natural distribution shift (which changes the loss landscape slightly) is less likely to cause large degradation. Sharp minima, being narrow, may become poor solutions on the test distribution even if they have zero training error.

**Qualification.** Dinh et al. (2017) showed this is not a well-defined notion: given any minimum, one can reparametrize the network to make it arbitrarily sharp or flat without changing the function. This is a **refutation** of flatness as a scale-invariant measure, not merely a caveat. Flatness must be measured in a scale-invariant way (e.g., via SAM's $\rho$-ball) to be meaningful. Filter-normalized sharpness (Li et al., 2018) and the Hessian trace relative to parameter norm are more meaningful alternatives to naive $\lambda_{\max}(H)$.

**Filter normalization for visualization.** To visualize the 2D loss landscape of a neural network, Li et al. (2018) define two random directions $\delta$ and $\eta$, normalized to have the same $\ell_2$ norm per layer (filter normalization), and plot $f(\theta^* + \alpha\delta + \beta\eta)$ as a function of $(\alpha, \beta)$. This reveals qualitative differences between, say, ResNet (smooth landscape) and networks without skip connections (chaotic landscape).

## Sharpness-Aware Minimization (SAM)

**SAM** (Foret, Kleiner, Mobahi & Neyshabur, 2021) directly targets flat minima by solving a minimax problem:

$$\min_\theta \max_{\|\varepsilon\|_2 \leq \rho} L(\theta + \varepsilon)$$

Instead of minimizing the loss at $\theta$, minimize the **worst-case loss** within an $\rho$-ball around $\theta$. The solution of the inner maximization is approximately:

$$\hat{\varepsilon}(\theta) = \rho \cdot \frac{\nabla_\theta L(\theta)}{\|\nabla_\theta L(\theta)\|_2}$$

This is the perturbation direction that maximally increases the loss (gradient ascent step, normalized to lie on the $\rho$-ball boundary).

**SAM two-step update:**

> **Remember:** SAM update: $\theta \leftarrow \theta - \eta\nabla L(\theta + \hat{\varepsilon})$ where $\hat{\varepsilon} = \rho \cdot \nabla L(\theta) / \|\nabla L(\theta)\|$. First perturb to the worst-case point on the $\rho$-ball boundary (one gradient evaluation), then take a gradient step from that perturbed point (second gradient evaluation). Cost: 2× gradient evaluations per step compared to standard GD or SGD.

1. Compute $\hat{\varepsilon} = \rho \cdot \nabla L(\theta) / \|\nabla L(\theta)\|$
2. Gradient step at the perturbed point: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t + \hat{\varepsilon})$

This requires **two forward-backward passes** per update — doubling the compute cost. Efficient-SAM and mSAM variants reduce this cost.

**Why SAM finds flatter minima.** The gradient at the perturbed point $\theta + \hat{\varepsilon}$ points away from sharp directions. In a sharp valley, the gradient at the perturbed point is large and points away from the minimum; SAM thus penalizes and escapes sharp minima. In a flat region, the perturbed gradient is small and similar to the unperturbed gradient — SAM is content to stay.

**Empirical improvements.** SAM consistently improves generalization on image classification (0.5–2% on CIFAR-10/100, ImageNet) and language tasks. It is particularly effective when training with large batch sizes, which otherwise tend to find sharper minima.

> **Key insight:** Flat minima correlate with better generalization, but the causal mechanism is debated — SAM provides a practical algorithm to explicitly seek flat minima at the cost of 2× compute.

## Implicit Regularization of SGD

A remarkable property of SGD: even without explicit regularization, it has a preference for certain types of solutions. This **implicit regularization** is now understood to arise from the noise structure of the stochastic gradient.

**Noise scale.** For mini-batch SGD with batch size $B$ and learning rate $\eta$, the gradient noise has covariance:

$$\text{Cov}(g) = \frac{\eta}{B} \cdot \Sigma(\theta)$$

where $\Sigma(\theta) = \frac{1}{n}\sum_i (\nabla f_i(\theta) - \nabla f(\theta))(\nabla f_i(\theta) - \nabla f(\theta))^\top$ is the gradient covariance. The "noise scale" $\eta/B$ controls the effective temperature of the implicit Gibbs distribution.

**SDE limit.** As $\eta, B \to \infty$ with $\eta/B = \text{const}$, the SGD trajectory converges to a SDE. The stationary distribution concentrates near flat minima — specifically, it is proportional to $\exp(-f(\theta)/(\eta/(2B)))$, which for fixed loss value $f(\theta)$ assigns higher probability to regions with smaller Hessian volume (i.e., flatter regions).

**Bias toward flat minima.** Large batch training (large $B$) reduces the noise scale, causing the stationary distribution to concentrate more sharply. Empirically, large-batch training finds sharper minima and generalizes worse — this is the large-batch generalization gap observed in practice, explained by the reduction in implicit noise-induced regularization.

> **Intuition:** SGD with batch size $B$ and learning rate $\eta$ injects noise with scale proportional to $\eta/B$. This noise acts like a temperature: high temperature (small $B$ or large $\eta$) prevents the optimizer from settling into narrow sharp valleys, biasing it toward wide flat basins. Sharp minima are "washed out" by the noise because they are sensitive to small perturbations. Smaller batch size means more noise, which means flatter minima found, which often means better generalization — the empirical "large-batch generalization gap" is this mechanism in action.

## Stochastic Weight Averaging (SWA)

**SWA** (Izmailov et al., 2018) improves generalization by averaging model weights over the training trajectory:

$$\theta_{\text{SWA}} = \frac{1}{T_2 - T_1} \sum_{t=T_1}^{T_2} \theta_t$$

where $T_1$ is some point after initial convergence and $T_2$ is the end of training. The averaged weights typically lie in a flatter region than any single $\theta_t$ because:
1. The training trajectory oscillates around local minima (due to SGD noise)
2. The average of multiple points in a neighborhood of a minimum tends to be closer to the center of a wide flat basin than any single point

**Connection to ensembles.** SWA approximates an ensemble prediction by averaging parameters rather than averaging outputs. For linear models, these are equivalent; for nonlinear networks, SWA is an approximation but uses only one model at inference time.

**Practical implementation.** Run standard training until near convergence, then switch to a high constant learning rate (to keep exploring) and collect weight snapshots periodically. Average these snapshots. After averaging, run batch normalization statistics update (forward pass over training data to recompute running stats for the averaged weights).

## Double Descent

The **double descent** phenomenon overturns classical statistical learning theory in the modern overparameterized regime.

> **Intuition:** Classical statistics says more parameters leads to overfitting and higher test error — the familiar U-shaped bias-variance tradeoff. Modern ML has overturned this: past the interpolation threshold (where the model can perfectly fit the training data), adding even more parameters causes test error to decrease again. The minimum-norm interpolating solution found by gradient descent generalizes surprisingly well because it spreads its capacity across the signal subspace rather than fitting noise.

**Classical bias-variance tradeoff.** In classical statistics, as model complexity (parameters) increases:
- Bias decreases (the model family contains better approximations to the truth)
- Variance increases (with limited data, complex models overfit)
- The optimal model has intermediate complexity: test error forms a U-shaped curve

This gives the traditional advice: "don't use more parameters than samples."

**Interpolation threshold.** The regime where the model is just complex enough to perfectly fit the training data is the **interpolation threshold** (roughly when parameters $\approx$ samples). Classical theory predicts this should be the worst operating point.

**Modern (second descent).** Belkin et al. (2019) and others observed that past the interpolation threshold, as overparameterization increases further, test error *decreases again* — forming a double descent curve:

$$\text{test error} \approx \begin{cases} \text{U-shaped} & \text{under-parameterized} \\ \text{peak} & \text{at interpolation threshold} \\ \text{decreasing} & \text{over-parameterized (modern regime)} \end{cases}$$

**Why overparameterized models generalize (benign overfitting).** Multiple mechanisms explain this:
1. **Implicit regularization.** Gradient descent finds the *minimum norm* interpolating solution among the infinitely many that fit the data perfectly. The minimum norm solution generalizes well if the true signal is low-dimensional
2. **Benign overfitting** (Bartlett et al., 2020). In high dimensions, a model can interpolate training noise while ignoring it in the test distribution, if the signal and noise occupy different subspaces
3. **Neural tangent kernel regime.** In the infinite-width limit, neural networks behave like kernel methods with the NTK, and the kernel regression solution is well-understood. However, the NTK convergence guarantee requires network width $\Omega(n^6/\lambda_0^4)$ in terms of training points $n$ — orders of magnitude wider than practical networks. In practice, the NTK regime (where parameters barely move during training) does not hold, and 'feature learning' (parameter evolution) is responsible for practical convergence.

The double descent curve has been observed for linear models, decision trees, and neural networks, suggesting it is a fundamental property of learning with gradient descent in the overparameterized regime.

> **Key insight:** Double descent overturns classical bias-variance tradeoff for modern overparameterized models — using far more parameters than samples can improve generalization, as long as implicit regularization (minimum norm, gradient noise) selects well-behaved solutions.

## Python: Loss Landscape Visualization and SAM vs SGD

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(0)

# ---- 2D Loss Landscape Visualization ----
# A non-convex function with saddle points and multiple minima
def loss_fn_2d(x, y):
    """Non-convex function: sum of Gaussians creates multiple basins."""
    # Two minima at (-1, -1) and (1, 1), saddle at origin
    z = (x**2 + y**2) - 2 * np.exp(-((x+1)**2 + (y+1)**2)) - 2 * np.exp(-((x-1)**2 + (y-1)**2))
    return z

xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
zz = loss_fn_2d(xx, yy)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Contour plot with GD trajectory
axes[0].contourf(xx, yy, zz, levels=30, cmap='viridis', alpha=0.8)
axes[0].contour(xx, yy, zz, levels=30, colors='white', alpha=0.3, linewidths=0.5)

def grad_2d(x, y, eps=1e-5):
    """Numerical gradient."""
    gx = (loss_fn_2d(x + eps, y) - loss_fn_2d(x - eps, y)) / (2 * eps)
    gy = (loss_fn_2d(x, y + eps) - loss_fn_2d(x, y - eps)) / (2 * eps)
    return np.array([gx, gy])

# Gradient descent trajectory from different initializations
for x_init, color, label in [([2.0, -2.0], 'red', 'GD from (2,-2)'),
                               ([-2.0, 2.0], 'orange', 'GD from (-2,2)')]:
    pt = np.array(x_init, dtype=float)
    traj = [pt.copy()]
    for _ in range(100):
        g = grad_2d(pt[0], pt[1])
        pt = pt - 0.1 * g
        traj.append(pt.copy())
    traj = np.array(traj)
    axes[0].plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, label=label)
    axes[0].plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)
    axes[0].plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=8)

axes[0].set_title('Non-Convex Loss Landscape\n(multiple minima + saddle at origin)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend(fontsize=8)

# ---- Simple SAM vs SGD demonstration ----
# Binary classification: sharp vs flat minimum
n_pts = 50
X_cls = np.vstack([np.random.randn(n_pts, 2) + [2, 0],
                   np.random.randn(n_pts, 2) + [-2, 0]])
y_cls = np.array([1]*n_pts + [-1]*n_pts, dtype=float)

def cls_loss(w):
    scores = X_cls @ w[:2] + w[2]
    return np.mean(np.log1p(np.exp(-y_cls * scores)))

def cls_grad(w):
    scores = X_cls @ w[:2] + w[2]
    probs = 1 / (1 + np.exp(y_cls * scores))
    gw = -X_cls.T @ (y_cls * probs) / len(y_cls)
    gb = -np.mean(y_cls * probs)
    return np.append(gw, gb)

def sgd_run(lr=0.1, T=500):
    w = np.zeros(3)
    losses = []
    for _ in range(T):
        losses.append(cls_loss(w))
        w -= lr * cls_grad(w)
    return w, losses

def sam_run(lr=0.1, rho=0.1, T=500):
    w = np.zeros(3)
    losses = []
    for _ in range(T):
        losses.append(cls_loss(w))
        g = cls_grad(w)
        g_norm = np.linalg.norm(g) + 1e-12
        eps_hat = rho * g / g_norm  # perturbation toward sharpest direction
        g_sam = cls_grad(w + eps_hat)  # gradient at perturbed point
        w -= lr * g_sam
    return w, losses

w_sgd, losses_sgd_cls = sgd_run(lr=0.5, T=300)
w_sam, losses_sam_cls = sam_run(lr=0.5, rho=0.3, T=300)

axes[1].semilogy(losses_sgd_cls, label='SGD', linewidth=2)
axes[1].semilogy(losses_sam_cls, label='SAM (ρ=0.3)', linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('SAM vs SGD on Simple Classification')
axes[1].legend()
axes[1].grid(True)

# ---- Stochastic Weight Averaging (SWA) ----
# SWA averages weights from the later part of the training trajectory.
# We continue SGD from the converged w_sgd and collect checkpoints.
w_swa_start = w_sgd.copy()
swa_checkpoints = []
lr_swa = 0.01  # constant LR for SWA phase
for step in range(100):
    grad = cls_grad(w_swa_start)
    w_swa_start = w_swa_start - lr_swa * grad
    if step % 10 == 9:  # collect every 10 steps
        swa_checkpoints.append(w_swa_start.copy())

w_swa = np.mean(swa_checkpoints, axis=0)

# Compare losses
test_losses = []
labels_test = ['SGD final', 'SAM final', 'SWA (avg over 100 steps)']
for w_test, lbl in [(w_sgd, 'SGD'), (w_sam, 'SAM'), (w_swa, 'SWA')]:
    axes[2].bar(lbl, cls_loss(w_test), alpha=0.7)
axes[2].set_ylabel('Final Training Loss')
axes[2].set_title('SGD vs SAM vs SWA\nFinal Loss Comparison')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.savefig('nonconvex_landscapes.png', dpi=150)
plt.show()

# Sharpness estimate: max eigenvalue of Hessian at solution
def hessian_numerical(w, eps=1e-4):
    d = len(w)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = eps; ej[j] = eps
            H[i, j] = (cls_loss(w+ei+ej) - cls_loss(w+ei-ej)
                       - cls_loss(w-ei+ej) + cls_loss(w-ei-ej)) / (4*eps**2)
    return H

H_sgd = hessian_numerical(w_sgd)
H_sam = hessian_numerical(w_sam)
print(f"SGD solution — max Hessian eigenvalue (sharpness): {np.linalg.eigvalsh(H_sgd).max():.4f}")
print(f"SAM solution — max Hessian eigenvalue (sharpness): {np.linalg.eigvalsh(H_sam).max():.4f}")
print("SAM typically finds a flatter minimum (smaller max eigenvalue).")
```

:::quiz
question: "Near a saddle point in high dimensions, why does stochastic gradient descent (SGD) typically escape while full-batch gradient descent may stagnate?"
options:
  - "SGD uses a larger effective learning rate, which causes the optimizer to jump over saddle points"
  - "The gradient at a saddle point is exactly zero for full-batch GD, but stochastic gradient noise provides perturbations along escape directions (negative curvature directions)"
  - "SGD computes the Hessian implicitly, allowing it to detect negative curvature and move in the escape direction"
  - "Full-batch gradient descent converges to saddle points because it computes the exact gradient, while SGD approximates it"
correct: 1
explanation: "At a strict saddle point, the full gradient ∇f(x*) = 0 exactly, so full-batch GD makes no progress (dx/dt = 0). SGD computes ∇f_i(x*) for a random sample, which is generally non-zero even at a saddle point (individual sample gradients don't average to zero at the same point as the full gradient). These stochastic perturbations have components in the negative curvature directions (escape directions), allowing SGD to drift away from the saddle over time. This is a key advantage of gradient noise in non-convex optimization."
:::

:::quiz
question: "The double descent phenomenon shows test error can decrease after the interpolation threshold (where parameters ≈ data points). What mechanism allows an overparameterized model to interpolate training data yet still generalize?"
options:
  - "Overparameterized models have higher effective capacity, which reduces approximation error sufficiently to overcome increased variance"
  - "Gradient descent implicitly finds the minimum L2-norm solution among all interpolating solutions, which concentrates the model's capacity on the signal rather than fitting noise directions"
  - "Overparameterized models can memorize training data and generalize because neural network optimization is equivalent to kernel regression"
  - "The interpolation threshold is a numerical artifact caused by finite precision arithmetic in gradient computation"
correct: 1
explanation: "Among the infinite set of interpolating solutions (all achieving zero training loss), gradient descent preferentially converges to the minimum L2-norm solution. If the true signal lies in a low-dimensional subspace, the minimum-norm interpolant spreads the remaining 'fitting budget' over noise directions in a way that has small effect on test predictions. This 'benign overfitting' — proven rigorously by Bartlett et al. (2020) for linear models in high dimensions — explains why overparameterized models can interpolate yet generalize. The key is that implicit regularization from gradient descent selects a structured solution."
:::

:::quiz
question: "SAM (Sharpness-Aware Minimization) minimizes max_{‖ε‖≤ρ} L(θ+ε). What is the computational cost compared to standard gradient descent, and what does the inner maximization's solution ε̂ = ρ·∇L/‖∇L‖ represent geometrically?"
options:
  - "SAM costs 4× per step because it requires computing second-order information; ε̂ points toward the nearest saddle point"
  - "SAM costs 2× per step (two gradient evaluations); ε̂ is the normalized gradient direction, which is the steepest ascent direction — the point on the ρ-ball where the loss is locally maximized"
  - "SAM has the same cost as gradient descent because the inner maximization has a closed form that requires only the current gradient; ε̂ is the eigenvector of the Hessian"
  - "SAM costs 2× per step; ε̂ points toward the nearest local minimum in the ρ-ball, representing the most stable parameter configuration"
correct: 1
explanation: "SAM requires two gradient computations per update: (1) compute ∇L(θ) to find ε̂ = ρ·∇L(θ)/‖∇L(θ)‖, and (2) compute ∇L(θ + ε̂) to perform the actual parameter update. This doubles the per-step cost. The perturbation ε̂ lies on the boundary of the ρ-ball in the gradient direction — this is the first-order approximation to the argmax of L(θ+ε) over ‖ε‖ ≤ ρ, obtained by linearizing L around θ. Geometrically, it moves to the point on the ρ-ball that most increases the loss, and then takes a gradient step there, steering the optimizer away from sharp regions."
:::
