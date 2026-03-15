---
title: "Matrix Calculus for Deep Learning"
estimatedMinutes: 35
tags: ["matrix-calculus", "jacobian", "hessian", "backpropagation", "gradient"]
prerequisites: ["l1-svd-low-rank", "l2-eigendecomposition"]
---

# Matrix Calculus for Deep Learning

Every backward pass in a neural network is an exercise in matrix calculus. Autograd frameworks handle the mechanics, but understanding the underlying mathematics is essential for implementing custom layers, debugging gradient pathologies, and grasping why certain architectures train better than others. This lesson builds the matrix calculus toolkit from conventions through the key identities, then connects them directly to backpropagation, the Hessian, and the Fisher information matrix.

## Layout Conventions

Matrix calculus has two competing conventions, and conflating them is a perennial source of bugs and confusion. We adopt the **numerator layout** (also called the Jacobian formulation):

> **Refresher:** The Jacobian is the multivariable generalization of the derivative. For a scalar function $f: \mathbb{R} \to \mathbb{R}$, the derivative $f'(x)$ tells you how $f$ changes per unit change in $x$. For a vector function $f: \mathbb{R}^m \to \mathbb{R}^n$, the Jacobian $J \in \mathbb{R}^{n \times m}$ is the matrix of all partial derivatives: entry $J_{ij}$ tells you how output $y_i$ changes per unit change in input $x_j$. The Jacobian is to vector functions what the derivative is to scalar functions.

For $f: \mathbb{R}^m \to \mathbb{R}^n$, the Jacobian is:

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times m}, \quad J_{ij} = \frac{\partial y_i}{\partial x_j}$$

Row $i$ of $J$ is the gradient of $y_i$ with respect to all inputs. This is what PyTorch's `torch.autograd.functional.jacobian` returns, what JAX's `jacfwd` and `jacrev` compute, and the convention used in most ML research.

For a scalar-valued function $f: \mathbb{R}^m \to \mathbb{R}$, the gradient $\nabla f = \frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times m}$ is a row vector in numerator layout. In practice, frameworks return it as a column vector to match the shape of $x$. Be aware of this inconsistency.

In **denominator layout** (used in some statistics texts), the Jacobian is transposed: $\mathbb{R}^{m \times n}$, and scalar gradients are column vectors. We follow the numerator convention, which matches PyTorch, JAX, and Goodfellow et al.

> **Key insight:** When you see a matrix derivative, the first question is always: which layout convention? Numerator layout makes chain rules compose naturally as matrix products. Most bugs in manual gradient derivations trace back to a layout mismatch.

## Essential Matrix Derivatives

These identities form the working vocabulary of matrix calculus in ML. Each is stated in numerator layout.

**Vector derivatives:**

$$\frac{\partial}{\partial x}(a^T x) = a^T \quad \Rightarrow \quad \nabla_x (a^T x) = a$$

The gradient of a linear function is the coefficient vector — the simplest case.

$$\frac{\partial}{\partial x}(x^T A x) = x^T(A + A^T) = 2x^T A \quad \text{when } A = A^T$$

For a quadratic form with symmetric $A$: in numerator layout this is the row vector $2x^TA$. As a gradient column vector (as returned by PyTorch and most frameworks), it is $2Ax$. The two are transposes of each other — the distinction matters when you chain Jacobians. This identity governs the gradient of $L_2$ regularization, Mahalanobis distances, and any quadratic loss.

**Matrix derivatives (for scalar-valued functions of matrices):**

$$\frac{\partial}{\partial X} \text{tr}(AX) = A^T$$

The trace derivative is the workhorse of matrix calculus. Since many scalar losses can be written as traces (e.g., $\|M\|_F^2 = \text{tr}(M^T M)$), this identity handles a large fraction of practical cases.

$$\frac{\partial}{\partial X} \log \det X = X^{-T}$$

This appears in the log-likelihood of Gaussian models (the normalization term involves $\log \det \Sigma$), variational inference, and normalizing flows.

$$\frac{\partial}{\partial X} \|AXB - C\|_F^2 = 2A^T(AXB - C)B^T$$

The gradient of a Frobenius-norm regression objective. Setting this to zero gives the matrix normal equation, which generalizes ordinary least squares.

## The Chain Rule in Matrix Form

> **Intuition:** The matrix chain rule is the same idea as the 1D chain rule $\frac{d\ell}{dx} = \frac{d\ell}{dy} \cdot \frac{dy}{dx}$, just with matrices. In 1D, you multiply two scalars; in the matrix case, you multiply two Jacobians. The only new wrinkle is that matrix multiplication is not commutative, so the order matters — upstream gradient on the left, local Jacobian on the right. Every layer in a neural network applies this rule once.

The chain rule is the engine of backpropagation. For a composition $\ell = \ell(Y(X))$:

$$\frac{\partial \ell}{\partial X} = \frac{\partial \ell}{\partial Y} \cdot \frac{\partial Y}{\partial X}$$

In practice, $\frac{\partial Y}{\partial X}$ is a 4th-order tensor for matrix-to-matrix maps, which is unwieldy. The standard approach is to work with the **differential**: if $Y = f(X)$, then:

$$d\ell = \text{tr}\left(\frac{\partial \ell}{\partial Y}^T dY\right) = \text{tr}\left(\frac{\partial \ell}{\partial Y}^T \frac{\partial f}{\partial X}[dX]\right)$$

By rearranging the trace expression, we identify $\frac{\partial \ell}{\partial X}$. This "trace trick" is the standard technique for deriving gradients of matrix expressions.

The pattern that every backward pass implements is: **upstream gradient times local Jacobian**. Each layer receives $\frac{\partial \ell}{\partial Y}$ from above and computes $\frac{\partial \ell}{\partial X}$ by multiplying with the local derivative.

## Backpropagation as Matrix Calculus

> **Intuition:** Backpropagation is just the matrix chain rule applied layer by layer, from the loss back to the inputs. At each layer, you receive the upstream gradient (how the loss changes with respect to this layer's output) and multiply by the local Jacobian (how this layer's output changes with respect to its input and parameters). Autograd frameworks automate this bookkeeping — but they are computing exactly the matrix chain rule you can derive by hand, as the next section shows.

Consider a single linear layer: $Y = XW + \mathbf{1}b^T$, where $X \in \mathbb{R}^{B \times d_{\text{in}}}$ is the input batch, $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, and $b \in \mathbb{R}^{d_{\text{out}}}$.

Let $\delta = \frac{\partial \ell}{\partial Y} \in \mathbb{R}^{B \times d_{\text{out}}}$ be the upstream gradient. We derive the backward pass from first principles.

**Gradient with respect to $W$:** Starting from $Y = XW$:

$$d\ell = \text{tr}(\delta^T \, dY) = \text{tr}(\delta^T \, X \, dW) = \text{tr}((X^T \delta)^T \, dW)$$

Reading off the gradient: $\frac{\partial \ell}{\partial W} = X^T \delta$.

Each column of $X^T \delta$ is the input features weighted by the corresponding output gradient — the outer product structure that makes gradient computation a matrix multiply.

**Gradient with respect to $X$:** Similarly:

$$d\ell = \text{tr}(\delta^T \, dX \, W) = \text{tr}((W \delta^T)^T \, dX^T) = \text{tr}((\delta W^T)^T \, dX)$$

So $\frac{\partial \ell}{\partial X} = \delta W^T$.

The upstream gradient $\delta$ is projected back through the transpose of the weight matrix. This is why the backward pass through a linear layer is a matrix multiply with $W^T$ — it is literally the chain rule applied to $Y = XW$.

**Gradient with respect to $b$:** $\frac{\partial \ell}{\partial b} = \delta^T \mathbf{1} = \sum_i \delta_i$ — sum of upstream gradients across the batch.

> **Key insight:** The forward pass multiplies by $W$; the backward pass multiplies by $W^T$. The weight gradient is the outer product of the input and the upstream gradient. These three operations are the entire backward pass of a linear layer, derived purely from matrix calculus.

## Jacobians of Common Nonlinearities

Understanding the local Jacobian of each layer is essential for tracing gradient flow. For **elementwise activations** $\sigma: \mathbb{R}^n \to \mathbb{R}^n$ where $y_i = \sigma(x_i)$ independently:

$$J_\sigma = \text{diag}(\sigma'(x_1), \sigma'(x_2), \ldots, \sigma'(x_n))$$

The Jacobian is diagonal — gradient computation costs $O(n)$, not $O(n^2)$. For ReLU: $\sigma'(x_i) = \mathbf{1}[x_i > 0]$. For sigmoid: $\sigma'(x_i) = \sigma(x_i)(1 - \sigma(x_i))$.

> **Intuition:** The softmax Jacobian has off-diagonal terms because softmax outputs are coupled — they must sum to 1. Increasing logit $x_j$ raises $y_j$ but forces all other $y_i$ to decrease proportionally. So $\partial y_i / \partial x_j$ for $i \neq j$ is negative (specifically $-y_i y_j$): raising a competing logit always reduces your probability. This coupling is precisely what makes softmax output a probability simplex rather than independent predictions.

**Softmax Jacobian** is more interesting. For $y = \text{softmax}(x)$ where $y_i = e^{x_i}/\sum_j e^{x_j}$:

$$\frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j)$$

In matrix form: $J_{\text{softmax}} = \text{diag}(y) - yy^T$. This is a rank-1 update of a diagonal matrix.

> **Key insight:** The softmax Jacobian is symmetric and has rank $n-1$ (it is singular — softmax outputs sum to 1, so one degree of freedom is lost). In practice, the backward pass of cross-entropy + softmax is simplified to $\delta = \hat{y} - y_{\text{true}}$ by combining both Jacobians analytically.

## The Hessian Matrix

The Hessian of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is the matrix of second derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

For a twice-differentiable function, $H$ is symmetric. At a critical point ($\nabla f = 0$), the eigenvalues of $H$ classify the geometry:

- All $\lambda_i > 0$: strict local minimum (positive definite Hessian)
- All $\lambda_i < 0$: strict local maximum
- Mixed signs: saddle point

For neural network loss functions, the Hessian is enormous ($p \times p$ for $p$ parameters) and never formed explicitly. But its spectral properties matter enormously:

- **Sharpness**: the largest eigenvalue $\lambda_{\max}(H)$ measures how curved the loss surface is at a minimum. Sharper minima (larger $\lambda_{\max}$) correlate with worse generalization. This motivates Sharpness-Aware Minimization (SAM), which explicitly seeks flat minima.
- **Convergence rate**: gradient descent on a quadratic converges at rate $O((\kappa(H) - 1)/(\kappa(H) + 1))$ per step, where $\kappa(H)$ is the condition number of the Hessian.
- **Saddle points**: in high-dimensional loss landscapes, most critical points are saddle points (the Hessian has both positive and negative eigenvalues). The fraction of negative eigenvalues correlates with the loss value (Baldi & Hornik, Choromanska et al.).

## The Fisher Information Matrix

> **Refresher:** Fisher information measures how much the distribution $p(x|\theta)$ changes when you perturb $\theta$. If $F$ is large in some direction, moving $\theta$ that way dramatically changes the distribution — the data is highly informative about $\theta$ in that direction. If $F$ is small, the distribution barely changes — you can move $\theta$ without affecting what the model predicts. This curvature of the likelihood surface is exactly what makes the Fisher a natural metric for parameter space.

The Fisher information matrix is the bridge between statistics and optimization:

$$F = \mathbb{E}_{x \sim p(x|\theta)}\left[\nabla_\theta \log p(x|\theta) \, \nabla_\theta \log p(x|\theta)^T\right]$$

This is the covariance of the score function. Three key properties:

1. $F$ equals the negative expected Hessian of the log-likelihood: $F = -\mathbb{E}[\nabla^2 \log p(x|\theta)]$. It measures how quickly the log-likelihood changes as we move in parameter space.

2. $F$ equals the Hessian of the KL divergence $D_{\text{KL}}(p_\theta \| p_{\theta + d\theta})$ at $d\theta = 0$. So the Fisher defines a local metric on the space of distributions.

3. The **natural gradient** is $\tilde{\nabla} = F^{-1} \nabla$ — the steepest descent direction in the KL divergence metric rather than Euclidean metric. This is the theoretically optimal update direction, invariant to reparameterization.

In practice, computing $F^{-1}$ is intractable for large models. K-FAC approximates $F$ as a block-diagonal of Kronecker products, and Adam can be seen as a diagonal approximation to natural gradient.

> **Key insight:** The Fisher information matrix is simultaneously the Hessian of KL divergence, the curvature of the log-likelihood, and the metric tensor of the statistical manifold. Natural gradient descent follows this geometry, which is why it converges faster than vanilla SGD in theory — and why its approximations (Adam, K-FAC) work so well in practice.

## Python: Gradient Verification

```python
import torch
import torch.nn as nn

# Manual gradient computation for a linear layer
torch.manual_seed(42)
B, d_in, d_out = 32, 64, 16
X = torch.randn(B, d_in, requires_grad=True)
W = torch.randn(d_in, d_out, requires_grad=True)
b = torch.randn(d_out, requires_grad=True)

# Forward pass
Y = X @ W + b
loss = (Y ** 2).sum()  # simple scalar loss

# Autograd backward
loss.backward()

# Manual gradients (from our derivations above)
delta = 2 * Y.detach()              # d(loss)/dY = 2Y for sum-of-squares
grad_W_manual = X.detach().T @ delta  # X^T @ delta
grad_X_manual = delta @ W.detach().T  # delta @ W^T
grad_b_manual = delta.sum(dim=0)      # sum over batch

# Verify
print(f"W grad error: {(W.grad - grad_W_manual).abs().max():.2e}")
print(f"X grad error: {(X.grad - grad_X_manual).abs().max():.2e}")
print(f"b grad error: {(b.grad - grad_b_manual).abs().max():.2e}")

# Numerical Hessian for a small function
def f(x):
    return (x ** 3).sum() + (x[0] * x[1]) ** 2

x0 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
H_numerical = torch.autograd.functional.hessian(f, x0)
print(f"\nNumerical Hessian:\n{H_numerical}")
print(f"Hessian is symmetric: {torch.allclose(H_numerical, H_numerical.T)}")
```

:::quiz
question: "In the backward pass of Y = XW, the gradient of the loss with respect to X is delta @ W^T. Why is it W^T and not W?"
options:
  - "W^T is used to match the shape: delta is B x d_out and W^T is d_out x d_in, giving B x d_in"
  - "It's an arbitrary convention that could be either W or W^T"
  - "Because the forward pass already used W, so the backward pass must use its inverse"
  - "W^T is used because the chain rule for Y = XW gives dL/dX = (dL/dY)(dY/dX)^T = delta @ W^T, following from the matrix chain rule"
correct: 3
explanation: "The chain rule applied to Y = XW gives dL/dX_ij = sum_k (dL/dY_ik)(dY_ik/dX_ij) = sum_k delta_ik * W_jk = (delta @ W^T)_ij. The transpose arises naturally from the calculus, not from shape matching (though the shapes must also work out). Option A describes a consequence, not the cause."
:::

:::quiz
question: "The Fisher information matrix F for a model equals the Hessian of which quantity?"
options:
  - "The training loss at the current parameters"
  - "The KL divergence D_KL(p_theta || p_{theta+d_theta}) evaluated at d_theta = 0"
  - "The L2 regularization term"
  - "The entropy of the model's output distribution"
correct: 1
explanation: "The Fisher information matrix equals the Hessian of the KL divergence between the model distribution at theta and at theta + d_theta, evaluated at d_theta = 0. This makes F a local metric on the space of distributions. It also equals the negative expected Hessian of the log-likelihood, but it is precisely the KL Hessian — which is why natural gradient (F^{-1} nabla) follows the geometry of distribution space."
:::

:::quiz
question: "At a critical point of the loss, the Hessian has eigenvalues {50, 2, 0.1, -0.01, -3}. What type of critical point is this?"
options:
  - "A local minimum because most eigenvalues are positive"
  - "A saddle point because the Hessian has both positive and negative eigenvalues"
  - "A local maximum because the Hessian is indefinite"
  - "Cannot be determined without knowing the gradient"
correct: 1
explanation: "A local minimum requires ALL eigenvalues to be non-negative. Here we have negative eigenvalues (-0.01 and -3), so there exist directions of negative curvature — the loss decreases in those directions. This is a saddle point. In high-dimensional neural network loss surfaces, most critical points are saddle points of exactly this type."
:::
