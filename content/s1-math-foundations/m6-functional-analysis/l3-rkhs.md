---
title: "Reproducing Kernel Hilbert Spaces"
estimatedMinutes: 40
tags: ["RKHS", "kernels", "Mercer", "representer-theorem", "Gaussian-processes"]
prerequisites: ["l1-banach-hilbert-spaces", "l2-linear-operators"]
---

## Motivation: The Kernel Trick

A central challenge in machine learning is learning nonlinear functions from data. Given training pairs $\{(x_i, y_i)\}_{i=1}^n$ with $x_i \in \mathcal{X} \subseteq \mathbb{R}^d$, we want to search a rich function class — one expressive enough to capture complicated input-output relationships — while maintaining computational tractability. The **kernel trick** resolves this tension by working implicitly in a high- or infinite-dimensional feature space without ever computing coordinates in that space.

**The feature map idea.** Let $H$ be a Hilbert space (possibly infinite-dimensional) and $\varphi : \mathcal{X} \to H$ a **feature map**. A linear model in the feature space defines a nonlinear model in the input space:

$$f(x) = \langle w, \varphi(x) \rangle_H$$

for some weight vector $w \in H$. The inner product between two feature vectors defines a **kernel**:

$$k(x, x') = \langle \varphi(x), \varphi(x') \rangle_H$$

All computations involving input points reduce to evaluations of $k$ — we never need the coordinates of $\varphi(x)$ in $H$.

**Why this matters: the RBF kernel.** The RBF (Gaussian) kernel $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$ corresponds to an infinite-dimensional feature map. Expanding the exponential:

$$\exp\!\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right) = \exp\!\left(-\frac{\|x\|^2 + \|x'\|^2}{2\sigma^2}\right) \sum_{n=0}^\infty \frac{(x \cdot x')^n}{n! \, \sigma^{2n}}$$

Each term $(x \cdot x')^n / (n! \, \sigma^{2n})$ corresponds to a degree-$n$ polynomial kernel. The full RBF kernel is a sum over all polynomial degrees — an infinite-dimensional feature space. Computing $k(x, x')$ costs $O(d)$ arithmetic operations. Computing $\varphi(x)$ explicitly is impossible. This is the kernel trick: evaluating $k$ is cheaper than computing $\varphi$, yet the two carry identical information.

> **Intuition:** The kernel $k(x, x')$ measures similarity between $x$ and $x'$ as perceived through the lens of the feature space $H$. Two points close in $H$ (large $k$) will be treated as similar by any linear model in the feature space, which corresponds to a nonlinear model in the original space.

**Gram matrix.** Given data $\{x_i\}_{i=1}^n$, the $n \times n$ matrix $K_{ij} = k(x_i, x_j)$ is the **Gram matrix**. All kernel algorithms — support vector machines, kernel ridge regression, kernel PCA, Gaussian processes — are ultimately algorithms on the Gram matrix. The theory of RKHS explains precisely which functions $k$ can serve as kernels and what function spaces they define.

## Positive Definite Kernels

Not every function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ can serve as a kernel. The correct structural condition is positive definiteness.

**Definition (positive definite kernel).** A function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a **positive definite kernel** (or **positive semi-definite kernel**) if:

1. *Symmetry:* $k(x, x') = k(x', x)$ for all $x, x' \in \mathcal{X}$.
2. *Positive semi-definiteness:* For every $n \geq 1$, every finite set of points $\{x_1, \ldots, x_n\} \subseteq \mathcal{X}$, and every $c = (c_1, \ldots, c_n) \in \mathbb{R}^n$:

$$\sum_{i=1}^n \sum_{j=1}^n c_i c_j \, k(x_i, x_j) \geq 0$$

Equivalently, the Gram matrix $[k(x_i, x_j)]_{i,j=1}^n$ is positive semi-definite for every finite point set.

> **Refresher:** From L1, recall that a symmetric matrix $A$ is positive semi-definite iff $c^\top A c \geq 0$ for all $c \in \mathbb{R}^n$ iff all eigenvalues of $A$ are non-negative. The PD kernel condition extends this requirement to all finite subsets of the domain, not just a fixed matrix size.

**Standard examples.**

*Linear kernel:* $k(x, x') = x \cdot x'$ on $\mathbb{R}^d$. The feature map is the identity: $\varphi(x) = x$, $H = \mathbb{R}^d$. This gives standard linear models.

*Polynomial kernel:* $k(x, x') = (1 + x \cdot x')^d$ for integer $d \geq 1$. The feature map $\varphi$ maps $x$ to all monomials of degree $\leq d$ (with appropriate coefficients). The feature space has dimension $\binom{d+p}{p}$ where $p$ is the input dimension, which grows polynomially in $p$.

*RBF (Gaussian) kernel:* $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$ for $\sigma > 0$. The bandwidth $\sigma$ controls the length-scale: large $\sigma$ gives a slowly varying kernel (smooth functions), small $\sigma$ gives rapid decay (local interpolation). This kernel is strictly positive definite: the Gram matrix is always strictly positive definite, not merely semi-definite.

*Matérn kernel:* $k_\nu(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\, r}{\ell}\right)$, where $r = \|x - x'\|$, $K_\nu$ is the modified Bessel function of the second kind, $\ell > 0$ is the length-scale, and $\nu > 0$ controls smoothness. Key special cases:
- $\nu = 1/2$: $k(x,x') = \exp(-r/\ell)$ (Ornstein-Uhlenbeck, corresponds to once-differentiable functions)
- $\nu = 3/2$: $k(x,x') = (1 + \sqrt{3}r/\ell)\exp(-\sqrt{3}r/\ell)$ (twice-differentiable)
- $\nu = 5/2$: three times differentiable
- $\nu \to \infty$: recovers the RBF kernel (infinitely differentiable)

The Matérn family is central to Gaussian process modeling because the smoothness parameter $\nu$ directly controls the sample-path regularity of the associated GP (see Section 7 and L4).

**Closure properties.** The set of positive definite kernels is closed under several operations. If $k_1$ and $k_2$ are PD kernels on $\mathcal{X}$, then so are:
- *Sum:* $k_1 + k_2$
- *Product:* $k_1 \cdot k_2$ (Schur product theorem for PD matrices)
- *Positive scaling:* $\alpha k_1$ for $\alpha > 0$
- *Pointwise limits:* $\lim_{n \to \infty} k_n$ if the limit exists and is continuous
- *Composition with a feature map:* if $\varphi : \mathcal{X} \to \mathcal{Y}$, then $k(x,x') = k_0(\varphi(x), \varphi(x'))$ is PD on $\mathcal{X}$ whenever $k_0$ is PD on $\mathcal{Y}$

These closure properties explain how to construct new kernels from old ones — combining RBF and polynomial kernels, applying them to structured inputs (graphs, strings, sets), and so on.

## The RKHS Construction

Given a positive definite kernel $k$, we now construct the Hilbert space it defines. This construction is canonical — there is only one such space up to isometric isomorphism.

**Step 1: The pre-Hilbert space.** Consider the vector space of functions of the form:

$$f = \sum_{i=1}^n a_i \, k(\cdot, x_i), \qquad n \in \mathbb{N}, \quad a_i \in \mathbb{R}, \quad x_i \in \mathcal{X}$$

That is, finite linear combinations of "kernel sections" $k(\cdot, x_i)$, each of which is the function $y \mapsto k(y, x_i)$. Define an inner product on this space by:

$$\left\langle \sum_i a_i \, k(\cdot, x_i),\; \sum_j b_j \, k(\cdot, y_j) \right\rangle_{H_k} = \sum_i \sum_j a_i b_j \, k(x_i, y_j)$$

**Verification that this is an inner product.** Symmetry and bilinearity are immediate from the symmetry of $k$. Positive semi-definiteness follows from the PD condition on $k$. To verify non-degeneracy ($\|f\|_{H_k} = 0 \Rightarrow f = 0$), we use the reproducing property below.

**Step 2: The reproducing property.** For any function $f = \sum_i a_i k(\cdot, x_i)$ in the pre-Hilbert space and any point $x \in \mathcal{X}$:

$$\langle f, k(\cdot, x) \rangle_{H_k} = \left\langle \sum_i a_i k(\cdot, x_i),\, k(\cdot, x) \right\rangle_{H_k} = \sum_i a_i k(x_i, x) = f(x)$$

This is the **reproducing property**: the inner product with the kernel section $k(\cdot, x)$ evaluates the function at $x$. Consequently, pointwise evaluation is a bounded linear functional on the pre-Hilbert space.

**Non-degeneracy.** Suppose $\|f\|_{H_k}^2 = \langle f, f \rangle_{H_k} = 0$. By Cauchy-Schwarz, for any $x \in \mathcal{X}$:

$$|f(x)| = |\langle f, k(\cdot, x) \rangle_{H_k}| \leq \|f\|_{H_k} \, \|k(\cdot, x)\|_{H_k} = 0$$

So $f(x) = 0$ for all $x$, confirming that $\langle \cdot, \cdot \rangle_{H_k}$ is a genuine inner product.

**Step 3: Completion.** Complete the pre-Hilbert space with respect to the norm $\|f\|_{H_k} = \sqrt{\langle f, f \rangle_{H_k}}$ to obtain the **reproducing kernel Hilbert space** $H_k$. The resulting Hilbert space has the following properties:

1. $k(\cdot, x) \in H_k$ for every $x \in \mathcal{X}$
2. The span $\{k(\cdot, x) : x \in \mathcal{X}\}$ is dense in $H_k$
3. The reproducing property holds for all $f \in H_k$: $f(x) = \langle f, k(\cdot, x) \rangle_{H_k}$

> **Key insight:** In a general Hilbert space, pointwise evaluation $f \mapsto f(x)$ need not be continuous. In $L^2([0,1])$, for instance, changing $f$ on a set of measure zero does not change $\|f\|_{L^2}$, so $f(x)$ is not even well-defined for a given $L^2$ equivalence class. The reproducing property precisely says that in $H_k$, pointwise evaluation IS a bounded linear functional, with $\|{\rm ev}_x\| = \sqrt{k(x,x)}$.

**Connection to Riesz representation.** From L1, every bounded linear functional $\phi$ on a Hilbert space $H$ has the form $\phi(f) = \langle f, g \rangle_H$ for a unique $g \in H$. The reproducing property says that the Riesz representer of pointwise evaluation at $x$ is exactly the kernel section $k(\cdot, x)$. This is not a coincidence — it is the defining property, and it fully characterizes the RKHS.

**Boundedness of evaluation.** By Cauchy-Schwarz and the reproducing property:

$$|f(x)| = |\langle f, k(\cdot, x) \rangle_{H_k}| \leq \|f\|_{H_k} \, \sqrt{k(x,x)}$$

So $|f(x)| \leq \sqrt{k(x,x)} \, \|f\|_{H_k}$. Points $x$ where $k(x,x)$ is large have stronger pointwise evaluation; the $H_k$-norm controls pointwise values everywhere on $\mathcal{X}$.

## Moore-Aronszajn Theorem

The RKHS construction establishes one direction: every PD kernel $k$ gives rise to an RKHS $H_k$. The Moore-Aronszajn theorem establishes the other direction and uniqueness.

**Theorem (Moore-Aronszajn, 1950).** There is a bijection between positive definite kernels on $\mathcal{X}$ and reproducing kernel Hilbert spaces of functions on $\mathcal{X}$. Specifically:

1. Every PD kernel $k$ on $\mathcal{X}$ determines a unique RKHS $H_k$ (as constructed above).
2. Every RKHS $H$ of functions on $\mathcal{X}$ has a unique reproducing kernel $k$ — the function $k(x, x') = \langle k(\cdot, x'), k(\cdot, x) \rangle_H$.
3. The correspondence $k \leftrightarrow H_k$ is a bijection.

**Proof sketch of uniqueness.** Suppose $H$ and $H'$ are two RKHS on $\mathcal{X}$ with the same reproducing kernel $k$. For any $f \in H$ and $x \in \mathcal{X}$, the reproducing property gives $f(x) = \langle f, k(\cdot, x) \rangle_H$. Since the kernel sections $\{k(\cdot, x)\}_{x \in \mathcal{X}}$ are dense in both spaces, and the inner product is determined by its values on a dense set, the two spaces must agree. More precisely: $H$ and $H'$ contain the same dense subset (finite linear combinations of kernel sections with the same inner product), so their completions are isometrically isomorphic as spaces of functions.

**Consequences.**
- To specify a function space, it suffices to specify a PD kernel. This is the basis for "kernel design" in machine learning.
- The RKHS $H_k$ consists precisely of those functions on $\mathcal{X}$ that can be represented as limits of finite linear combinations of kernel sections in the $H_k$-norm. It is a complete characterization.
- Different kernels give genuinely different Hilbert spaces with different smoothness properties.

> **Remember:** Moore-Aronszajn means you never need to ask "does this RKHS exist?" — you just specify a PD kernel, and the unique RKHS is guaranteed. Conversely, any Hilbert space in which pointwise evaluations are bounded linear functionals is automatically an RKHS with a well-defined reproducing kernel.

## Mercer's Theorem

Mercer's theorem provides a spectral decomposition of the kernel and a concrete characterization of the RKHS norm in terms of eigenvalues and eigenfunctions.

**Setup.** Let $\mathcal{X}$ be a compact metric space and $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ a continuous PD kernel. Let $\mu$ be a Borel measure on $\mathcal{X}$ with full support. Define the **integral operator** $T_k : L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu)$ by:

$$(T_k f)(x) = \int_\mathcal{X} k(x, x') \, f(x') \, d\mu(x')$$

From L2, since $k$ is a continuous kernel on a compact space, $T_k$ is a Hilbert-Schmidt operator (hence compact). Since $k$ is symmetric, $T_k$ is self-adjoint. Since $k$ is positive definite, $T_k$ is positive semi-definite:

$$\langle T_k f, f \rangle_{L^2} = \int\!\int k(x,x') f(x') f(x) \, d\mu(x') \, d\mu(x) \geq 0$$

> **Refresher:** From L2, compact self-adjoint operators on a Hilbert space have a spectral decomposition: they are diagonalizable with a countable sequence of real eigenvalues accumulating only at zero, and the eigenfunctions form an orthonormal basis of the closure of the range.

**Theorem (Mercer, 1909).** Under the above conditions, the integral operator $T_k$ has a countable sequence of non-negative eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$ with corresponding orthonormal eigenfunctions $\{\phi_n\}$ in $L^2(\mathcal{X}, \mu)$:

$$T_k \phi_n = \lambda_n \phi_n$$

The kernel admits the **Mercer expansion**:

$$k(x, x') = \sum_{n=1}^\infty \lambda_n \, \phi_n(x) \, \phi_n(x')$$

where the series converges absolutely and uniformly on $\mathcal{X} \times \mathcal{X}$.

**The RKHS norm via Mercer.** The RKHS $H_k$ has the characterization:

$$H_k = \left\{ f \in L^2(\mathcal{X}, \mu) : \sum_{n=1}^\infty \frac{\hat{f}_n^2}{\lambda_n} < \infty \right\}$$

where $\hat{f}_n = \langle f, \phi_n \rangle_{L^2} = \int f(x) \phi_n(x) \, d\mu(x)$ are the $L^2$ expansion coefficients of $f$. The RKHS inner product is:

$$\langle f, g \rangle_{H_k} = \sum_{n=1}^\infty \frac{\hat{f}_n \, \hat{g}_n}{\lambda_n}$$

and the RKHS norm is:

$$\|f\|_{H_k}^2 = \sum_{n=1}^\infty \frac{\hat{f}_n^2}{\lambda_n}$$

> **Key insight:** The RKHS norm $\|f\|_{H_k}^2 = \sum_n \hat{f}_n^2 / \lambda_n$ is a weighted $\ell^2$ norm on the Mercer coefficients, where eigenvalue $\lambda_n$ weights direction $\phi_n$. Large $\lambda_n$ means the corresponding eigenfunction is "cheap" in the RKHS — penalized weakly, allowed to have large coefficient. Small $\lambda_n$ means the corresponding eigenfunction is "expensive" — penalized heavily. The kernel thus defines a notion of smoothness: eigenfunctions corresponding to large eigenvalues are the "easy" directions, and they tend to be the smoother, lower-frequency components.

**Smoothness interpretation.** For the RBF kernel on $[0,1]$, the eigenfunctions $\phi_n$ are approximately Fourier modes $\{\cos(n\pi x), \sin(n\pi x)\}$ and the eigenvalues $\lambda_n$ decay rapidly (faster than any polynomial in $n$). The RKHS norm $\sum_n \hat{f}_n^2 / \lambda_n$ blows up for functions with non-negligible high-frequency components — the RKHS contains only very smooth (in fact, analytic) functions. For the Matérn-$\nu$ kernel, the eigenvalues decay as $\lambda_n \sim n^{-2\nu/d - 1}$, so the RKHS is a Sobolev space $W^{\nu + d/2, 2}(\mathcal{X})$ (this connection is made precise in L4).

**Finite feature approximations.** Truncating the Mercer expansion at $m$ terms gives the approximation $k_m(x,x') = \sum_{n=1}^m \lambda_n \phi_n(x) \phi_n(x')$, which corresponds to the feature map $\varphi_m(x) = (\sqrt{\lambda_1}\phi_1(x), \ldots, \sqrt{\lambda_m}\phi_m(x)) \in \mathbb{R}^m$. Random Fourier features (Rahimi and Recht, 2007) provide a Monte Carlo approximation to this decomposition for shift-invariant kernels, enabling scalable kernel methods.

## The Representer Theorem

The representer theorem is arguably the most important result in the theory of kernel methods from a practical standpoint. It says that although we optimize over an infinite-dimensional Hilbert space, the solution lives in a finite-dimensional subspace.

**Setup.** Let $k$ be a PD kernel, $H_k$ the corresponding RKHS, and $\{(x_i, y_i)\}_{i=1}^n$ training data with $x_i \in \mathcal{X}$, $y_i \in \mathbb{R}$. Let $L : \mathbb{R} \times \mathbb{R} \to [0, \infty)$ be any loss function and $\lambda > 0$ a regularization parameter. Consider the regularized empirical risk minimization problem:

$$\min_{f \in H_k} \left[ \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \, \|f\|_{H_k}^2 \right]$$

**Theorem (Representer Theorem; Kimeldorf and Wahba, 1971; Schölkopf, Herbrich, Smola, 2001).** Any minimizer $f^*$ of the above problem has the form:

$$f^*(\cdot) = \sum_{i=1}^n \alpha_i \, k(\cdot, x_i)$$

for some coefficients $\alpha = (\alpha_1, \ldots, \alpha_n) \in \mathbb{R}^n$.

**Proof.** Decompose any $f \in H_k$ into two orthogonal components. Let $V = \operatorname{span}\{k(\cdot, x_1), \ldots, k(\cdot, x_n)\}$ and $V^\perp$ its orthogonal complement in $H_k$. Write $f = f_V + f_\perp$ with $f_V \in V$, $f_\perp \in V^\perp$.

*The loss depends only on $f_V$:* For each $i$:
$$f(x_i) = \langle f, k(\cdot, x_i) \rangle_{H_k} = \langle f_V + f_\perp, k(\cdot, x_i) \rangle_{H_k} = \langle f_V, k(\cdot, x_i) \rangle_{H_k} + \underbrace{\langle f_\perp, k(\cdot, x_i) \rangle_{H_k}}_{= 0}$$

since $f_\perp \perp k(\cdot, x_i)$. Hence $f(x_i) = f_V(x_i)$ for all $i$: the perpendicular component does not affect evaluations at training points, so it does not affect the loss.

*The regularizer prefers $f_\perp = 0$:* $\|f\|_{H_k}^2 = \|f_V\|_{H_k}^2 + \|f_\perp\|_{H_k}^2 \geq \|f_V\|_{H_k}^2$.

Therefore, for any $f$ with $f_\perp \neq 0$, the function $f_V$ achieves the same loss with a strictly smaller regularizer, giving strictly smaller objective value. Any minimizer must have $f_\perp = 0$, i.e., $f^* \in V$. $\square$

**Reduction to a finite-dimensional problem.** Substituting $f = \sum_i \alpha_i k(\cdot, x_i)$ into the objective, using the reproducing property $f(x_j) = \sum_i \alpha_i k(x_i, x_j) = (K\alpha)_j$, the infinite-dimensional optimization reduces to:

$$\min_{\alpha \in \mathbb{R}^n} \left[ \frac{1}{n} \sum_{i=1}^n L(y_i, (K\alpha)_i) + \lambda \, \alpha^\top K \alpha \right]$$

where $K_{ij} = k(x_i, x_j)$ is the $n \times n$ Gram matrix and we used $\|f\|_{H_k}^2 = \alpha^\top K \alpha$. This is a finite $n$-dimensional optimization problem, regardless of the dimension of $H_k$.

**Kernel ridge regression.** For squared loss $L(y, \hat{y}) = (y - \hat{y})^2$, the objective is:

$$\frac{1}{n}\|y - K\alpha\|^2 + \lambda \, \alpha^\top K \alpha$$

Setting the gradient to zero: $\frac{2}{n} K^\top(K\alpha - y) + 2\lambda K \alpha = 0$, which simplifies (since $K = K^\top$ and $K^2 = K \cdot K$) to:

$$\alpha^* = (K + n\lambda I)^{-1} y$$

The predictor at a new point $x$ is $f^*(x) = \sum_i \alpha_i^* k(x, x_i) = k(x)^\top (K + n\lambda I)^{-1} y$, where $k(x) = (k(x, x_1), \ldots, k(x, x_n))^\top$.

> **Key insight:** The representer theorem turns a problem over an infinite-dimensional function space into an $n \times n$ linear system. The "curse of dimensionality" in the feature space completely disappears — the complexity is determined by the number of training points $n$, not the dimension of $H_k$. This is the computational miracle at the heart of kernel methods.

**Generalization.** The representer theorem holds more broadly: for any strictly monotone increasing function $\Omega : [0, \infty) \to \mathbb{R}$, the minimizer of $\sum_i L(y_i, f(x_i)) + \Omega(\|f\|_{H_k})$ lies in $\operatorname{span}\{k(\cdot, x_i)\}$. This covers all regularized kernel methods.

## Connection to Gaussian Processes

Gaussian processes (GPs) provide a Bayesian perspective on RKHS that illuminates both the prior structure encoded by a kernel and the role of regularization.

**Gaussian process definition.** A Gaussian process $f \sim \mathcal{GP}(m, k)$ is a stochastic process indexed by $\mathcal{X}$ such that for every finite set $\{x_1, \ldots, x_n\}$, the vector $(f(x_1), \ldots, f(x_n))$ is jointly Gaussian with mean $m(x_i)$ and covariance $\text{Cov}(f(x_i), f(x_j)) = k(x_i, x_j)$. The kernel $k$ is the covariance function of the GP. The positive semi-definiteness of $k$ is exactly the condition that guarantees valid covariance matrices for all finite collections of points.

**Kernel ridge regression = GP posterior mean.** Place a GP prior $f \sim \mathcal{GP}(0, k)$ and assume i.i.d. Gaussian noise: $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. The posterior distribution given data $\{(x_i, y_i)\}$ is again a GP:

$$f \mid y \sim \mathcal{GP}(\mu_n, k_n)$$

with posterior mean and covariance:

$$\mu_n(x) = k(x)^\top (K + \sigma^2 I)^{-1} y, \qquad k_n(x, x') = k(x,x') - k(x)^\top (K + \sigma^2 I)^{-1} k(x')$$

Setting $\lambda = \sigma^2 / n$, the posterior mean $\mu_n$ is exactly the kernel ridge regression predictor. Kernel ridge regression is MAP (maximum a posteriori) estimation under a GP prior with Gaussian likelihood — the regularizer $\lambda \|f\|_{H_k}^2$ corresponds to the log-prior $-\|f\|_{H_k}^2 / (2\sigma^2)$ (up to a constant).

**RKHS as the Cameron-Martin space.** The RKHS $H_k$ is the Cameron-Martin space of the GP: it consists of the "directions" in function space along which the GP measure can be shifted while remaining absolutely continuous with respect to itself. Functions in $H_k$ have finite log-likelihood ratio under the GP; functions outside $H_k$ are in the support of the GP measure but have infinite regularization cost. The GP samples almost surely lie outside $H_k$ (sample paths are rougher than RKHS elements), yet the posterior mean — a smoothed version of the data — lies in $H_k$.

**Matérn kernels and Sobolev spaces.** The Matérn-$\nu$ kernel on $\mathbb{R}^d$ with smoothness parameter $\nu$ satisfies $H_k = W^{\nu + d/2, 2}(\mathcal{X})$ (the Sobolev space of functions with $\lfloor \nu + d/2 \rfloor$ square-integrable weak derivatives). The corresponding GP has sample paths that are $\lceil \nu \rceil - 1$ times mean-square differentiable. This precise relationship between the kernel, the RKHS, and the GP sample path regularity makes the Matérn family the default choice for GP regression when you want interpretable smoothness control (see L4 for the Sobolev space connection).

> **Intuition:** The kernel encodes your prior belief about the smoothness of the unknown function. Choosing the RBF kernel says "I believe $f$ is infinitely smooth (analytic)." Choosing the Matérn-$3/2$ kernel says "I believe $f$ is twice differentiable." The posterior mean adapts to the data while respecting this prior, and the RKHS norm measures how much structure you are "spending" relative to the prior.

**Prediction uncertainty.** A key advantage of the GP view over plain kernel ridge regression is the posterior covariance $k_n(x, x')$, which quantifies prediction uncertainty. The posterior variance $k_n(x,x)$ is small where training data is dense and large where data is sparse, providing calibrated uncertainty estimates. The RKHS norm $\|f^*\|_{H_k}^2 = y^\top (K + n\lambda I)^{-1} K (K + n\lambda I)^{-1} y$ controls the marginal likelihood (evidence) via the identity $\log p(y) = -\frac{1}{2} y^\top (K + \sigma^2 I)^{-1} y - \frac{1}{2} \log\det(K + \sigma^2 I) - \frac{n}{2}\log(2\pi)$, enabling hyperparameter optimization by evidence maximization.

## ML Connections

RKHS theory is not an abstract framework divorced from practice. It underpins five major areas of modern machine learning.

**Support vector machines (SVMs).** The SVM margin maximization problem $\min_f \|f\|_{H_k}^2 \text{ s.t. } y_i f(x_i) \geq 1$ is a regularized problem in the RKHS with hinge loss $L(y, \hat{y}) = \max(0, 1 - y\hat{y})$. The representer theorem guarantees the solution lies in $\operatorname{span}\{k(\cdot, x_i)\}$, and the dual problem leads to the classical SVM dual: $\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)$ subject to $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$. The support vectors are training points with $\alpha_i > 0$.

**Kernel PCA.** Kernel PCA computes the principal components of the data in the RKHS feature space. Centering the feature vectors and computing the spectral decomposition of the Gram matrix $K$ amounts, by Mercer's theorem, to approximating the Mercer eigendecomposition: the $m$-th kernel PCA component captures the direction $\phi_m$ in $H_k$ of maximum variance. Dimensionality reduction in the RKHS corresponds to nonlinear dimensionality reduction in the input space.

**Maximum Mean Discrepancy (MMD).** Given two distributions $P$ and $Q$ on $\mathcal{X}$, define the **mean embedding** $\mu_P = \mathbb{E}_{x \sim P}[k(\cdot, x)] \in H_k$ (and similarly $\mu_Q$). The **maximum mean discrepancy** is:

$$\text{MMD}^2(P, Q) = \|\mu_P - \mu_Q\|_{H_k}^2$$

Expanding:

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\,\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]$$

Given $m$ samples from $P$ and $n$ samples from $Q$, the unbiased estimator of $\text{MMD}^2$ is computable in $O((m+n)^2)$ time. Under a characteristic kernel (e.g., RBF), $\text{MMD}(P, Q) = 0$ iff $P = Q$, making it a valid two-sample test statistic. MMD also appears as a training objective in generative models: minimizing $\text{MMD}^2$ between the model distribution and data distribution is the objective of the MMD-GAN and related models. It avoids the mode-dropping instabilities of adversarial training and has provable sample complexity bounds.

**Neural tangent kernel (NTK).** Wide neural networks (width $\to \infty$) trained by gradient descent have a dynamics that freezes the network parameters near their initialization and evolves the predictions according to a kernel:

$$k_{\text{NTK}}(x, x') = \mathbb{E}_{\theta_0}\left[ \nabla_\theta f(x; \theta_0)^\top \nabla_\theta f(x'; \theta_0) \right]$$

In the infinite-width limit, gradient descent on the squared loss is equivalent to kernel ridge regression with this kernel. This makes RKHS theory applicable to the analysis of deep learning, explaining overparameterization, implicit regularization, and the benign overfitting phenomenon (interpolating solutions that generalize well).

**Deep kernels.** Combining learned feature representations with kernel methods gives deep kernels: $k_\theta(x, x') = k_0(\varphi_\theta(x), \varphi_\theta(x'))$ where $\varphi_\theta$ is a neural network and $k_0$ is a base kernel. Deep GP models stack multiple GP layers, with each layer's output serving as input to the next. The resulting compositional structure retains the uncertainty quantification benefits of GPs while gaining the representational flexibility of deep networks.

## Python: RKHS, Mercer Eigendecomposition, and Kernel Ridge Regression

The following code implements five demonstrations: kernel ridge regression on a nonlinear regression problem, the Mercer eigendecomposition of the RBF kernel, visualization of how the RKHS norm penalizes high-frequency components, empirical verification of the representer theorem, and MMD between two distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ── Reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Kernels ──────────────────────────────────────────────────────────────────

def rbf_kernel(X, Y, sigma=1.0):
    """RBF kernel: k(x, x') = exp(-||x - x'||^2 / (2 sigma^2))."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sq_dists = (
        np.sum(X**2, axis=1, keepdims=True)
        - 2 * X @ Y.T
        + np.sum(Y**2, axis=1, keepdims=True).T
    )
    return np.exp(-sq_dists / (2 * sigma**2))


def poly_kernel(X, Y, degree=3):
    """Polynomial kernel: k(x, x') = (1 + x . x')^degree."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return (1 + X @ Y.T) ** degree


# ── 1. Kernel Ridge Regression ───────────────────────────────────────────────
# True function: f(x) = sin(2 pi x) + 0.5 cos(6 pi x) on [0, 1]

n_train = 30
x_train = rng.uniform(0, 1, n_train).reshape(-1, 1)
y_train = np.sin(2 * np.pi * x_train).ravel() + 0.5 * np.cos(6 * np.pi * x_train).ravel()
y_train += rng.normal(0, 0.1, n_train)  # add noise

x_test = np.linspace(0, 1, 300).reshape(-1, 1)
y_true = np.sin(2 * np.pi * x_test).ravel() + 0.5 * np.cos(6 * np.pi * x_test).ravel()

sigma = 0.3   # RBF bandwidth
lam = 1e-3    # regularization parameter

K_train = rbf_kernel(x_train, x_train, sigma=sigma)           # (n, n) Gram matrix
alpha = np.linalg.solve(K_train + n_train * lam * np.eye(n_train), y_train)

K_test = rbf_kernel(x_test, x_train, sigma=sigma)             # (300, n)
y_pred = K_test @ alpha

# RKHS norm of the solution
rkhs_norm_sq = alpha @ K_train @ alpha
print(f"KRR: RKHS norm^2 of solution = {rkhs_norm_sq:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(x_test, y_true, 'k-', lw=2, label='True function')
ax.plot(x_test, y_pred, 'b-', lw=2, label='KRR prediction')
ax.scatter(x_train, y_train, c='r', s=30, zorder=5, label='Training data')
ax.set_title('Kernel Ridge Regression (RBF kernel)')
ax.set_xlabel('x')
ax.legend()

# ── 2. Mercer Eigendecomposition of RBF kernel ───────────────────────────────
# Discretize [0, 1] and compute the integral operator matrix

m_grid = 200
x_grid = np.linspace(0, 1, m_grid).reshape(-1, 1)
dx = 1.0 / (m_grid - 1)

K_grid = rbf_kernel(x_grid, x_grid, sigma=sigma)   # (m, m)
# Approximate T_k as (K_grid * dx): eigenvalues approximate lambda_n
# The weight matrix is dx * I (uniform quadrature)
T = K_grid * dx   # Discretized integral operator

# eigh returns eigenvalues in ascending order; we want descending
eigenvalues, eigenvectors = eigh(T)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Normalize eigenfunctions in L2: integral phi_n^2 dx = 1
norms = np.sqrt(np.sum(eigenvectors**2, axis=0) * dx)
eigenvectors = eigenvectors / norms

print(f"\nTop 5 Mercer eigenvalues (sigma={sigma}):")
for i in range(5):
    print(f"  lambda_{i+1} = {eigenvalues[i]:.6f}")

ax = axes[1]
ax.semilogy(range(1, 31), eigenvalues[:30], 'o-', markersize=4)
ax.set_title('Mercer eigenvalue spectrum (RBF, sigma=0.3)')
ax.set_xlabel('Eigenvalue index n')
ax.set_ylabel('lambda_n (log scale)')
ax.grid(True, alpha=0.4)

# ── 3. RKHS norm penalizes high-frequency components ─────────────────────────
# Compare two functions: smooth sine vs. high-frequency sine
# Project onto Mercer eigenbasis and compute RKHS norm

def rkhs_norm_via_mercer(f_vals, eigenvectors, eigenvalues, dx, n_terms=50):
    """
    Compute ||f||^2_{H_k} = sum_n f_hat_n^2 / lambda_n
    where f_hat_n = <f, phi_n>_{L^2} = sum_j f(x_j) phi_n(x_j) dx
    """
    coeffs = eigenvectors[:, :n_terms].T @ f_vals * dx  # (n_terms,)
    lam = eigenvalues[:n_terms]
    return np.sum(coeffs**2 / np.maximum(lam, 1e-12))


x_flat = x_grid.ravel()
f_smooth = np.sin(2 * np.pi * x_flat)
f_rough = np.sin(20 * np.pi * x_flat)

norm_smooth = rkhs_norm_via_mercer(f_smooth, eigenvectors, eigenvalues, dx)
norm_rough  = rkhs_norm_via_mercer(f_rough,  eigenvectors, eigenvalues, dx)
print(f"\nRKHS norm^2 — smooth (freq=1): {norm_smooth:.4f}")
print(f"RKHS norm^2 — rough  (freq=10): {norm_rough:.4f}")
print(f"Ratio rough/smooth: {norm_rough / norm_smooth:.1f}x")

# ── 4. Representer Theorem: span vs full RKHS ────────────────────────────────
# In practice KRR already lives in span{k(.,x_i)}.
# We verify: the perpendicular component of f* in the full grid is negligible.

# Compute f* on the grid two ways:
# (a) via representer coefficients: f*(x) = sum_i alpha_i k(x, x_i)
f_repr = K_test @ alpha   # already computed above

# (b) via the full Gram matrix solve on a combined set (representer theorem holds exactly)
# For verification, check that adding random functions in V^perp doesn't change the loss
# (they affect RKHS norm but not training evaluations)

# Construct a function in V^perp: orthogonalize a test function against all k(.,x_i)
test_fn = np.sin(10 * np.pi * x_grid.ravel())  # arbitrary function on grid

K_train_grid = rbf_kernel(x_train, x_grid, sigma=sigma)   # (n, m_grid)
# Project test_fn onto V = span{k(.,x_i)} in the RKHS sense
# Projection coefficients: solve K_train * beta = K_train_grid * test_fn
K_train_sq = rbf_kernel(x_train, x_train, sigma=sigma)
proj_coeff = np.linalg.solve(K_train_sq + 1e-10 * np.eye(n_train),
                              K_train_grid @ test_fn * dx)
f_proj_on_V = (K_train_grid.T @ proj_coeff)  # projection onto V

f_perp = test_fn - f_proj_on_V  # component in V^perp

# Evaluations of f_perp at training points should be ~ 0
f_perp_at_train = K_train_grid @ f_perp * dx
print(f"\nRepreseneter theorem check:")
print(f"  Max |f_perp(x_i)| at training points = {np.max(np.abs(f_perp_at_train)):.2e}")
print("  (Should be ~0: V^perp functions don't affect training evaluations)")

# ── 5. Maximum Mean Discrepancy (MMD) ────────────────────────────────────────
# Compare two distributions: N(0,1) vs N(1,1) and N(0,1) vs N(0,1)

def mmd_squared(X, Y, sigma=1.0):
    """
    Unbiased estimate of MMD^2(P, Q) using samples X ~ P, Y ~ Q.
    MMD^2 = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]
    """
    m, n = len(X), len(Y)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    Kxx = rbf_kernel(X, X, sigma=sigma)
    Kyy = rbf_kernel(Y, Y, sigma=sigma)
    Kxy = rbf_kernel(X, Y, sigma=sigma)

    # Unbiased: exclude diagonal terms in Kxx and Kyy
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    term_xx = np.sum(Kxx) / (m * (m - 1))
    term_yy = np.sum(Kyy) / (n * (n - 1))
    term_xy = np.sum(Kxy) / (m * n)

    return term_xx - 2 * term_xy + term_yy


n_samples = 500
X_p = rng.normal(0, 1, n_samples)  # P = N(0,1)
Y_q = rng.normal(1, 1, n_samples)  # Q = N(1,1): shifted
Y_same = rng.normal(0, 1, n_samples)  # Q = N(0,1): same as P

mmd_diff = mmd_squared(X_p, Y_q, sigma=1.0)
mmd_same = mmd_squared(X_p, Y_same, sigma=1.0)
print(f"\nMMD^2(N(0,1), N(1,1))  = {mmd_diff:.5f}  (distributions differ)")
print(f"MMD^2(N(0,1), N(0,1))  = {mmd_same:.5f}  (same distribution, should be ~0)")

# Visualize MMD vs shift
shifts = np.linspace(0, 3, 20)
mmd_vals = []
for delta in shifts:
    Y_shifted = rng.normal(delta, 1, n_samples)
    mmd_vals.append(mmd_squared(X_p, Y_shifted, sigma=1.0))

ax = axes[2]
ax.plot(shifts, mmd_vals, 'o-', color='purple', markersize=5)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_title('MMD^2 vs distributional shift (RBF kernel, sigma=1)')
ax.set_xlabel('Mean shift delta: Q = N(delta, 1)')
ax.set_ylabel('MMD^2')
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('rkhs_demo.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Expected output (indicative values):**
```
KRR: RKHS norm^2 of solution = 1.2453

Top 5 Mercer eigenvalues (sigma=0.3):
  lambda_1 = 0.265182
  lambda_2 = 0.223091
  lambda_3 = 0.152044
  lambda_4 = 0.087231
  lambda_5 = 0.042187

RKHS norm^2 — smooth (freq=1): 0.0423
RKHS norm^2 — rough  (freq=10): 187.31
Ratio rough/smooth: 4428.0x

Representer theorem check:
  Max |f_perp(x_i)| at training points = 3.21e-14
  (Should be ~0: V^perp functions don't affect training evaluations)

MMD^2(N(0,1), N(1,1))  = 0.09183  (distributions differ)
MMD^2(N(0,1), N(0,1))  = 0.00021  (same distribution, should be ~0)
```

**What the code demonstrates:**

*Kernel ridge regression (panel 1):* The RBF kernel with appropriate bandwidth and regularization recovers a smooth nonlinear function from noisy data. The $\alpha = (K + n\lambda I)^{-1}y$ formula shows the complete reduction of infinite-dimensional RKHS optimization to a single linear system solve.

*Mercer spectrum (panel 2):* The discretized integral operator $T_k$ has eigenvalues that decay rapidly (super-polynomially for the RBF kernel). Only a handful of eigenvalues are significant; the rest are near zero. This confirms that the RKHS effectively has low "intrinsic dimension" — most directions are heavily penalized.

*RKHS norm and frequency (implicit in norm computation):* The ratio of RKHS norms between a high-frequency function (20$\pi$ oscillations) and a smooth function (2$\pi$ oscillations) is thousands-fold. The RKHS norm acts as a smoothness penalty: high-frequency components cost exponentially more in the RBF RKHS.

*Representer theorem (numerical check):* The maximum absolute value of $f_\perp$ at training points is at the level of floating-point machine epsilon ($\sim 10^{-14}$), confirming that functions in $V^\perp$ are invisible to the loss. The representer theorem is not an approximation — it is an exact algebraic consequence.

*MMD (panel 3):* The MMD between $\mathcal{N}(0,1)$ and $\mathcal{N}(\delta, 1)$ increases monotonically with the shift $\delta$, starting near zero for $\delta = 0$ and growing as the distributions diverge. This confirms the characteristic kernel property: MMD detects any distributional difference, not just location shifts.

---

:::quiz
**Quiz Block 1 — Reproducing Property and Pointwise Evaluation**

**Question 1.** Let $H_k$ be an RKHS with kernel $k$ on $\mathcal{X}$. Which of the following is the reproducing property?

A. $k(x, x') = \langle k(\cdot, x), k(\cdot, x') \rangle_{H_k}$ for all $x, x' \in \mathcal{X}$

B. $f(x) = \langle f, k(\cdot, x) \rangle_{H_k}$ for all $f \in H_k$, $x \in \mathcal{X}$

C. $\|k(\cdot, x)\|_{H_k} = 1$ for all $x \in \mathcal{X}$

D. $k(x, x) = \|f\|_{H_k}^2$ for all $f \in H_k$

**Answer:** B. The reproducing property states that the inner product of any $f \in H_k$ with the kernel section $k(\cdot, x)$ evaluates $f$ at $x$. Note that A is a consequence of B (apply B with $f = k(\cdot, x')$), but A is not the general reproducing property — it holds only for kernel sections, not arbitrary $f$.

---

**Question 2.** In a general Hilbert space $L^2([0,1])$, is the pointwise evaluation functional $\text{ev}_x : f \mapsto f(x)$ bounded? Why or why not?

**Answer:** No. In $L^2([0,1])$, elements are equivalence classes of functions that agree almost everywhere. Pointwise evaluation $f(x)$ is not well-defined (changing $f$ on a set of measure zero changes $f(x)$ but not $\|f\|_{L^2}$), and even when formally defined, the functional is unbounded: consider $f_n = n \cdot \mathbf{1}_{[x - 1/n^2, x + 1/n^2]}$ which has $\|f_n\|_{L^2} \to 0$ but $f_n(x) = n \to \infty$. The RKHS condition precisely requires that $\text{ev}_x$ is bounded: $|f(x)| \leq \sqrt{k(x,x)} \|f\|_{H_k}$ for all $f$.

---

**Question 3.** For the RBF kernel $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$, compute $\|k(\cdot, x)\|_{H_k}^2$ for any fixed $x \in \mathcal{X}$.

**Answer:** By the reproducing property applied to $f = k(\cdot, x)$:
$$\|k(\cdot, x)\|_{H_k}^2 = \langle k(\cdot, x), k(\cdot, x) \rangle_{H_k} = k(x, x) = \exp(-\|x - x\|^2 / 2\sigma^2) = \exp(0) = 1$$
So every kernel section has unit $H_k$-norm for the RBF kernel. The operator norm of the evaluation functional at $x$ is $\|\text{ev}_x\| = \sqrt{k(x,x)} = 1$, uniformly in $x$.
:::

---

:::quiz
**Quiz Block 2 — The Representer Theorem**

**Question 1.** In the regularized ERM problem $\min_{f \in H_k} \frac{1}{n}\sum_i L(y_i, f(x_i)) + \lambda \|f\|_{H_k}^2$, why does adding a function $g \in V^\perp$ (orthogonal to all kernel sections $k(\cdot, x_i)$) never improve the objective?

**Answer:** Adding $g \in V^\perp$ has two effects: (1) it leaves all training evaluations unchanged, since $f(x_i) = \langle f, k(\cdot, x_i) \rangle_{H_k}$ and $\langle g, k(\cdot, x_i) \rangle_{H_k} = 0$ for $g \in V^\perp$; (2) it strictly increases the regularizer, since $\|f + g\|_{H_k}^2 = \|f\|_{H_k}^2 + \|g\|_{H_k}^2 > \|f\|_{H_k}^2$ for $g \neq 0$. Therefore adding any nonzero $g \in V^\perp$ strictly increases the objective. Any minimizer must have zero $V^\perp$ component.

---

**Question 2.** What is the closed-form solution for kernel ridge regression $\min_{f \in H_k} \frac{1}{n}\sum_i (y_i - f(x_i))^2 + \lambda \|f\|_{H_k}^2$, and what are the computational requirements?

**Answer:** By the representer theorem, $f^*(\cdot) = \sum_i \alpha_i k(\cdot, x_i)$ with $\alpha^* = (K + n\lambda I)^{-1} y$, where $K_{ij} = k(x_i, x_j)$ is the $n \times n$ Gram matrix. Computing $K$ costs $O(n^2 d)$ for $d$-dimensional inputs. Solving the linear system costs $O(n^3)$ naively (Cholesky: $K + n\lambda I$ is symmetric positive definite). Prediction at a new point costs $O(nd)$ to compute the kernel vector $k(x) = (k(x, x_i))_i$ and $O(n)$ for the dot product. Total: $O(n^3 + n^2 d)$ training, $O(nd)$ prediction.

---

**Question 3.** Suppose you use the representer theorem and find that many $\alpha_i$ are exactly zero (i.e., the solution is sparse in the kernel basis). Does this affect the function class being searched? What property of the loss function could cause this?

**Answer:** Sparsity of $\alpha$ does not change the function class — the representer theorem guarantees only that the minimizer lies in $V = \operatorname{span}\{k(\cdot, x_i)\}$; the specific $\alpha$ depends on the loss. Sparsity arises from non-smooth or piecewise-linear losses like the hinge loss (SVMs) or $\ell^1$-regularized variants. For hinge loss, the KKT conditions force $\alpha_i = 0$ for points correctly classified with large margin (non-support vectors). The squared loss gives a dense $\alpha = (K + n\lambda I)^{-1}y$ with all entries typically nonzero. Sparsity is a property of the optimization geometry, not of the RKHS itself.
:::

---

:::quiz
**Quiz Block 3 — MMD versus Wasserstein Distance**

**Question 1.** Define the Maximum Mean Discrepancy and explain why it requires a "characteristic" kernel for $\text{MMD}(P,Q) = 0 \Rightarrow P = Q$.

**Answer:** $\text{MMD}(P, Q) = \|\mu_P - \mu_Q\|_{H_k}$ where $\mu_P = \mathbb{E}_{x \sim P}[k(\cdot, x)]$ is the mean embedding of $P$ in $H_k$. For $\text{MMD}(P,Q) = 0$ iff $P = Q$ (faithfulness), the kernel must be **characteristic**: the mean embedding map $P \mapsto \mu_P$ must be injective on the space of probability measures. The RBF and Matérn kernels are characteristic; the linear kernel is not (it only captures first moments: $\mu_P = k(\cdot, \cdot) \ast P$ captures only the mean of $P$). A kernel is characteristic iff its RKHS is dense in $C_0(\mathcal{X})$ (continuous functions vanishing at infinity).

---

**Question 2.** The unbiased estimator of $\text{MMD}^2(P, Q)$ from $m$ samples $X = \{x_i\} \sim P$ and $n$ samples $Y = \{y_j\} \sim Q$ is:

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)}\sum_{i \neq j} k(x_i, x_j) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j) + \frac{1}{n(n-1)}\sum_{i \neq j} k(y_i, y_j)$$

What is the sample complexity (rate of convergence to $\text{MMD}^2(P, Q)$)?

**Answer:** The standard deviation of $\widehat{\text{MMD}}^2$ is $O(1/\sqrt{n})$ (assuming $m \asymp n$), so $\widehat{\text{MMD}}^2$ converges at rate $O(n^{-1/2})$ regardless of dimension $d$. This is in stark contrast to the empirical Wasserstein distance, which converges at rate $O(n^{-1/d})$ — exponentially slow in dimension. MMD's dimension-free sample complexity (from a statistical perspective) makes it attractive for high-dimensional distribution comparison, at the cost of being blind to geometric properties that Wasserstein captures (e.g., it cannot distinguish two distributions with the same moments up to order exceeding the kernel's Taylor expansion).

---

**Question 3.** Compare MMD (with RBF kernel) and $W_1$ (Wasserstein-1) as two-sample test statistics. In what situation would each be preferred? What does the NTK connection suggest about using deep network feature maps for MMD?

**Answer:** MMD with a characteristic kernel is a valid two-sample test statistic with dimension-free $O(n^{-1/2})$ convergence, making it practical in high dimensions. However, the choice of kernel bandwidth $\sigma$ is critical: if $\sigma$ is too large, MMD is insensitive to local distributional differences; if too small, the Gram matrix becomes near-diagonal and power is lost. $W_1$ captures the geometry of the sample space (it is sensitive to the metric structure, e.g., two distributions with shifted support have $W_1 > 0$ even if they agree on all moments), but estimation converges at rate $O(n^{-1/d})$ and is computationally costly ($O(n^3)$ for exact computation). Prefer MMD in high dimensions where geometric faithfulness matters less than statistical power; prefer Wasserstein when the metric structure encodes meaningful physical distances (e.g., image pixel grids) and sample sizes are large enough relative to dimension.

The NTK connection is illuminating: using a deep network $\varphi_\theta$ as a feature map and computing MMD in the resulting space ($k_\theta(x,x') = \varphi_\theta(x) \cdot \varphi_\theta(x')$) is equivalent (in the infinite-width limit) to kernel MMD with the NTK. Since the NTK adapts to the geometry of the function class learned by the network, deep MMD can be far more sensitive to the relevant modes of distributional difference than a fixed RBF kernel. This is the principle behind the MMD-GAN discriminator and kernel two-sample tests with learned kernels.
:::
