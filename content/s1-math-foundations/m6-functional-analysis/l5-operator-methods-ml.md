---
title: "Operator Methods in Modern Machine Learning"
estimatedMinutes: 40
tags: ["NTK", "mean-embeddings", "MMD", "covariance-operators", "Koopman"]
prerequisites: ["l1-banach-hilbert-spaces", "l2-linear-operators", "l3-rkhs", "l4-sobolev-spaces"]
---

## Overview

This lesson is the synthesis of Semester 1. You have built the machinery — Hilbert spaces, bounded and compact operators, reproducing kernels, Sobolev spaces — and now that machinery turns out to govern the most important theoretical objects in modern machine learning. The Neural Tangent Kernel explains why massively overparameterized networks generalize. Mean embeddings and MMD underlie every modern test for whether two distributions are the same. Covariance operators in RKHS capture statistical dependence without moment assumptions. The Stein operator lets you measure how well a model fits data without ever sampling from it. Koopman theory turns nonlinear dynamical systems into linear ones. All of these are operator-theoretic ideas, and all of them connect back to M1–M5.

This is not a survey. Each section gives the precise functional-analytic construction, derives the key identity, and then shows exactly how it appears in a current ML context.

## The Neural Tangent Kernel

**Setup.** Let $f_\theta : \mathbb{R}^d \to \mathbb{R}$ be a neural network with parameters $\theta \in \mathbb{R}^P$. During gradient flow training with squared loss on a dataset $(X, y) \in \mathbb{R}^{n \times d} \times \mathbb{R}^n$, the parameters evolve as:

$$\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta), \qquad \mathcal{L}(\theta) = \frac{1}{2}\|f_\theta(X) - y\|^2$$

The network outputs on the training set evolve as:

$$\frac{df_\theta(X)}{dt} = \frac{\partial f_\theta(X)}{\partial \theta} \frac{d\theta}{dt} = -J_\theta(X) J_\theta(X)^\top (f_\theta(X) - y)$$

where $J_\theta(X) \in \mathbb{R}^{n \times P}$ is the Jacobian of the outputs with respect to the parameters. The matrix $\Theta_\theta(X, X) = J_\theta(X) J_\theta(X)^\top \in \mathbb{R}^{n \times n}$ is the **Neural Tangent Kernel (NTK)** matrix, with entries:

$$\Theta_\theta(x, x') = \langle \nabla_\theta f_\theta(x),\, \nabla_\theta f_\theta(x') \rangle_{\mathbb{R}^P} = \sum_{p=1}^{P} \frac{\partial f_\theta(x)}{\partial \theta_p} \frac{\partial f_\theta(x')}{\partial \theta_p}$$

**The infinite-width limit.** Jacot, Gabriel, and Hongler (2018) proved that for a fully-connected network with $L$ layers, each of width $n_\ell$, initialized with weights $W^{(\ell)}_{ij} \sim \mathcal{N}(0, \sigma^2_w / n_\ell)$:

$$\Theta_\theta(x, x') \xrightarrow{n_1, \ldots, n_{L-1} \to \infty} \Theta^\infty(x, x')$$

where $\Theta^\infty$ is a **deterministic, fixed kernel** that depends only on the activation function, depth, and initialization variance. This kernel does not change during training — it is constant at infinite width. The limit is constructed recursively. Define the arc-cosine kernel values:

$$\Sigma^{(0)}(x, x') = \frac{x^\top x'}{d}, \qquad \Sigma^{(\ell)}(x, x') = \sigma_w^2 \, \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(\ell)})}[\phi(u)\phi(v)]$$

where $\Lambda^{(\ell)} = \begin{pmatrix} \Sigma^{(\ell-1)}(x,x) & \Sigma^{(\ell-1)}(x,x') \\ \Sigma^{(\ell-1)}(x',x) & \Sigma^{(\ell-1)}(x',x') \end{pmatrix}$ and $\phi$ is the activation function. The limiting NTK is:

$$\Theta^\infty(x, x') = \sum_{\ell=1}^{L} \left( \Sigma^{(\ell)}(x, x') \prod_{j=\ell+1}^{L} \dot{\Sigma}^{(j)}(x, x') \right)$$

where $\dot{\Sigma}^{(\ell)}(x, x') = \sigma_w^2 \, \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(\ell-1)})}[\phi'(u)\phi'(v)]$ is the derivative kernel.

**The NTK as a Hilbert-Schmidt operator.** On a compact domain $\mathcal{X}$, $\Theta^\infty$ defines an integral operator $T_\Theta : L^2(\mathcal{X}) \to L^2(\mathcal{X})$:

$$[T_\Theta f](x) = \int_\mathcal{X} \Theta^\infty(x, x') f(x') \, d\mu(x')$$

Since $\Theta^\infty$ is continuous and $\mathcal{X}$ is compact, $T_\Theta$ is a Hilbert-Schmidt operator (from L2 in Section 2). Its eigendecomposition $T_\Theta \phi_i = \lambda_i \phi_i$ determines the RKHS $\mathcal{H}_\Theta$ of the NTK:

$$\mathcal{H}_\Theta = \left\{ f = \sum_i a_i \phi_i : \sum_i \frac{a_i^2}{\lambda_i} < \infty \right\}, \qquad \|f\|^2_{\mathcal{H}_\Theta} = \sum_i \frac{a_i^2}{\lambda_i}$$

This is precisely the Mercer RKHS from L3.

**Training dynamics as a linear ODE.** With $\Theta_\theta \equiv \Theta^\infty$ fixed, the output trajectory is:

$$\frac{d\hat{f}}{dt} = -\Theta^\infty(X, X)(\hat{f} - y)$$

This is a linear ODE! Let $\Theta^\infty(X,X) = V \Lambda V^\top$ be the eigendecomposition (symmetric positive semidefinite). Projecting onto eigenvectors $v_i$:

$$\frac{d}{dt}(v_i^\top \hat{f}) = -\lambda_i (v_i^\top \hat{f} - v_i^\top y)$$

Each mode decays as $e^{-\lambda_i t}$. The solution for the training residual is:

$$\hat{f}(t) - y = (I - e^{-\Theta^\infty t})(f_0 - y) - (I - e^{-\Theta^\infty t})y + (f_0 - y)$$

More precisely, if we set $f_0 = 0$ at initialization:

$$\hat{f}(t) = (I - e^{-\Theta^\infty(X,X) t}) y$$

As $t \to \infty$, $\hat{f}(t) \to y$ on the training set. The **eigenvalue ordering** is critical: the mode corresponding to eigenvalue $\lambda_i$ is learned at rate $\lambda_i$. High-eigenvalue modes (smooth, low-frequency functions in the NTK's spectral ordering) are learned first; high-frequency modes come later. This is **spectral bias** or the **frequency principle**.

> **Key insight:** The NTK converts the nonlinear optimization of a neural network into a linear ODE in function space. At infinite width, the network performs kernel regression in $\mathcal{H}_\Theta$. The generalization properties of the infinite-width network are entirely determined by the spectral properties of $T_\Theta$. High eigenvalues correspond to functions the network learns to fit; low eigenvalues correspond to directions the network cannot efficiently represent, providing implicit regularization.

**Kernel regression connection.** The infinite-time limit of gradient flow is exactly the minimum-RKHS-norm interpolant (from L3):

$$f^* = \Theta^\infty(\cdot, X) [\Theta^\infty(X,X)]^{-1} y = \arg\min_{f \in \mathcal{H}_\Theta} \|f\|_{\mathcal{H}_\Theta}^2 \text{ subject to } f(x_i) = y_i$$

This is the **kernel regression solution** in $\mathcal{H}_\Theta$. The bias-variance tradeoff in kernel regression (from M2, now formalized via the NTK RKHS) controls generalization.

> **Refresher:** From L3, the reproducing property gives $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}_k}$. The NTK $\Theta^\infty$ is a positive-definite kernel, so it generates an RKHS in which the minimum-norm interpolant is uniquely defined via the representer theorem.

**Limitations of the NTK.** Finite-width networks undergo **feature learning**: the kernel $\Theta_\theta$ changes during training as representations in intermediate layers are updated. This is what makes modern deep learning powerful. The NTK regime (fixed kernel, no feature learning) holds exactly only at infinite width, and practically only for very wide networks early in training. The NTK is a rigorous mathematical baseline — it explains overparameterized generalization without feature learning — but the full theory requires understanding kernel change (mean-field theory, tensor programs).

## RKHS Mean Embeddings

**Embedding a distribution into a Hilbert space.** Given a probability measure $P$ on $\mathcal{X}$ and a kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ with $\mathcal{H}_k$ its RKHS, the **mean embedding** of $P$ is:

$$\mu_P = \int_\mathcal{X} k(\cdot, x) \, dP(x) \in \mathcal{H}_k$$

This is a Bochner integral (integral of a Hilbert-space-valued function). It is well-defined whenever $\mathbb{E}_{x \sim P}[\sqrt{k(x,x)}] < \infty$, which holds for bounded kernels like the RBF kernel.

The key property is that the inner product in $\mathcal{H}_k$ with the mean embedding computes expectations:

$$\langle f, \mu_P \rangle_{\mathcal{H}_k} = \int_\mathcal{X} f(x) \, dP(x) = \mathbb{E}_P[f]$$

for all $f \in \mathcal{H}_k$. This follows immediately from the reproducing property: $\langle f, k(\cdot, x) \rangle = f(x)$, so linearity of the inner product and the integral give $\langle f, \mu_P \rangle = \int \langle f, k(\cdot, x) \rangle dP(x) = \mathbb{E}_P[f]$.

**The mean embedding operator.** Define the operator $\mathcal{E}_P : \mathcal{H}_k \to \mathbb{R}$ by $\mathcal{E}_P(f) = \mathbb{E}_P[f]$. Then $\mu_P$ is the Riesz representative of $\mathcal{E}_P$ in $\mathcal{H}_k$ (from L1, the Riesz representation theorem). The embedding $P \mapsto \mu_P$ is a map from the space of probability measures to $\mathcal{H}_k$.

**Maximum Mean Discrepancy.** The distance between two distributions in the RKHS metric is:

$$\text{MMD}^2(P, Q) = \|\mu_P - \mu_Q\|^2_{\mathcal{H}_k}$$

Expanding via the inner product:

$$\text{MMD}^2(P, Q) = \langle \mu_P - \mu_Q, \mu_P - \mu_Q \rangle_{\mathcal{H}_k}$$
$$= \langle \mu_P, \mu_P \rangle - 2\langle \mu_P, \mu_Q \rangle + \langle \mu_Q, \mu_Q \rangle$$
$$= \mathbb{E}_{x, x' \sim P}[k(x, x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)] + \mathbb{E}_{y, y' \sim Q}[k(y, y')]$$

This formula requires only kernel evaluations — no density estimation, no explicit embedding computation. Given samples $\{x_i\}_{i=1}^m \sim P$ and $\{y_j\}_{j=1}^n \sim Q$, the unbiased estimator is:

$$\widehat{\text{MMD}}^2(P, Q) = \frac{1}{m(m-1)} \sum_{i \neq i'} k(x_i, x_{i'}) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j) + \frac{1}{n(n-1)} \sum_{j \neq j'} k(y_j, y_{j'})$$

The computational cost is $O(m^2 + mn + n^2)$, which is $O(n^2)$ for equal sample sizes. This compares favorably to the $O(n^3)$ cost of computing the Wasserstein distance (which requires solving a linear program).

**When does MMD = 0 imply P = Q?** The embedding $P \mapsto \mu_P$ is injective — meaning $\mu_P = \mu_Q \Rightarrow P = Q$ — if and only if $k$ is a **characteristic kernel**. The RBF kernel $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$ is characteristic on $\mathbb{R}^d$ for any $\sigma > 0$. The Laplace kernel is also characteristic. Polynomial kernels are not characteristic (they cannot detect all moment differences). The characterization uses the fact that the Fourier transform of the kernel's spectral measure must have full support.

> **Intuition:** The mean embedding $\mu_P$ encodes all the "moments" of $P$ that $\mathcal{H}_k$ can see. For a characteristic kernel, $\mathcal{H}_k$ is rich enough to see all moments — equivalently, functions in $\mathcal{H}_k$ are dense in $C_0(\mathcal{X})$. For polynomial kernels, $\mathcal{H}_k$ only sees polynomial moments; two distributions with identical polynomial moments but different tails would have $\mu_P = \mu_Q$ while $P \neq Q$.

**Hypothesis testing with MMD.** The two-sample test asks: given $\{x_i\} \sim P$ and $\{y_j\} \sim Q$, is $P = Q$? Under $H_0 : P = Q$, $\widehat{\text{MMD}}^2$ has a known null distribution (a weighted chi-squared under the incomplete U-statistic decomposition). The test statistic $n \cdot \widehat{\text{MMD}}^2 \to \sum_l \lambda_l (Z_l^2 - 1)$ where $Z_l \sim \mathcal{N}(0,1)$ i.i.d. and $\lambda_l$ are eigenvalues of the kernel operator under $P$. Permutation testing gives a distribution-free test with exact level $\alpha$.

> **Key insight:** MMD has a quadratic-time estimator ($O(n^2)$), a valid hypothesis test with known asymptotics, and equality to zero if and only if $P = Q$ (for characteristic kernels). It operates entirely through kernel evaluations, requiring no density estimation. These properties make it the workhorse for two-sample testing, generative model evaluation, and as a training objective.

**MMD as a training loss.** The generative moment matching network (Li et al., 2015; Dziugaite et al., 2015) trains a generator $G_\theta$ by minimizing $\text{MMD}^2(p_\text{data}, p_\theta)$ directly, computed from mini-batches. This is a fully simulation-based objective — no discriminator network is needed. Recent flow-matching models (Lipman et al., 2022) can be interpreted as minimizing Bregman divergences related to MMD objectives.

## Kernel Covariance Operators

**Cross-covariance operator.** Given a joint distribution $P_{XY}$ on $\mathcal{X} \times \mathcal{Y}$ with marginals $P_X$, $P_Y$, and kernels $k_X$, $k_Y$ with RKHSs $\mathcal{H}_{k_X}$, $\mathcal{H}_{k_Y}$, the **cross-covariance operator** $C_{YX} : \mathcal{H}_{k_X} \to \mathcal{H}_{k_Y}$ is defined by:

$$\langle g, C_{YX} f \rangle_{\mathcal{H}_{k_Y}} = \text{Cov}(f(X), g(Y)) = \mathbb{E}[f(X) g(Y)] - \mathbb{E}[f(X)] \mathbb{E}[g(Y)]$$

for all $f \in \mathcal{H}_{k_X}$, $g \in \mathcal{H}_{k_Y}$. This is well-defined: for each $f$, the map $g \mapsto \mathbb{E}[f(X)g(Y)] - \mathbb{E}[f(X)]\mathbb{E}[g(Y)]$ is a bounded linear functional on $\mathcal{H}_{k_Y}$ (bounded because $|f(X)| \leq \|f\|_{\mathcal{H}} \sqrt{k_X(X,X)}$ and the kernel is bounded), so the Riesz theorem gives a unique representative $C_{YX} f \in \mathcal{H}_{k_Y}$.

In Bochner integral form:

$$C_{YX} = \mathbb{E}[k_Y(\cdot, Y) \otimes k_X(\cdot, X)] - \mu_{P_Y} \otimes \mu_{P_X}$$

where $\otimes$ denotes the tensor product in $\mathcal{H}_{k_Y} \otimes \mathcal{H}_{k_X}$.

The covariance operator $C_{XX} : \mathcal{H}_{k_X} \to \mathcal{H}_{k_X}$ is the self-adjoint case:

$$C_{XX} = \mathbb{E}[k_X(\cdot, X) \otimes k_X(\cdot, X)] - \mu_{P_X} \otimes \mu_{P_X}$$

**Hilbert-Schmidt Independence Criterion.** The Hilbert-Schmidt norm of the cross-covariance operator is:

$$\|C_{YX}\|^2_{\text{HS}} = \text{HSIC}(X, Y)$$

Expanding using the definition of the Hilbert-Schmidt norm and the tensor product structure:

$$\text{HSIC}(X, Y) = \mathbb{E}_{XY}\mathbb{E}_{X'Y'}[k_X(X, X') k_Y(Y, Y')] - 2 \mathbb{E}_{XY}\mathbb{E}_{X'}\mathbb{E}_{Y'}[k_X(X,X') k_Y(Y,Y')] + \mathbb{E}_{X}\mathbb{E}_{X'}[k_X(X,X')] \cdot \mathbb{E}_{Y}\mathbb{E}_{Y'}[k_Y(Y,Y')]$$

where primed variables are independent copies. Given samples $\{(x_i, y_i)\}_{i=1}^n$, the biased estimator simplifies to:

$$\widehat{\text{HSIC}}_b = \frac{1}{n^2} \text{tr}(KHLH)$$

where $K_{ij} = k_X(x_i, x_j)$, $L_{ij} = k_Y(y_i, y_j)$, and $H = I - \frac{1}{n} \mathbf{1}\mathbf{1}^\top$ is the centering matrix. This is a degree-4 statistic in the samples.

**HSIC = 0 iff independence.** For characteristic kernels $k_X$ and $k_Y$:

$$\text{HSIC}(X, Y) = 0 \iff X \perp Y$$

The proof: $\|C_{YX}\|^2_{\text{HS}} = 0 \Rightarrow C_{YX} = 0 \Rightarrow \mathbb{E}[f(X)g(Y)] = \mathbb{E}[f(X)]\mathbb{E}[g(Y)]$ for all $f \in \mathcal{H}_{k_X}$, $g \in \mathcal{H}_{k_Y}$. Since characteristic kernels have dense RKHSs in $C_0$, this extends to all bounded continuous functions, which characterizes independence (from M2, mutual information and the factorization condition).

> **Refresher:** From M3 (information theory), mutual information $I(X;Y) = 0$ iff $X \perp Y$. HSIC provides an alternative nonparametric measure of dependence that does not require density estimation. Both equal zero iff independent; both are non-negative. HSIC is computationally more tractable (no binning or kernel density estimation required for entropy) but captures only dependences that $\mathcal{H}_{k_X} \otimes \mathcal{H}_{k_Y}$ can see.

**Applications.** HSIC is used as a regularizer in disentangled representation learning: minimize $\text{HSIC}(Z_i, Z_j)$ between different latent dimensions to encourage independence. In neural architecture search and feature selection, HSIC between each feature and the target measures relevance, while HSIC between features measures redundancy. The HSIC Lasso (Yamada et al., 2014) solves a group lasso problem with HSIC-based relevance scores.

## Integral Probability Metrics

**The IPM framework.** An **Integral Probability Metric (IPM)** between distributions $P$ and $Q$ is defined by a class $\mathcal{F}$ of real-valued "witness" functions:

$$\text{IPM}_\mathcal{F}(P, Q) = \sup_{f \in \mathcal{F}} \left| \mathbb{E}_P[f] - \mathbb{E}_Q[f] \right|$$

The intuition is: find the function $f$ in $\mathcal{F}$ that "sees" the biggest difference between $P$ and $Q$. The richness of $\mathcal{F}$ controls how sensitive the metric is.

**Special cases unify the major divergences.**

(1) $\mathcal{F} = \{f : \|f\|_{\mathcal{H}_k} \leq 1\}$ — the unit ball in an RKHS. By the reproducing property and Cauchy-Schwarz:

$$\sup_{\|f\|_{\mathcal{H}_k} \leq 1} |\mathbb{E}_P[f] - \mathbb{E}_Q[f]| = \sup_{\|f\|_{\mathcal{H}_k} \leq 1} |\langle f, \mu_P - \mu_Q \rangle_{\mathcal{H}_k}| = \|\mu_P - \mu_Q\|_{\mathcal{H}_k} = \text{MMD}(P, Q)$$

The supremum is attained by $f^* = (\mu_P - \mu_Q) / \|\mu_P - \mu_Q\|_{\mathcal{H}_k}$, the normalized "mean difference function."

(2) $\mathcal{F} = \{f : \|f\|_{\text{Lip}} \leq 1\}$ — the class of 1-Lipschitz functions. By the Kantorovich-Rubinstein duality (M5, L2):

$$\sup_{\|f\|_{\text{Lip}} \leq 1} |\mathbb{E}_P[f] - \mathbb{E}_Q[f]| = W_1(P, Q)$$

the 1-Wasserstein distance.

(3) $\mathcal{F} = \{f : \|f\|_\infty \leq 1\}$ — all bounded measurable functions. The supremum becomes:

$$\sup_{\|f\|_\infty \leq 1} |\mathbb{E}_P[f] - \mathbb{E}_Q[f]| = \text{TV}(P, Q) = \frac{1}{2}\|P - Q\|_{L^1}$$

the total variation distance.

(4) $\mathcal{F}$ a Sobolev ball $W^{s,2}$ (from L4) gives the **Sobolev IPM**, which interpolates between MMD (for finite-dimensional RKHS, $s$ large) and TV ($s \to 0$).

| Metric | Function class $\mathcal{F}$ | Sample complexity | Computation |
|--------|------------------------------|------------------|-------------|
| MMD | RKHS unit ball $\|f\|_{\mathcal{H}_k} \leq 1$ | $O(1/n)$ rate | $O(n^2)$ |
| $W_1$ | 1-Lipschitz $\|f\|_{\text{Lip}} \leq 1$ | $O(n^{-1/d})$ in dim $d$ | $O(n^3)$ LP |
| TV | Bounded $\|f\|_\infty \leq 1$ | Poor (requires density) | Intractable |
| Sinkhorn | Entropy-regularized OT (M5) | Between $W_1$ and MMD | $O(n^2)$ |

> **Key insight:** The IPM framework reveals that all these metrics are solving the same problem: find the function $f$ most sensitive to the distributional difference, subject to a constraint on $f$. The constraint determines the function class, which in turn determines the sample complexity (richer classes need more samples to estimate) and computational cost. MMD uses an RKHS constraint, which has a closed-form solution via the mean embedding. Wasserstein uses a Lipschitz constraint, which has no closed form but corresponds to a meaningful geometric distance. This tradeoff between discriminating power and computational tractability is fundamental.

**Discriminating power.** For $P$ and $Q$ supported on disjoint compact sets, all three metrics are positive. But for distributions that differ only in tail behavior, TV and $W_1$ may be more sensitive than MMD (which weights differences by the kernel bandwidth). The choice of metric should match the downstream task: if sample efficiency matters, use MMD; if geometric transport is meaningful, use $W_1$.

**Connection to M3 (information theory).** The KL divergence $D_{\text{KL}}(P \| Q)$ is not an IPM — it cannot be written as $\sup_{f \in \mathcal{F}} |\mathbb{E}_P[f] - \mathbb{E}_Q[f]|$ for any natural function class. The $f$-divergences (M3, L2) admit a variational representation $D_f(P \| Q) = \sup_{f} \mathbb{E}_P[f] - \mathbb{E}_Q[f^*]$ where $f^*$ is the convex conjugate, but this is not an IPM because the supremum is not over a symmetric constraint on $f$. IPMs and $f$-divergences are distinct families; both can be estimated from samples, but IPMs do not require density ratios while $f$-divergences do.

## Stein Operators and Kernelized Stein Discrepancy

**The Stein identity.** The **Stein operator** associated to a smooth density $p$ on $\mathbb{R}^d$ is:

$$(\mathcal{A}_p f)(x) = \nabla \log p(x) \cdot f(x) + \nabla \cdot f(x)$$

for vector-valued functions $f : \mathbb{R}^d \to \mathbb{R}^d$. The **Stein identity** states: for any smooth, suitably integrable $f$:

$$\mathbb{E}_{x \sim p}[(\mathcal{A}_p f)(x)] = 0$$

Proof by integration by parts:

$$\mathbb{E}_p[\mathcal{A}_p f] = \int \left( \nabla \log p(x) \cdot f(x) + \nabla \cdot f(x) \right) p(x) \, dx = \int \left( \frac{\nabla p(x)}{p(x)} \cdot f(x) + \nabla \cdot f(x) \right) p(x) \, dx$$
$$= \int \nabla p(x) \cdot f(x) \, dx + \int p(x) \nabla \cdot f(x) \, dx = \int \nabla \cdot (p(x) f(x)) \, dx = 0$$

where the last step uses the divergence theorem with boundary terms vanishing at infinity.

**Stein discrepancy.** If $q \neq p$, then $\mathbb{E}_{x \sim q}[(\mathcal{A}_p f)(x)] \neq 0$ in general. The **Stein discrepancy** of $q$ with respect to $p$ is:

$$\text{SD}(q \| p) = \sup_{f \in \mathcal{F}} \mathbb{E}_{x \sim q}[(\mathcal{A}_p f)(x)]$$

**The key property:** $\text{SD}(q \| p)$ requires only samples from $q$ and evaluations of $\nabla \log p(x)$ — not samples from $p$, and not the normalizing constant $Z$ in $p(x) = \tilde{p}(x)/Z$. This is crucial: for unnormalized models (energy-based models, score-based diffusion models), $\nabla \log p(x) = \nabla \log \tilde{p}(x)$ is computable without knowing $Z$.

**Kernelized Stein Discrepancy.** With $f$ ranging over the unit ball of a vector-valued RKHS $\mathcal{H}^d_k = \mathcal{H}_k \times \cdots \times \mathcal{H}_k$, the KSD has a closed form. Define the Stein kernel:

$$\kappa_p(x, x') = \nabla_x \log p(x)^\top \nabla_{x'} \log p(x') k(x, x') + \nabla_x \log p(x)^\top \nabla_{x'} k(x, x') + \nabla_{x'} \log p(x')^\top \nabla_x k(x, x') + \Delta_{x,x'} k(x, x')$$

where $\Delta_{x,x'} k(x, x') = \sum_i \frac{\partial^2}{\partial x_i \partial x'_i} k(x, x')$. Then:

$$\text{KSD}^2(q \| p) = \mathbb{E}_{x, x' \sim q}[\kappa_p(x, x')]$$

The estimator is:

$$\widehat{\text{KSD}}^2(q \| p) = \frac{1}{n(n-1)} \sum_{i \neq j} \kappa_p(x_i, x_j), \qquad \{x_i\}_{i=1}^n \sim q$$

This is computable from samples of $q$ alone, requiring only $\nabla \log p$ at those points.

> **Intuition:** The Stein operator "pushes" test functions through $p$, and if $q = p$ the expected output is zero. The kernelized version finds the best test function in the RKHS — by Mercer's theorem, the optimal test is the Stein kernel evaluated at the data points. The resulting discrepancy is a valid hypothesis test for $H_0 : q = p$ without samples from $p$.

**Application: Stein Variational Gradient Descent (SVGD).** SVGD (Liu and Wang, 2016) transports particles $\{x_i^{(0)}\}_{i=1}^n \sim q_0$ toward samples from a target $p$ (e.g., a posterior) via:

$$x_i \leftarrow x_i + \epsilon \, \phi^*(x_i), \qquad \phi^*(x) = \frac{1}{n} \sum_{j=1}^n \left[ k(x_j, x) \nabla_{x_j} \log p(x_j) + \nabla_{x_j} k(x_j, x) \right]$$

The update direction $\phi^*$ is the optimal perturbation in the RKHS that maximally decreases the KL divergence $D_{\text{KL}}(q_t \| p)$. The second term $\nabla_{x_j} k(x_j, x)$ is a repulsive force: it prevents all particles from collapsing to the mode. This connects to M2 (Langevin dynamics, L5: SDEs) and M4 (optimization, L2: gradient methods).

**Application: goodness-of-fit testing for score-based models.** Diffusion models and score-based generative models (Song and Ermon, 2019) learn $s_\theta(x) \approx \nabla \log p_\text{data}(x)$. To evaluate quality without generating samples, compute $\widehat{\text{KSD}}^2(p_\text{data} \| p_\theta)$ using the learned score $s_\theta$ in place of $\nabla \log p$, evaluated on held-out data points. This is a tractable quality metric that avoids both density evaluation and expensive generation.

## Koopman Operators for Dynamical Systems

**From nonlinear to linear.** Consider a discrete-time dynamical system on a state space $\mathcal{M}$:

$$x_{t+1} = F(x_t), \qquad x_t \in \mathcal{M}$$

The dynamics $F$ may be nonlinear and high-dimensional. The **Koopman operator** $\mathcal{K}$ acts on scalar-valued **observables** $g : \mathcal{M} \to \mathbb{R}$ by:

$$(\mathcal{K} g)(x) = g(F(x)) = g \circ F$$

$\mathcal{K}$ is a linear operator on the function space $L^2(\mathcal{M}, \mu)$ (where $\mu$ is an invariant measure), even though $F$ is nonlinear. This is the fundamental trade: replace a finite-dimensional nonlinear system with an infinite-dimensional linear system.

> **Key insight:** $\mathcal{K}$ is linear because composition with $F$ is linear in $g$: $\mathcal{K}(\alpha g + \beta h) = \alpha \mathcal{K} g + \beta \mathcal{K} h$. The dimension of $\mathcal{K}$ is infinite (it acts on all observables), but it is linear — and all the tools of operator theory from L1–L2 apply.

**Eigenfunctions and the spectral decomposition.** A function $\varphi : \mathcal{M} \to \mathbb{C}$ is a **Koopman eigenfunction** with eigenvalue $\lambda$ if:

$$\mathcal{K} \varphi = \lambda \varphi \iff \varphi(F(x)) = \lambda \varphi(x)$$

In the coordinate system given by the eigenfunctions, the dynamics become exactly linear:

$$\varphi_i(x_{t+1}) = \lambda_i \varphi_i(x_t) \implies \varphi_i(x_t) = \lambda_i^t \varphi_i(x_0)$$

If $\mathcal{M}$ is compact and $F$ is measure-preserving, $\mathcal{K}$ is a unitary operator on $L^2(\mu)$ (from L2: unitary operators have spectral radius 1 and form a complete spectral decomposition). For dissipative systems, $\mathcal{K}$ may not be unitary but still admits a spectral decomposition on suitable function spaces.

Any observable $g = \sum_i c_i \varphi_i$ evolves as:

$$g(x_t) = \mathcal{K}^t g(x_0) = \sum_i c_i \lambda_i^t \varphi_i(x_0)$$

This is exactly linear! The challenge is finding the eigenfunctions $\varphi_i$.

**Dynamic Mode Decomposition (DMD).** Given trajectory data $\{x_0, x_1, \ldots, x_T\}$, DMD approximates the Koopman operator in a finite-dimensional subspace. Assemble data matrices:

$$X = [x_0, x_1, \ldots, x_{T-1}], \quad X' = [x_1, x_2, \ldots, x_T]$$

Fit $X' \approx A X$ for a matrix $A \in \mathbb{R}^{d \times d}$. The DMD modes are the eigenvectors of $A$, and the eigenvalues of $A$ approximate the Koopman eigenvalues. In practice, DMD is computed via SVD: $A = X' V \Sigma^{-1} U^\top$ where $X = U \Sigma V^\top$, giving a numerically stable computation even in high dimensions. This connects to M1 (L1: SVD and low-rank approximation).

**Extended DMD (EDMD) and kernel DMD.** When the state space $\mathcal{M}$ is $d$-dimensional with small $d$, the standard DMD approximation in $\mathbb{R}^d$ may not capture the dynamics well if the relevant Koopman eigenfunctions are nonlinear. EDMD lifts the state to a richer feature space $\Phi(x) = [\varphi_1(x), \ldots, \varphi_N(x)]^\top$ and fits $\Phi(x_{t+1}) \approx K \Phi(x_t)$. Kernel DMD (Williams et al., 2015) performs this lift implicitly using a kernel $k$, computing the Koopman operator in the RKHS $\mathcal{H}_k$:

$$K_\mathcal{H} = (G')^\top G^{-1}$$

where $G_{ij} = k(x_i, x_j)$ and $G'_{ij} = k(x_i, x_{j+1})$ are Gram matrices of the data and its one-step successor. This connects Koopman theory directly to RKHS theory from L3.

**Continuous-time Koopman.** For flows $\dot{x} = f(x)$, the Koopman generator is:

$$\mathcal{L} g = f \cdot \nabla g$$

with $\mathcal{K}^t = e^{t \mathcal{L}}$ (operator exponential). Eigenvalues of $\mathcal{L}$ are the Lyapunov exponents of the system.

**Connection to reinforcement learning.** In RL, the value function $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$ satisfies the Bellman equation $V^\pi = r + \gamma P^\pi V^\pi$ where $P^\pi$ is the state transition operator. $P^\pi$ is exactly the Koopman operator for the stochastic dynamics under policy $\pi$! The Bellman equation is $(I - \gamma \mathcal{K}_{P^\pi}) V^\pi = r$, a resolvent equation for the Koopman operator. The spectral decomposition of $\mathcal{K}_{P^\pi}$ — proto-value functions (Mahadevan, 2005) — gives an efficient basis for value function approximation. This connects M4 (optimization, dynamic programming) to operator theory.

**Applications in science.** Koopman methods have been applied to:
- **Molecular dynamics**: eigenfunctions of the molecular Koopman operator identify slow collective coordinates (folding coordinates in protein folding). The TICA algorithm (time-lagged independent component analysis) is a linear DMD on molecular simulation data.
- **Climate science**: identifying teleconnection patterns and low-frequency climate modes as Koopman eigenfunctions. El Nino Southern Oscillation can be identified as a Koopman eigenfunction of the coupled ocean-atmosphere system.
- **Fluid mechanics**: spectral analysis of turbulence via DMD modes.

> **Intuition:** Koopman theory is a change of coordinates that makes the dynamics linear — at the cost of going infinite-dimensional. DMD is a finite-rank approximation of this infinite-dimensional linear system. The RKHS formulation gives the principled way to do this approximation: approximate in the RKHS, where the inner product structure (from L3) gives orthogonal projections and best approximations.

## Synthesis: A Unified Operator View of Modern ML

This module, and Semester 1 as a whole, has been building a single coherent framework. The following table shows where each ML method lives in the operator-theoretic hierarchy.

| ML Method | Operator | Space | Module connection |
|-----------|----------|-------|-------------------|
| Neural network training (infinite width) | NTK integral operator $T_\Theta$ | RKHS $\mathcal{H}_\Theta$ | M1 (eigendecomposition), L3 (RKHS) |
| Two-sample testing, generative evaluation | Mean embedding $\mu_P \in \mathcal{H}_k$ | RKHS $\mathcal{H}_k$ | M2 (hypothesis testing), L3 (RKHS) |
| Independence testing, disentanglement | Cross-covariance $C_{YX}$ (HS norm) | $\mathcal{H}_{k_X} \to \mathcal{H}_{k_Y}$ | M3 (mutual information), L2 (HS operators) |
| Wasserstein distance, OT | Optimal coupling (M5) and IPM $\mathcal{F} = \text{Lip-1}$ | $L^2(\mu)$ | M5 (OT), L4 (Sobolev = Lip-1 dual) |
| Score-based/energy models | Stein operator $\mathcal{A}_p$ | $\mathcal{H}_k^d$ | M2 (score functions), L3 (RKHS) |
| RL value functions | Koopman/transition operator $P^\pi$ | $L^2(\mathcal{M}, \mu)$ | M4 (dynamic programming), L2 (compact operators) |
| Time-series / dynamical systems | Koopman operator $\mathcal{K}$ | $L^2(\mathcal{M}, \mu)$ | M1 (SVD/DMD), L2 (spectral theory) |

**The five synthesis points:**

1. **Overparameterized generalization (NTK).** Modern networks are massively overparameterized yet generalize. The NTK explains this via the spectral bias of the kernel: the network's implicit RKHS norm acts as a regularizer, and gradient descent finds the minimum-RKHS-norm interpolant. This is a functional-analytic reformulation of the classical bias-variance tradeoff (M2).

2. **MMD as training loss (generative models).** Generative moment matching and modern flow-matching objectives minimize $\text{MMD}^2(p_\text{data}, p_\theta)$ directly. This is differentiable through the generator, requires no discriminator, and has theoretical guarantees (characteristic kernels give identifiability). The $O(n^2)$ cost is competitive with entropic OT.

3. **HSIC for disentanglement.** $\beta$-VAE and related models (Higgins et al., 2017) add a KL penalty to encourage independent latent codes. The operator-theoretic formulation via HSIC $= \|C_{YX}\|^2_\text{HS}$ gives a kernel test for independence that applies to any architecture, not just VAEs with Gaussian priors.

4. **KSD for score model training.** Training score-based models (denoising diffusion probabilistic models, Song et al., 2020) requires estimating $\nabla \log p_\text{data}$. The Stein operator provides both the training objective (score matching $\approx$ Stein identity) and the evaluation metric (KSD), all without requiring the data normalizing constant. This connects M2 (score functions, L5: SDEs) to Stein operator theory.

5. **Koopman for long-horizon RL and simulation.** Model-based RL requires learning transition dynamics. Koopman methods learn a globally linear representation of those dynamics, enabling long-horizon prediction by matrix exponentiation rather than expensive rollouts. Proto-value functions (Koopman eigenfunctions) provide state abstractions that transfer across tasks — a formal realization of representation learning.

> **Remember:** The unifying thread is operators on function spaces. An operator is a linear map between function spaces. The Hilbert space structure (from L1) gives inner products, adjoints, and spectral theory. The RKHS structure (from L3) gives reproducing kernels and mean embeddings. The Sobolev structure (from L4) gives regularity and Lipschitz constraints. Together, they provide the mathematical language for all of modern ML theory.

## Python: NTK, MMD, and HSIC

The following code is self-contained and requires only `numpy` and `scipy`. It demonstrates three core computations from this lesson: (1) finite-width NTK approximation and comparison to kernel regression, (2) unbiased MMD estimator, (3) HSIC computation.

```python
import numpy as np
from scipy.linalg import solve

# ============================================================
# Part 1: Neural Tangent Kernel — finite-width approximation
# ============================================================

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def build_network_and_ntk(X, width=512, depth=2, seed=42):
    """
    Two-layer ReLU network with NTK parameterization.
    Returns the finite-width NTK matrix Theta[i,j] = <grad f(x_i), grad f(x_j)>.
    NTK parameterization: outputs are O(1) at initialization regardless of width.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Layer 1: W1 shape (width, d), b1 shape (width,)
    W1 = rng.randn(width, d) / np.sqrt(d)
    b1 = rng.randn(width) / np.sqrt(d)
    # Layer 2: W2 shape (1, width), b2 scalar
    W2 = rng.randn(1, width) / np.sqrt(width)
    b2 = rng.randn(1) / np.sqrt(width)

    # Forward pass — compute pre-activations and activations
    Z1 = X @ W1.T + b1[None, :]        # (n, width)
    A1 = relu(Z1)                        # (n, width)
    f = (A1 @ W2.T + b2).squeeze(-1)    # (n,)

    # Gradients of f(x_i) wrt all parameters
    # f(x) = W2 @ relu(W1 x + b1) + b2
    # df/dW2[j] = relu(W1 x + b1)[j]         — shape (1, width)
    # df/db2    = 1
    # df/dW1[j,k] = W2[0,j] * relu'(Z1[j]) * x[k]
    # df/db1[j]   = W2[0,j] * relu'(Z1[j])

    # Build Jacobian: shape (n, num_params)
    dZ1 = relu_grad(Z1)  # (n, width)
    # Gradient wrt W2: (n, width)
    grad_W2 = A1  # each row is grad wrt W2 for that sample
    # Gradient wrt b2: (n, 1)
    grad_b2 = np.ones((n, 1))
    # Gradient wrt W1: (n, width, d) -> (n, width*d)
    # df/dW1[j,k] = W2[0,j] * dZ1[i,j] * x[i,k]
    delta = W2[0, :][None, :] * dZ1    # (n, width): scaled activation gradient
    grad_W1 = (delta[:, :, None] * X[:, None, :]).reshape(n, -1)  # (n, width*d)
    # Gradient wrt b1: (n, width)
    grad_b1 = delta  # (n, width)

    # Full Jacobian
    J = np.concatenate([grad_W2, grad_b2, grad_W1, grad_b1], axis=1)  # (n, P)

    # NTK matrix: Theta[i,j] = J[i,:] @ J[j,:]
    Theta = J @ J.T  # (n, n)

    return f, Theta, J

def ntk_kernel_regression(X_train, y_train, X_test, Theta_train, J_train, J_test):
    """
    Kernel regression in the NTK RKHS: f* = Theta(X_test, X_train) @ Theta_train^{-1} y
    """
    # Cross-NTK: Theta(X_test, X_train) = J_test @ J_train.T
    Theta_cross = J_test @ J_train.T  # (n_test, n_train)
    # Solve Theta_train @ alpha = y
    alpha = solve(Theta_train + 1e-6 * np.eye(len(y_train)), y_train)
    return Theta_cross @ alpha

# Generate 1D regression data
np.random.seed(0)
n_train = 30
X_train = np.linspace(-3, 3, n_train).reshape(-1, 1)
y_train = np.sin(X_train.squeeze()) + 0.1 * np.random.randn(n_train)

X_test = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)

# Finite-width NTK (width=2048 approximates infinite-width well for smooth f)
width = 2048
f_init, Theta_train, J_train = build_network_and_ntk(X_train, width=width)
_, _, J_test = build_network_and_ntk(
    np.vstack([X_train, X_test]), width=width
)
# J_test for the test points only (rows n_train onward)
J_test_only = J_test[n_train:]
J_train_check = J_test[:n_train]

# NTK kernel regression prediction
y_pred_ntk = ntk_kernel_regression(X_train, y_train, X_test, Theta_train, J_train, J_test_only)

# Compare: eigenvalue spectrum of NTK matrix
eigenvalues = np.linalg.eigvalsh(Theta_train)[::-1]
print("NTK eigenvalues (top 5):", np.round(eigenvalues[:5], 3))
print("NTK eigenvalues (bottom 5):", np.round(eigenvalues[-5:], 5))
print(f"Condition number of NTK: {eigenvalues[0]/eigenvalues[-1]:.1f}")
# High-eigenvalue modes are learned first during gradient descent (spectral bias)

# Simulate gradient flow on training outputs: df/dt = -Theta(f - y)
# Exact solution: f(t) = (I - exp(-Theta t)) y  [starting from f_0 = 0]
t_values = [0.01, 0.1, 1.0, 10.0]
lam, V = np.linalg.eigh(Theta_train)
print("\nTraining dynamics via NTK linear ODE:")
for t in t_values:
    decay = np.diag(np.exp(-lam * t))
    f_t = V @ (np.eye(n_train) - decay) @ V.T @ y_train
    residual = np.mean((f_t - y_train)**2)
    print(f"  t={t:.2f}: train MSE = {residual:.6f}")

print(f"\nNTK kernel regression train MSE: {np.mean((ntk_kernel_regression(X_train, y_train, X_train, Theta_train, J_train, J_train) - y_train)**2):.8f}")

# ============================================================
# Part 2: Maximum Mean Discrepancy with RBF kernel
# ============================================================

def rbf_kernel(X, Y, sigma=1.0):
    """Compute RBF kernel matrix K[i,j] = exp(-||x_i - y_j||^2 / (2 sigma^2))."""
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True)
    sq_dists = X2 + Y2.T - 2 * X @ Y.T
    return np.exp(-sq_dists / (2 * sigma**2))

def mmd_unbiased(X, Y, sigma=1.0):
    """
    Unbiased MMD^2 estimator.
    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    Uses U-statistic (excludes diagonal for XX and YY terms).
    """
    m, n = len(X), len(Y)
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    # Unbiased: exclude diagonal (i != i' terms)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.mean()  # all terms included (cross terms, no diagonal to exclude)

    return term_xx - 2 * term_xy + term_yy

# Test 1: Same distribution — MMD should be near 0
np.random.seed(1)
n_samples = 500
P1 = np.random.randn(n_samples, 2)
P2 = np.random.randn(n_samples, 2)
mmd_same = mmd_unbiased(P1, P2, sigma=1.0)
print(f"\nMMD^2 (same distribution, N=0,I): {mmd_same:.6f}  (expect ~0)")

# Test 2: Different distributions — MMD should be significantly positive
Q1 = np.random.randn(n_samples, 2) + np.array([2.0, 0.0])  # shifted by 2
mmd_diff = mmd_unbiased(P1, Q1, sigma=1.0)
print(f"MMD^2 (shifted by 2.0):           {mmd_diff:.6f}  (expect >> 0)")

# Test 3: Subtle difference — different covariance
Q2_raw = np.random.randn(n_samples, 2)
Q2 = Q2_raw @ np.array([[2.0, 0.5], [0.5, 0.5]])  # different covariance
mmd_subtle = mmd_unbiased(P1, Q2, sigma=1.0)
print(f"MMD^2 (different covariance):     {mmd_subtle:.6f}  (should detect)")

# Permutation test for significance
def mmd_permutation_test(X, Y, sigma=1.0, n_permutations=200, seed=42):
    """MMD two-sample test via permutation. Returns p-value."""
    rng = np.random.RandomState(seed)
    observed = mmd_unbiased(X, Y, sigma)
    combined = np.vstack([X, Y])
    m = len(X)
    null_stats = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:m]]
        Y_perm = combined[perm[m:]]
        null_stats.append(mmd_unbiased(X_perm, Y_perm, sigma))
    p_value = np.mean(np.array(null_stats) >= observed)
    return observed, p_value

obs_same, p_same = mmd_permutation_test(P1, P2, sigma=1.0)
obs_diff, p_diff = mmd_permutation_test(P1, Q1, sigma=1.0)
print(f"\nPermutation test (same dist):  MMD^2={obs_same:.5f}, p={p_same:.3f}")
print(f"Permutation test (diff dist):  MMD^2={obs_diff:.5f}, p={p_diff:.3f}")

# Bandwidth selection: use median heuristic
def median_bandwidth(X, Y):
    """Median heuristic: sigma = median of pairwise distances."""
    combined = np.vstack([X, Y])
    diffs = combined[:, None, :] - combined[None, :, :]
    sq_dists = np.sum(diffs**2, axis=-1)
    upper = sq_dists[np.triu_indices(len(combined), k=1)]
    return np.sqrt(np.median(upper) / 2)

sigma_med = median_bandwidth(P1, Q2)
print(f"\nMedian bandwidth for subtle test: sigma={sigma_med:.3f}")
obs_subtle, p_subtle = mmd_permutation_test(P1, Q2, sigma=sigma_med)
print(f"Permutation test (covar diff, median bw): MMD^2={obs_subtle:.5f}, p={p_subtle:.3f}")

# ============================================================
# Part 3: Hilbert-Schmidt Independence Criterion (HSIC)
# ============================================================

def hsic_biased(X, Y, sigma_x=1.0, sigma_y=1.0):
    """
    Biased HSIC estimator: HSIC = (1/n^2) * tr(K H L H)
    where H = I - (1/n) 11^T is the centering matrix.
    """
    n = len(X)
    K = rbf_kernel(X, X, sigma_x)
    L = rbf_kernel(Y, Y, sigma_y)
    H = np.eye(n) - np.ones((n, n)) / n
    # tr(K H L H) = tr((HKH) L) by cyclic trace property and H = H^T, H^2 = H
    KH = H @ K @ H
    return np.trace(KH @ L) / n**2

# Test 1: Independent variables
np.random.seed(2)
n = 300
X_indep = np.random.randn(n, 1)
Y_indep = np.random.randn(n, 1)
hsic_indep = hsic_biased(X_indep, Y_indep)
print(f"\nHSIC (independent X, Y): {hsic_indep:.6f}  (expect ~0)")

# Test 2: Linearly dependent
Y_linear = X_indep + 0.2 * np.random.randn(n, 1)
hsic_linear = hsic_biased(X_indep, Y_linear)
print(f"HSIC (linear dep. Y=X+noise): {hsic_linear:.6f}  (expect > 0)")

# Test 3: Nonlinear dependence (captures what correlation misses)
Y_nonlinear = X_indep**2 + 0.2 * np.random.randn(n, 1)
hsic_nonlinear = hsic_biased(X_indep, Y_nonlinear)
linear_corr = np.corrcoef(X_indep.squeeze(), Y_nonlinear.squeeze())[0, 1]
print(f"HSIC (Y=X^2+noise): {hsic_nonlinear:.6f}  (captures nonlinear dep.)")
print(f"Pearson corr (Y=X^2+noise): {linear_corr:.4f}  (near 0, misses the dependence)")

# HSIC permutation test
def hsic_permutation_test(X, Y, sigma_x=1.0, sigma_y=1.0, n_permutations=200, seed=42):
    rng = np.random.RandomState(seed)
    observed = hsic_biased(X, Y, sigma_x, sigma_y)
    null_stats = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(Y))
        null_stats.append(hsic_biased(X, Y[perm], sigma_x, sigma_y))
    p_value = np.mean(np.array(null_stats) >= observed)
    return observed, p_value

_, p_indep = hsic_permutation_test(X_indep, Y_indep)
_, p_nonlinear = hsic_permutation_test(X_indep, Y_nonlinear)
print(f"\nHSIC permutation test p-values:")
print(f"  Independent:          p={p_indep:.3f}  (should be large, fail to reject H0)")
print(f"  Nonlinear dependent:  p={p_nonlinear:.3f}  (should be small, reject H0)")

# Compare HSIC to mutual information (via binning estimator from M3)
def mi_binning(x, y, n_bins=15):
    """
    Binning-based mutual information estimate (from M3).
    Discretize continuous variables and compute I(X;Y) from joint histogram.
    """
    hist_joint, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_x = hist_joint.sum(axis=1)
    hist_y = hist_joint.sum(axis=0)
    # Convert to probabilities
    p_joint = hist_joint / hist_joint.sum()
    p_x = hist_x / hist_x.sum()
    p_y = hist_y / hist_y.sum()
    # MI = sum p(x,y) log(p(x,y) / (p(x) p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_x[i] * p_y[j]))
    return mi

mi_indep = mi_binning(X_indep.squeeze(), Y_indep.squeeze())
mi_nonlinear = mi_binning(X_indep.squeeze(), Y_nonlinear.squeeze())
print(f"\nMutual information (binning estimator from M3):")
print(f"  Independent: I(X;Y) = {mi_indep:.4f} nats  (expect ~0)")
print(f"  Y=X^2+noise: I(X;Y) = {mi_nonlinear:.4f} nats  (expect > 0)")
print(f"\nHSIC detects Y=X^2 dependence without density estimation (HSIC={hsic_nonlinear:.4f})")
print(f"MI binning requires discretization and is sensitive to bin count.")
```

Expected output:
```
NTK eigenvalues (top 5): [large values depending on width]
Condition number of NTK: [large, reflects spectral bias]
Training dynamics via NTK linear ODE:
  t=0.01: train MSE = [close to ||y||^2, little learning]
  t=10.00: train MSE = [very small, near interpolation]
MMD^2 (same distribution, N=0,I): ~0.000  (near zero)
MMD^2 (shifted by 2.0):           ~0.5+  (clearly positive)
Permutation test (same dist):  p=0.5+   (fail to reject)
Permutation test (diff dist):  p=0.000  (reject H0)
HSIC (independent X, Y): ~0.000
HSIC (Y=X^2+noise): [positive value]
Pearson corr (Y=X^2+noise): ~0.00   (near zero, misses nonlinear dep.)
HSIC permutation test: nonlinear dep. p-value near 0
```

**Implementation notes:**
- The NTK is computed for a finite-width network ($P \approx 10^6$ parameters at width 2048), which approximates the infinite-width NTK well for smooth targets.
- MMD bandwidth is critical: the median heuristic adapts the RBF bandwidth to the data scale.
- HSIC with the biased estimator converges at rate $O(1/\sqrt{n})$; an unbiased estimator (Song et al., 2012) converges faster and is preferred for small $n$.
- Both MMD and HSIC are quadratic in $n$; for large datasets, use random Fourier features (Rahimi and Recht, 2007) to reduce to $O(n D)$ cost with $D$ random features.

:::quiz
**Question 1**

A two-layer infinite-width ReLU network is trained with gradient flow on $n$ training points. The NTK matrix $\Theta^\infty(X, X)$ has eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n > 0$.

After training for time $t$, what fraction of the component of $y$ in the direction of eigenvector $v_1$ (the top eigenvector) has been learned?

A) $1 - e^{-\lambda_n t}$

B) $e^{-\lambda_1 t}$

C) $1 - e^{-\lambda_1 t}$

D) $\lambda_1 / \lambda_n$

**Correct answer: C**

The output trajectory is $\hat{f}(t) = (I - e^{-\Theta^\infty t}) y$. Projecting onto eigenvector $v_1$: $v_1^\top \hat{f}(t) = (1 - e^{-\lambda_1 t}) v_1^\top y$. The fraction learned is $(1 - e^{-\lambda_1 t})$. Since $\lambda_1$ is the largest eigenvalue, this component is learned fastest — this is spectral bias. Option A uses the smallest eigenvalue (slowest mode). Option B is the fraction not yet learned (the residual).
:::

:::quiz
**Question 2**

You want to test whether a generative model $q$ has learned the data distribution $p_\text{data}$. The model is an energy-based model (EBM): you can evaluate $\tilde{p}(x) = e^{-E_\theta(x)}$ and its gradient $\nabla_x \log \tilde{p}(x) = -\nabla_x E_\theta(x)$, but you cannot sample from $p_\theta$ without expensive MCMC, and you do not know the normalizing constant $Z = \int e^{-E_\theta(x)} dx$.

Which evaluation metric is most appropriate?

A) MMD$(p_\text{data}, p_\theta)$ using samples from both $p_\text{data}$ and $p_\theta$

B) KL divergence $D_{\text{KL}}(p_\text{data} \| p_\theta)$ estimated via importance sampling

C) Kernelized Stein Discrepancy KSD$(p_\text{data} \| p_\theta)$ using held-out data from $p_\text{data}$ and the score $\nabla \log \tilde{p}_\theta$

D) Wasserstein distance $W_1(p_\text{data}, p_\theta)$ via Sinkhorn with samples from both

**Correct answer: C**

KSD requires only (a) samples from $q = p_\text{data}$ (held-out data) and (b) the ability to evaluate $\nabla \log p_\theta(x)$ at those points, which equals $-\nabla_x E_\theta(x)$ without needing $Z$. Options A and D require samples from $p_\theta$, which is expensive (MCMC). Option B requires estimating the density ratio $p_\text{data} / p_\theta$, which requires $Z$. KSD is specifically designed for exactly this scenario: unnormalized models where sampling is expensive but the score is cheap to evaluate.
:::

:::quiz
**Question 3**

For distributions $P$ and $Q$ on $\mathbb{R}^d$ with $d$ large, the IPM framework implies different sample complexities for different metrics. Specifically:

- MMD with RBF kernel has estimation error $O(1/\sqrt{n})$, independent of $d$
- 1-Wasserstein distance has estimation error $O(n^{-1/d})$ for $d \geq 3$

A researcher in high dimensions ($d = 100$) needs $n$ samples to achieve error $\varepsilon = 0.01$ in estimating the distance between two distributions.

For MMD, $n \approx 1/\varepsilon^2 = 10{,}000$. For $W_1$, $n \approx (1/\varepsilon)^d$.

What is the ratio $n_{W_1} / n_{\text{MMD}}$ (approximate order of magnitude)?

A) About 10

B) About $10^{100}$

C) About $10^{200}$

D) About $10^{196}$

**Correct answer: D**

$n_{\text{MMD}} = (1/\varepsilon)^2 = (100)^2 = 10{,}000 = 10^4$. $n_{W_1} = (1/\varepsilon)^d = (100)^{100} = 10^{200}$. The ratio is $10^{200} / 10^4 = 10^{196}$. This is the curse of dimensionality for the Wasserstein distance: the Lipschitz-1 class is so rich in $\mathbb{R}^{100}$ that an astronomical number of samples are needed to estimate the supremum reliably. MMD avoids this because the RKHS unit ball in an RBF-kernel RKHS has a metric entropy that does not grow with $d$ in the same way — the smooth functions in $\mathcal{H}_k$ are "simpler" than arbitrary Lipschitz functions. This is the fundamental sample-complexity advantage of MMD in high dimensions.
:::

## Summary and Looking Ahead

This lesson completed the functional-analytic foundations of modern machine learning. The core objects are:

- **The NTK**: a Hilbert-Schmidt integral operator on $L^2(\mathcal{X})$ whose eigenfunctions determine what an infinite-width network learns and in what order. Training is a linear ODE in function space.
- **Mean embeddings**: the map $P \mapsto \mu_P \in \mathcal{H}_k$ turns distributions into vectors in a Hilbert space. Distance becomes $\|\mu_P - \mu_Q\|_{\mathcal{H}_k} = \text{MMD}(P,Q)$, computable in $O(n^2)$ with no density estimation.
- **Covariance operators**: $C_{YX} : \mathcal{H}_{k_X} \to \mathcal{H}_{k_Y}$ encodes the joint distribution of $(X,Y)$. Its Hilbert-Schmidt norm is HSIC, a nonparametric independence measure.
- **IPMs**: a unified framework showing MMD, Wasserstein, and TV are all the same operation — sup over a function class — with different constraints. The choice of constraint determines discriminating power, sample complexity, and computation.
- **Stein operators**: $\mathcal{A}_p$ turns the Stein identity into a tractable discrepancy (KSD) for unnormalized models, powering both SVGD and score-based model evaluation.
- **Koopman operators**: $\mathcal{K} g = g \circ F$ linearizes nonlinear dynamics, connecting RL (Bellman = Koopman resolvent), molecular simulation, and climate modeling.

Semester 2 builds on this foundation directly: the theory of deep learning generalization (connecting NTK theory to PAC learning bounds from M2), the mathematics of diffusion models (connecting Stein operators to SDEs from M2-L5), and the theory of representation learning (connecting covariance operators to information geometry).

**References:**
- Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. *NeurIPS*.
- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., and Smola, A. (2012). A Kernel Two-Sample Test. *JMLR*, 13:723–773.
- Gretton, A., Fukumizu, K., Teo, C., Song, L., Schölkopf, B., and Smola, A. (2008). A Kernel Statistical Test of Independence. *NeurIPS*.
- Liu, Q. and Wang, D. (2016). Stein Variational Gradient Descent. *NeurIPS*.
- Williams, M., Rowley, C., and Kevrekidis, I. (2015). A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition. *Journal of Nonlinear Science*, 25(6):1307–1346.
- Mahadevan, S. (2005). Proto-Value Functions: Developmental Reinforcement Learning. *ICML*.
- Müller, A. (1997). Integral Probability Metrics and Their Generating Classes of Functions. *Advances in Applied Probability*.
- Chung, K. T., Seo, J., and Kim, H. (2022). A Survey on Koopman Operator Methods in Machine Learning. *IEEE Transactions on Neural Networks*.
