---
title: "Eigendecomposition & Spectral Theory"
estimatedMinutes: 25
tags: ["eigenvalues", "spectral-theorem", "PSD", "graph-laplacian", "normalization"]
prerequisites: ["l1-svd-low-rank"]
---

# Eigendecomposition & Spectral Theory

Where the SVD applies to any matrix, eigendecomposition reveals the internal dynamics of square matrices — and for the symmetric matrices that pervade ML (covariance matrices, kernel matrices, Hessians, graph Laplacians), the spectral theorem guarantees a clean orthogonal decomposition with real eigenvalues. This lesson develops eigendecomposition, the spectral theorem, and their deep connections to the structures that make machine learning work.

## Eigendecomposition Fundamentals

> **Intuition:** An eigenvector is a direction that the matrix doesn't rotate — it only stretches or flips. For any other vector, applying $A$ changes both magnitude and direction. Eigendecomposition asks: what are the special directions where $A$ acts like pure scalar multiplication? Finding these directions transforms the matrix into a coordinate system where its action is completely transparent.

A scalar $\lambda$ is an **eigenvalue** of $A \in \mathbb{R}^{n \times n}$ with corresponding **eigenvector** $v \neq 0$ if:

$$Av = \lambda v$$

The matrix $A$ stretches or flips $v$ without changing its direction. The set of all eigenvalues is the **spectrum** of $A$.

If $A$ has $n$ linearly independent eigenvectors, we can form the **eigendecomposition**:

$$A = Q \Lambda Q^{-1}$$

where $Q$ has eigenvectors as columns and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$. Not every matrix is diagonalizable — the key condition is that the **geometric multiplicity** (dimension of the eigenspace) equals the **algebraic multiplicity** (multiplicity as a root of the characteristic polynomial) for every eigenvalue. Defective matrices (where these differ) require Jordan normal form, which arises rarely in ML practice but matters theoretically.

> **Key insight:** Eigendecomposition transforms a matrix into a coordinate system where its action is pure scaling. This is why eigenvectors are the natural basis for understanding linear operators.

## The Spectral Theorem

The spectral theorem is the foundation of spectral methods in ML.

> **Refresher:** A symmetric matrix satisfies $A = A^T$ — its $(i,j)$ entry equals its $(j,i)$ entry. This means the matrix "looks the same" from row and column perspectives. Symmetry arises naturally whenever $A$ encodes undirected relationships: the covariance between features $i$ and $j$ is the same as between $j$ and $i$; the similarity of point $i$ to point $j$ equals the similarity of $j$ to $i$. The spectral theorem is why symmetric matrices are so analytically tractable.

**Theorem.** If $A \in \mathbb{R}^{n \times n}$ is symmetric ($A = A^T$), then:

$$A = Q \Lambda Q^T$$

where $Q$ is orthogonal ($Q^T Q = I$) and $\Lambda$ is diagonal with **real** eigenvalues.

Three guarantees distinguish the symmetric case: (1) all eigenvalues are real, (2) eigenvectors corresponding to distinct eigenvalues are orthogonal, and (3) there always exists a full set of $n$ orthonormal eigenvectors, even when eigenvalues repeat. The factorization $Q \Lambda Q^T$ is an orthogonal change of basis — no stretching of the coordinate system, just rotation.

An equivalent outer-product form is often more illuminating:

$$A = \sum_{i=1}^n \lambda_i q_i q_i^T$$

Each term $\lambda_i q_i q_i^T$ is a rank-1 symmetric matrix that projects onto the $i$-th eigenvector and scales by $\lambda_i$. The matrix $A$ is a weighted sum of orthogonal projections.

## Positive Semidefinite Matrices

A symmetric matrix $A$ is **positive semidefinite** (PSD), written $A \succeq 0$, if any of the following equivalent conditions holds:

- All eigenvalues satisfy $\lambda_i \geq 0$
- $x^T A x \geq 0$ for all $x \in \mathbb{R}^n$
- $A = B^T B$ for some matrix $B$

PSD matrices are everywhere in ML:

- **Covariance matrices**: $\Sigma = \mathbb{E}[(x - \mu)(x - \mu)^T]$ is always PSD (it equals $X^T X / n$ for centered data).
- **Gram matrices**: $G_{ij} = \langle x_i, x_j \rangle$ — the matrix of inner products between data points.
- **Kernel matrices**: $K_{ij} = k(x_i, x_j)$ for any positive definite kernel $k$. Mercer's theorem guarantees $K \succeq 0$.
- **Hessians at local minima**: at a local minimum of a smooth function, $\nabla^2 f \succeq 0$.

If $A \succ 0$ (strictly positive definite, all $\lambda_i > 0$), then $A$ is invertible and defines a valid inner product $\langle x, y \rangle_A = x^T A y$. This is the Mahalanobis inner product used in metric learning.

> **Key insight:** Checking whether a matrix is PSD is equivalent to checking that all its eigenvalues are non-negative. This single spectral condition governs whether covariance matrices are valid, whether kernel matrices define valid similarity measures, and whether optimization has reached a local minimum.

## Matrix Functions via Spectral Decomposition

> **Intuition:** Matrix functions work by the same logic as the spectral decomposition itself. Rotate into the eigenbasis (via $Q^T$), apply the scalar function to each eigenvalue independently (since the matrix is now diagonal), then rotate back (via $Q$). This is only possible because symmetric matrices have an orthonormal eigenbasis — the "rotation" is clean. The result is that any function you can apply to a number, you can apply to a symmetric matrix via its eigenvalues.

Given $A = Q \Lambda Q^T$, we can define **matrix functions** by applying scalar functions to the eigenvalues:

$$f(A) = Q \, f(\Lambda) \, Q^T = Q \, \text{diag}(f(\lambda_1), \ldots, f(\lambda_n)) \, Q^T$$

Important examples in ML:

- **Matrix exponential**: $e^A = Q \, \text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n}) \, Q^T$. Appears in continuous-time dynamics, matrix Lie groups, and the Fisher-Rao metric.
- **Matrix square root**: $A^{1/2} = Q \, \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n}) \, Q^T$ (requires $A \succeq 0$). Used in whitening transforms and natural gradient preconditioning.
- **Matrix logarithm**: $\log A = Q \, \text{diag}(\log \lambda_1, \ldots, \log \lambda_n) \, Q^T$ (requires $A \succ 0$). Appears in Riemannian optimization on the manifold of SPD matrices.
- **Matrix inverse**: $A^{-1} = Q \, \text{diag}(\lambda_1^{-1}, \ldots, \lambda_n^{-1}) \, Q^T$. This reveals why near-zero eigenvalues cause numerical instability.

## The Graph Laplacian and Spectral Graph Theory

> **Refresher:** The graph Laplacian $L = D - W$ encodes graph structure in a matrix. The degree matrix $D$ captures how many edges each node has; the adjacency matrix $W$ captures which nodes are connected. Their difference $L$ has a key property: $x^T L x = \sum_{(i,j) \in E} W_{ij}(x_i - x_j)^2$, which measures how much a signal $x$ on the graph varies across edges. Small eigenvalues of $L$ correspond to smooth signals (slowly varying across edges); large eigenvalues correspond to rapidly oscillating signals. This is the graph analogue of Fourier frequency.

Given an undirected weighted graph with adjacency matrix $W$ (where $W_{ij} \geq 0$), the **graph Laplacian** is:

$$L = D - W$$

where $D = \text{diag}(\sum_j W_{ij})$ is the degree matrix. The normalized Laplacian $\tilde{L} = D^{-1/2} L D^{-1/2}$ has eigenvalues in $[0, 2]$.

Key spectral properties of $L$:

- $L$ is symmetric PSD, so all eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$.
- The multiplicity of $\lambda = 0$ equals the number of connected components.
- The **Fiedler value** $\lambda_2$ (the spectral gap) measures how well-connected the graph is. Large $\lambda_2$ means the graph is hard to cut — information flows freely.

**Graph Neural Networks** can be understood as spectral filters. A spectral graph convolution applies a learnable function $g_\theta(\Lambda)$ to graph signals:

$$g_\theta(L) \, x = Q \, g_\theta(\Lambda) \, Q^T x$$

ChebNet and GCN approximate this with polynomials of $L$ to avoid the $O(n^2)$ eigendecomposition, but the spectral perspective explains *what* these networks compute: they filter graph signals by amplifying or suppressing different frequency components (eigenvectors of $L$).

**Note on GCN normalization:** The standard GCN formulation (Kipf & Welling, 2017) uses the **normalized** Laplacian $\tilde{L} = D^{-1/2}LD^{-1/2}$, which also has orthonormal eigenvectors, so the spectral convolution formula $g_\theta(L)\,x = Q\,g_\theta(\Lambda)\,Q^T x$ holds for both normalized and unnormalized variants. Standard GCN implementations use the normalized version for numerical stability, as it bounds eigenvalues to $[0, 2]$.

## Layer Normalization: A Spectral View

LayerNorm computes, for each token's activation vector $x \in \mathbb{R}^d$:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu, \sigma$ are the mean and standard deviation of $x$'s components.

Viewed spectrally: across a batch of activations, the Gram matrix $G = X X^T$ has an eigenvalue spectrum. Without normalization, a few eigenvalues dominate (activations concentrate along a few directions), causing gradient instabilities. As an interpretive lens (rather than a formal theorem), LayerNorm can be understood as compressing the eigenvalue spread of $G$ toward uniformity by centering and scaling the activation distribution. This is a useful geometric intuition rather than a mathematically proven statement — the precise relationship between LayerNorm and the spectral properties of activations is an active area of analysis. Nonetheless, the practical effect is that the condition number of the effective Gram matrix stays bounded, stabilizing the backward pass.

## Condition Number

> **Intuition:** The condition number answers the question: how much does a small error in the input amplify into an error in the output? If $\kappa(A) = 10^6$, an error of size $10^{-6}$ in $A$ can produce an error of size 1 in the solution — catastrophic loss of precision. For optimization, the condition number of the Hessian determines how different the curvature is in different directions of parameter space. A large condition number means gradient descent takes tiny steps in the flat directions while oscillating in the steep ones.

The **condition number** of an invertible matrix is:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

For symmetric matrices, $\kappa(A) = |\lambda_{\max}| / |\lambda_{\min}|$.

A large condition number means the matrix is nearly singular — small perturbations to the input cause large perturbations to the output. In optimization, the condition number of the Hessian determines the convergence rate of gradient descent: for a quadratic $f(x) = \frac{1}{2} x^T A x$, gradient descent converges at rate $\left(\frac{\kappa - 1}{\kappa + 1}\right)^2$ per step. When $\kappa = 10^6$, this is essentially 1 — convergence is glacial without preconditioning.

> **Key insight:** Ill-conditioned weight matrices create a vicious cycle: gradients along the smallest singular direction are amplified relative to the largest, causing oscillation. Preconditioning (Adam, natural gradient) and normalization (BatchNorm, LayerNorm) both attack this problem by reshaping the spectrum.

## ML Connections

Eigendecomposition underlies the spectral theory that connects neural network architectures to graph structure, explains why normalization works, and determines the stability of optimization.

- **Graph Neural Networks (GNNs):** The Graph Convolutional Network (GCN) layer $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W)$ uses the normalized graph Laplacian $\hat{A} = D^{-1/2}AD^{-1/2}$. The eigenvectors of the Laplacian define the "graph Fourier basis" — spectral GNNs like ChebNet and GCN are essentially polynomial filters in the Laplacian eigenspectrum.
- **LayerNorm and BatchNorm:** Normalization layers reduce the condition number $\kappa(H)$ of the Hessian of the loss by equalizing the curvature across directions. When $\kappa$ is large (extreme eigenvalue ratio), gradient descent oscillates; normalization collapses the eigenvalue spread, accelerating convergence.
- **Hessian Analysis:** The Hessian of the loss at a local minimum has eigenvalues that reveal sharpness (largest eigenvalue) and effective dimensionality (number of significant eigenvalues). Tools like PyHessian compute the Hessian spectrum to understand generalization and guide learning rate selection.
- **Spectral Normalization for GANs:** Constraining each weight matrix to have spectral norm (largest singular value) ≤ 1 enforces Lipschitz continuity on the discriminator — essential for Wasserstein GANs and stable GAN training.
- **PSD Matrices in Gaussian Processes:** The kernel matrix $K$ in a Gaussian process is required to be positive semidefinite; its eigendecomposition determines the principal modes of variation in the prior. Choosing a good kernel means designing a PSD matrix with the right eigenspectrum.

> **Key insight:** The eigenspectrum of key matrices — Hessians, Laplacians, kernel matrices, weight matrices — is a diagnostic tool for understanding deep learning. Sharpness, effective rank, connectivity, and convergence speed all reduce to questions about eigenvalues.

## Python: Spectral Methods in Practice

```python
import numpy as np
from scipy.linalg import expm

# Eigendecomposition of a covariance matrix
np.random.seed(42)
X = np.random.randn(500, 10)  # 500 samples, 10 features
X[:, :3] = X[:, :3] @ np.array([[3, 1, 0], [1, 2, 0.5], [0, 0.5, 1]])
cov = X.T @ X / len(X)

eigenvalues, Q = np.linalg.eigh(cov)  # eigh for symmetric matrices
print(f"PSD check: all eigenvalues >= 0? {np.all(eigenvalues >= -1e-10)}")
print(f"Condition number: {eigenvalues[-1] / eigenvalues[0]:.1f}")

# Matrix exponential via eigendecomposition vs scipy
exp_cov_spectral = Q @ np.diag(np.exp(eigenvalues)) @ Q.T
exp_cov_scipy = expm(cov)
print(f"Spectral vs scipy expm error: {np.linalg.norm(exp_cov_spectral - exp_cov_scipy):.2e}")

# Graph Laplacian: ring graph with 50 nodes
n = 50
W = np.zeros((n, n))
for i in range(n):
    W[i, (i + 1) % n] = 1
    W[i, (i - 1) % n] = 1
D = np.diag(W.sum(axis=1))
L = D - W
eigs_L = np.linalg.eigvalsh(L)
print(f"Spectral gap (Fiedler value): {eigs_L[1]:.4f}")
print(f"Number of connected components: {np.sum(eigs_L < 1e-10)}")
```

:::quiz
question: "A symmetric matrix A has eigenvalues {5, 3, 3, 0, 0}. Which statements are true? (i) A is PSD. (ii) A is invertible. (iii) rank(A) = 3."
options:
  - "Only (i) and (iii)"
  - "Only (i) and (ii)"
  - "(i), (ii), and (iii)"
  - "Only (iii)"
correct: 0
explanation: "All eigenvalues are >= 0, so A is PSD (i). Two eigenvalues are 0, so A is singular (not invertible), ruling out (ii). The rank equals the number of nonzero eigenvalues = 3, confirming (iii). So (i) and (iii) are true."
:::

:::quiz
question: "In a graph neural network using spectral convolution, what does applying a low-pass filter g_theta(Lambda) to a graph signal accomplish?"
options:
  - "It removes nodes with low degree from the computation"
  - "It smooths the signal across the graph, making neighboring nodes more similar"
  - "It increases the spectral gap of the Laplacian"
  - "It converts the graph from directed to undirected"
correct: 1
explanation: "The eigenvectors of the graph Laplacian with small eigenvalues represent low-frequency (smooth) components — signals that vary slowly across edges. A low-pass filter retains these and suppresses high-frequency (rapidly varying) components, effectively smoothing the signal so that connected nodes receive similar values. This is exactly what GCN-style message passing does."
:::

:::quiz
question: "You are training a linear model and observe that gradient descent converges extremely slowly despite a reasonable learning rate. The Hessian has eigenvalues ranging from 0.001 to 1000. What is the most likely explanation?"
options:
  - "The loss function has multiple local minima"
  - "The condition number is 10^6, causing gradients to oscillate along ill-conditioned directions"
  - "The learning rate is too high"
  - "The model is underfitting due to insufficient parameters"
correct: 1
explanation: "The condition number kappa = 1000/0.001 = 10^6. Gradient descent on a quadratic with this condition number converges at rate ((kappa-1)/(kappa+1))^2 per step, which is essentially 1.0 — near-zero progress per iteration. The fix is preconditioning (e.g., Newton's method, Adam) to equalize the eigenvalue spread."
:::
