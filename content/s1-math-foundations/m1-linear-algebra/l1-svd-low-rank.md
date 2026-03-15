---
title: "SVD & Low-Rank Approximations"
estimatedMinutes: 30
tags: ["SVD", "low-rank", "PCA", "LoRA", "attention", "compression"]
prerequisites: []
---

# SVD & Low-Rank Approximations

The singular value decomposition is arguably the single most important factorization in applied mathematics. It reveals the geometric skeleton of any linear map, provides optimal low-rank approximations, and underpins techniques from principal component analysis to the parameter-efficient fine-tuning methods reshaping how we adapt large language models. This lesson develops SVD rigorously and then traces its influence through the core machinery of modern ML.

## The Singular Value Decomposition

Every real matrix $A \in \mathbb{R}^{m \times n}$ admits a factorization

$$A = U \Sigma V^T$$

where $U \in \mathbb{R}^{m \times m}$ is orthogonal (its columns are the **left singular vectors**), $V \in \mathbb{R}^{n \times n}$ is orthogonal (its columns are the **right singular vectors**), and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$ called the **singular values**.

In plain English: any linear map can be decomposed into three steps — rotate the input space (via $V^T$), scale along the coordinate axes (via $\Sigma$), then rotate the output space (via $U$). The singular values tell you how much each axis gets stretched or compressed.

**Existence and uniqueness.** The SVD always exists. The singular values are unique. The singular vectors are unique up to sign when the corresponding singular values are distinct; when singular values coincide, the corresponding singular vectors span a well-defined subspace but individual vectors within that subspace are not unique.

The proof is constructive: the columns of $V$ are eigenvectors of $A^T A$, the columns of $U$ are eigenvectors of $A A^T$, and $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are the eigenvalues of $A^T A$ (which are guaranteed non-negative since $A^T A$ is positive semidefinite).

> **Key insight:** The SVD exists for *every* matrix — rectangular, rank-deficient, even the zero matrix. This universality is what makes it the default tool for understanding linear structure.

## Truncated SVD and the Eckart-Young Theorem

Given the SVD $A = U \Sigma V^T$, define the **rank-$k$ truncated SVD**:

$$A_k = U_k \Sigma_k V_k^T$$

where $U_k$ contains the first $k$ columns of $U$, $\Sigma_k$ is the $k \times k$ upper-left block of $\Sigma$, and $V_k$ contains the first $k$ columns of $V$.

The **Eckart-Young-Mirsky theorem** states that $A_k$ is the best rank-$k$ approximation of $A$ in both the Frobenius and spectral norms:

$$\|A - A_k\|_F = \min_{\text{rank}(B) \leq k} \|A - B\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_{\min(m,n)}^2}$$

The reconstruction error equals the root sum of squares of the discarded singular values. This gives a precise budget: retaining the top $k$ singular values captures $\sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^{\min(m,n)} \sigma_i^2$ of the total "energy" (squared Frobenius norm) of $A$.

> **Key insight:** The truncated SVD is not just *a* good low-rank approximation — it is provably the *best* one. No other rank-$k$ matrix can do better under any unitarily invariant norm.

## PCA as SVD

Principal Component Analysis is SVD applied to centered data. Given a data matrix $X \in \mathbb{R}^{n \times d}$ (rows are samples, columns are features), center it so each column has zero mean, then compute $X = U \Sigma V^T$.

The **principal components** are the right singular vectors (columns of $V$). The projections of data onto the first $k$ principal components are $X V_k = U_k \Sigma_k$. The **explained variance ratio** for the $i$-th component is:

$$\text{EVR}_i = \frac{\sigma_i^2}{\sum_{j=1}^{\min(n,d)} \sigma_j^2}$$

This tells you what fraction of the total data variance is captured by each component — the singular value spectrum of your data matrix is a fingerprint of its intrinsic dimensionality.

## LoRA: Low-Rank Adaptation

Low-Rank Adaptation (LoRA) is one of the most impactful applications of low-rank structure in modern ML. Instead of fine-tuning all parameters of a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA parameterizes the weight update as:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d,k)$.

Why does this work? Aghajanyan et al. (2021) showed that pretrained language models have a low **intrinsic dimensionality** — the weight updates during fine-tuning live in a subspace whose dimension is orders of magnitude smaller than the full parameter count. LoRA exploits this by restricting updates to a rank-$r$ manifold, typically with $r$ between 4 and 64, reducing trainable parameters by 1000x or more while matching full fine-tuning performance on many tasks.

During inference, $BA$ is merged into $W_0$ with zero additional latency. During training, only $B$ and $A$ are updated (typically with $A$ initialized from a Gaussian and $B$ initialized to zero so that $\Delta W = 0$ at the start).

## Attention as Low-Rank

The self-attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The matrix $QK^T \in \mathbb{R}^{T \times T}$ (where $T$ is the sequence length) is the product of two matrices each of rank at most $d_k$, the head dimension. So $\text{rank}(QK^T) \leq d_k$, which is typically 64 or 128 — far smaller than $T$ for long sequences.

Each attention head is a rank-$d_k$ projection of the full attention pattern. **Multi-head attention** concatenates $H$ such projections, each scanning a different $d_k$-dimensional subspace. This is an ensemble of low-rank views of the same sequence, collectively reconstructing a richer (up to rank $H \cdot d_k = d_{\text{model}}$) representation of token-token interactions.

> **Key insight:** Multi-head attention is essentially a structured low-rank decomposition of what would otherwise need to be a full-rank $T \times T$ interaction matrix. The head dimension $d_k$ is the rank budget per head.

## Stable Rank

The standard matrix rank is fragile — adding tiny noise to a rank-$k$ matrix makes it full rank. The **stable rank** provides a noise-robust alternative:

$$\text{srank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_{i} \sigma_i^2}{\sigma_1^2}$$

Stable rank is always between 1 and $\text{rank}(A)$. It equals 1 when the matrix is rank-1 (all energy in a single singular value) and equals $\text{rank}(A)$ when all nonzero singular values are equal. Stable rank measures the **effective number of significant dimensions** — it tells you how "spread out" the energy is across singular values.

In ML, stable rank is used to analyze weight matrix conditioning, measure the effective capacity of linear layers, and bound generalization error (PAC-Bayes bounds often involve stable rank).

## Randomized SVD: Scaling to Large Matrices

The exact SVD costs $O(mn^2)$ for an $m \times n$ matrix ($m \geq n$) — prohibitive when both dimensions are large. In practice, most large-scale ML systems use **randomized SVD** (Halko, Martinsson & Tropp, 2011), which computes an approximate rank-$k$ SVD in $O(mnk)$ time by:

1. Draw a random sketch matrix $\Omega \in \mathbb{R}^{n \times (k+p)}$ (e.g., Gaussian, $p$ is oversampling)
2. Form $Y = A\Omega$ to project $A$ into a low-dimensional subspace
3. Orthogonalize: $Q, \_ = \text{QR}(Y)$, so $A \approx QQ^TA$
4. Compute the small SVD of $B = Q^T A \in \mathbb{R}^{(k+p) \times n}$
5. Recover approximate singular vectors of $A$

> **Key insight:** You don't need the full SVD — you need the top-$k$ singular vectors, and a random projection finds them with high probability. `sklearn.utils.extmath.randomized_svd` and PyTorch's `torch.svd_lowrank` both implement this.

## Python: SVD in Practice

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a matrix with decaying singular value structure
np.random.seed(42)
m, n = 200, 100
# Low-rank signal + noise
U_true = np.linalg.qr(np.random.randn(m, 10))[0]
V_true = np.linalg.qr(np.random.randn(n, 10))[0]
signal = U_true @ np.diag(np.logspace(2, 0, 10)) @ V_true.T
A = signal + 0.5 * np.random.randn(m, n)

# Thin SVD (full_matrices=False returns economy-size U, S, Vt)
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruction error vs rank k
errors = [np.sqrt(np.sum(sigma[k:]**2)) for k in range(1, len(sigma))]
evr = sigma**2 / np.sum(sigma**2)  # explained variance ratios

# LoRA-style factorization: rank-r approximation of a weight update
r = 8
B = U[:, :r] * sigma[:r]  # d x r  (absorb singular values into B)
A_lora = Vt[:r, :]         # r x k
delta_W = B @ A_lora        # reconstructed rank-r update
print(f"Stable rank: {np.sum(sigma**2) / sigma[0]**2:.1f}")
print(f"Rank-{r} captures {np.sum(evr[:r])*100:.1f}% of energy")
print(f"Reconstruction error: {np.linalg.norm(A - delta_W):.3f}")
```

This script computes the full SVD, shows how reconstruction error drops with increasing rank, and demonstrates a LoRA-style rank-$r$ factorization. The stable rank tells you the effective dimensionality — here it will be close to 10, reflecting the planted low-rank signal.

:::quiz
question: "You compute the SVD of a 1000x500 data matrix and find that the top 20 singular values capture 95% of the squared Frobenius norm. What does this imply?"
options:
  - "The matrix has exact rank 20"
  - "A rank-20 approximation reconstructs 95% of the matrix energy, suggesting the data lies near a 20-dimensional subspace"
  - "The remaining 480 singular values are all zero"
  - "PCA with 20 components will perfectly reconstruct every data point"
correct: 1
explanation: "The squared Frobenius norm ratio tells us how much energy (variance) is captured. 95% captured by 20 components means the data is well-approximated by a 20-dimensional subspace, not that it's exactly rank-20 (the remaining singular values are small but nonzero) or that reconstruction is perfect."
:::

:::quiz
question: "In LoRA fine-tuning with rank r=4 on a weight matrix W of shape 4096x4096, how many trainable parameters does the low-rank update introduce?"
options:
  - "4096"
  - "16,777,216"
  - "32,768"
  - "8,192"
correct: 2
explanation: "The LoRA update is Delta W = BA where B is 4096x4 and A is 4x4096. Total trainable parameters = 4096*4 + 4*4096 = 16,384 + 16,384 = 32,768. This is 0.2% of the 16.8M parameters in the full weight matrix. (Option C = 32,768.)"
:::

:::quiz
question: "The stable rank of a matrix A is 1.0. What can you conclude about A?"
options:
  - "A is the identity matrix"
  - "A is a rank-1 matrix (or all its energy is concentrated in a single singular value)"
  - "A is orthogonal"
  - "A has condition number 1"
correct: 1
explanation: "Stable rank = sum(sigma_i^2) / sigma_1^2 = 1 means sigma_1^2 = sum(sigma_i^2), which happens only when all singular values except sigma_1 are zero — i.e., A is rank-1. An identity matrix has stable rank equal to its dimension, and an orthogonal matrix also has all singular values equal to 1."
:::
