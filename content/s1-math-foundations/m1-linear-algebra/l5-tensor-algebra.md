---
title: "Tensor Algebra & Decompositions"
estimatedMinutes: 30
tags: ["tensors", "Tucker-decomposition", "CP-decomposition", "multi-head-attention", "tensor-networks"]
prerequisites: ["l1-svd-low-rank"]
---

# Tensor Algebra & Decompositions

Matrices capture bilinear relationships — interactions between two indices. But the structures in modern ML are fundamentally higher-order: a convolutional filter has spatial, input-channel, and output-channel dimensions; multi-head attention has head, query, and key dimensions; a batch of feature maps has batch, channel, height, and width dimensions. Tensors generalize matrices to arbitrarily many indices, and tensor decompositions generalize SVD to extract low-rank structure from these multi-dimensional arrays. This lesson develops the algebraic foundations and connects them to the architectures where tensors naturally arise.

## Tensors as Multi-Dimensional Arrays

> **Refresher:** Tensors generalize scalars, vectors, and matrices along a single axis of order. A scalar is order-0 (no indices), a vector is order-1 (one index), a matrix is order-2 (two indices: row and column), and a 3rd-order tensor has three indices. Each additional order adds a new "dimension" of interaction. A convolutional weight kernel, for example, is naturally order-4 (output channels × input channels × height × width) — it cannot be fully captured by any single matrix without losing structural information.

A **tensor** $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is an $N$-dimensional array. The integer $N$ is the **order** (or mode) of the tensor. Scalars are order-0, vectors are order-1, matrices are order-2, and everything beyond is where tensor algebra becomes essential.

The **fibers** of a tensor are the higher-order analogues of matrix rows and columns. For a 3rd-order tensor $\mathcal{T} \in \mathbb{R}^{I \times J \times K}$:

- **Mode-1 fibers** $\mathcal{T}_{:jk} \in \mathbb{R}^I$ — fix two indices, vary the first (analogous to columns)
- **Mode-2 fibers** $\mathcal{T}_{i:k} \in \mathbb{R}^J$ — fix first and third, vary the second (analogous to rows)
- **Mode-3 fibers** $\mathcal{T}_{ij:} \in \mathbb{R}^K$ — fix first two, vary the third

**Slices** fix one index: $\mathcal{T}_{::k}$ is a matrix slice along mode-3.

The **mode-$n$ product** of a tensor $\mathcal{T}$ with a matrix $U \in \mathbb{R}^{J \times I_n}$ is:

$$(\mathcal{T} \times_n U)_{i_1 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n=1}^{I_n} \mathcal{T}_{i_1 \cdots i_N} U_{j,i_n}$$

This multiplies each mode-$n$ fiber by $U$, transforming the $n$-th dimension from $I_n$ to $J$. Mode products are the building blocks of tensor decompositions.

> **Key insight:** Tensors are not just "matrices with more dimensions." The interactions between modes create structure that cannot be captured by reshaping into a matrix. Tensor decompositions exploit this multi-modal structure in ways that matrix methods cannot.

## Mode-$n$ Unfolding (Matricization)

> **Intuition:** Mode-$n$ unfolding reshapes a tensor into a matrix by "unrolling" all dimensions except the $n$-th into a single column index. Concretely: pick one mode to be the rows; concatenate all the tensor's slices along that mode to fill the columns. This is the bridge that lets you apply matrix algorithms (SVD, rank computations) to tensors. The mode-$n$ rank of the tensor is just the rank of this unfolded matrix — different modes can have different ranks, giving the tensor its richer "multi-rank" structure.

The **mode-$n$ unfolding** rearranges a tensor into a matrix by mapping mode $n$ to the rows and all other modes to the columns:

$$T_{(n)} \in \mathbb{R}^{I_n \times (I_1 \cdots I_{n-1} I_{n+1} \cdots I_N)}$$

For a 3rd-order tensor $\mathcal{T} \in \mathbb{R}^{3 \times 4 \times 5}$:
- $T_{(1)} \in \mathbb{R}^{3 \times 20}$ — each row is a mode-1 fiber, concatenated across all combinations of the other indices
- $T_{(2)} \in \mathbb{R}^{4 \times 15}$
- $T_{(3)} \in \mathbb{R}^{5 \times 12}$

Unfolding is the bridge between tensor and matrix operations. The **$n$-rank** of $\mathcal{T}$ is $\text{rank}(T_{(n)})$ — the rank of the mode-$n$ unfolding. Unlike matrix rank, a tensor has a different rank along each mode, and these mode-ranks collectively constrain the tensor's structure.

The key identity connecting mode products and unfoldings is:

$$Y = \mathcal{T} \times_n U \quad \Leftrightarrow \quad Y_{(n)} = U \, T_{(n)}$$

This means mode-$n$ multiplication is just matrix multiplication on the unfolded tensor — making it computationally straightforward.

## Tucker Decomposition

> **Intuition:** Tucker decomposition is SVD generalized to tensors. In matrix SVD, you find two orthogonal bases (one per mode) and a diagonal "core" matrix of singular values. In Tucker, you find one orthogonal factor matrix per mode and a dense core tensor that captures all the mode-to-mode interactions. The factor matrices compress each mode independently; the core encodes the structure of how compressed modes interact. Think of it as rotating into a compact coordinate system along every mode simultaneously.

The **Tucker decomposition** expresses a tensor as a core tensor multiplied by factor matrices along each mode:

$$\mathcal{T} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

where $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ is the **core tensor** and $U^{(n)} \in \mathbb{R}^{I_n \times R_n}$ are the **factor matrices** (typically with orthonormal columns). The tuple $(R_1, R_2, R_3)$ is the **multilinear rank**.

The Tucker decomposition is the natural generalization of truncated SVD to tensors. Just as truncated SVD finds the best rank-$k$ matrix approximation, Tucker finds the best multilinear-rank-$(R_1, R_2, R_3)$ approximation.

The **Higher-Order SVD (HOSVD)** computes the Tucker decomposition by performing SVD on each mode-$n$ unfolding independently:

1. Compute $T_{(n)} = U^{(n)} S^{(n)} V^{(n)T}$ for each mode $n$
2. Truncate $U^{(n)}$ to the first $R_n$ columns
3. Compute the core: $\mathcal{G} = \mathcal{T} \times_1 U^{(1)T} \times_2 U^{(2)T} \times_3 U^{(3)T}$

Unlike matrix SVD, HOSVD does not give the globally optimal Tucker approximation (it is only optimal per-mode). The **Higher-Order Orthogonal Iteration (HOOI)** algorithm iteratively refines the factor matrices to improve the approximation, but convergence to the global optimum is not guaranteed.

> **Key insight:** Tucker decomposition compresses a tensor along every mode simultaneously. A tensor of size $100 \times 100 \times 100$ with multilinear rank $(5, 5, 5)$ is stored with $5^3 + 3 \times 100 \times 5 = 1625$ parameters instead of $10^6$ — a compression ratio of over 600x.

## CP Decomposition

The **CP decomposition** (CANDECOMP/PARAFAC) expresses a tensor as a sum of rank-1 tensors:

$$\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r \, a_r \otimes b_r \otimes c_r$$

where $\otimes$ denotes the outer product: $(a \otimes b \otimes c)_{ijk} = a_i b_j c_k$. Each term is a rank-1 tensor, and $R$ is the **CP rank**.

The CP decomposition is more constrained than Tucker: the core tensor $\mathcal{G}$ is restricted to be superdiagonal (only $\mathcal{G}_{rrr}$ entries are nonzero). This means CP decomposes the tensor into independent components, each acting along a single direction in every mode — analogous to the outer-product form of SVD ($A = \sum_i \sigma_i u_i v_i^T$).

**Critical difference from matrix rank:** computing the CP rank of a tensor is NP-hard in general. There is no analogue of the Eckart-Young theorem — the best rank-$R$ CP approximation may not exist (the infimum may not be achieved), and algorithms like Alternating Least Squares (ALS) provide only local optima.

In element-wise form:

$$\mathcal{T}_{ijk} \approx \sum_{r=1}^{R} \lambda_r \, a_{ir} \, b_{jr} \, c_{kr}$$

This is useful for interpretation: each component $r$ contributes a multiplicative interaction between the $r$-th columns of the factor matrices $A$, $B$, and $C$.

## Multi-Head Attention as Tensor Operation

> **Refresher:** Recall from l1-svd that the attention matrix $QK^T \in \mathbb{R}^{T \times T}$ has rank at most $d_k$, the head dimension. Each attention head is therefore a rank-$d_k$ approximation of the full token-to-token interaction pattern, where the $Q$ and $K$ projections define the subspace. Stacking $H$ heads into a tensor $\mathcal{W}_Q \in \mathbb{R}^{d \times d_k \times H}$ makes this multi-rank structure explicit: the full attention mechanism is a collection of $H$ rank-$d_k$ views of the same sequence, computed simultaneously via tensor contraction.

In a transformer with $H$ attention heads and model dimension $d$, each head $h$ has projection matrices $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_k}$ where $d_k = d/H$. These can be stacked into 3rd-order tensors:

$$\mathcal{W}_Q \in \mathbb{R}^{d \times d_k \times H}$$

where slice $\mathcal{W}_{Q}[:,:,h] = W_Q^{(h)}$.

The multi-head attention computation for all heads simultaneously becomes:

```
Q_all = einsum('btd,dkh->btkh', X, W_Q)   # all queries
K_all = einsum('bsd,dkh->bskh', X, W_K)   # all keys
A = einsum('btkh,bskh->btsh', Q_all, K_all) / sqrt(d_k)  # all attention logits
```

This batched tensor contraction computes all $H$ attention patterns in a single fused operation. The tensor perspective reveals that multi-head attention is a **mode-specific low-rank factorization**: the full $T \times T$ attention matrix is decomposed along the head mode, with each head providing a rank-$d_k$ slice.

Efficient implementations (FlashAttention, xFormers) exploit this tensor structure to tile the computation across GPU memory hierarchies, processing attention in blocks that fit in SRAM rather than materializing the full $T \times T \times H$ attention tensor in HBM.

## Tensor Networks

> **Intuition:** The tensor train factorizes an exponentially large tensor as a chain of small 3D tensors (the "cores"), connected by shared indices. Reading the TT formula left to right, each core $G^{(k)}$ is a $r_{k-1} \times I_k \times r_k$ array; the shared indices $r_k$ are the TT-ranks. The product of all these small matrices (one matrix per value of the physical index $i_k$) yields a single scalar — the $(i_1, i_2, \ldots, i_N)$ entry of the original tensor. Expressiveness grows with the TT-ranks, but storage only grows linearly in $N$. This is the tensor analogue of why an $n$-qubit quantum state can sometimes be represented efficiently: when correlations between distant indices are limited, the TT-ranks stay small.

Tensor networks generalize tensor decompositions by arranging tensors into graphs, where edges represent contracted indices. The most important for ML is the **tensor train** (TT) decomposition, also known as **matrix product states** (MPS) in physics:

$$\mathcal{T}_{i_1 i_2 \cdots i_N} = G^{(1)}_{i_1} G^{(2)}_{i_2} \cdots G^{(N)}_{i_N}$$

where each $G^{(k)}_{i_k} \in \mathbb{R}^{r_{k-1} \times r_k}$ is a matrix (with $r_0 = r_N = 1$ so the product yields a scalar). The integers $r_k$ are the **TT-ranks** and control the expressiveness of the decomposition.

A tensor of size $I^N$ has $I^N$ entries — exponential in $N$. The TT decomposition stores this with $O(N I r^2)$ parameters where $r = \max_k r_k$, which is *linear* in $N$. This exponential compression is why tensor networks are indispensable for:

- **LLM weight compression**: weight matrices reshaped into high-order tensors can be compressed via TT decomposition, achieving 10-50x compression with minimal accuracy loss.
- **Quantum-inspired ML**: variational quantum circuits can be simulated efficiently as tensor networks when entanglement (TT-rank) is low.
- **Probabilistic models**: Born machines use tensor networks to represent high-dimensional probability distributions with tractable normalization.

> **Key insight:** Tensor networks turn the curse of dimensionality into a blessing of structure. By decomposing a high-order tensor into a chain of low-order cores, they represent exponentially large objects with linearly many parameters — as long as the correlations between distant modes are bounded (low TT-rank).

## Why Tensors Matter for ML

The deep structures of modern ML are inherently tensorial:

1. **Weight tensors**: A convolutional layer's kernel is a 4th-order tensor $\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times H \times W}$. Tensor decomposition of these kernels is a proven approach to model compression.

2. **Multi-modal representations**: In vision-language models, the interaction between visual features ($v$), textual features ($t$), and positional information ($p$) is naturally a 3rd-order tensor. Tucker decomposition can factorize this interaction efficiently.

3. **Multi-task learning**: With $T$ tasks, $D$ input dimensions, and $K$ output dimensions, the full weight array is a 3rd-order tensor. CP decomposition reveals shared structure across tasks (shared factor vectors) and task-specific structure (individual components).

4. **Attention patterns**: Across layers ($L$), heads ($H$), and positions ($T \times T$), the full attention pattern of a transformer is a high-order tensor. Analyzing its decomposition reveals redundancy across heads and layers.

## Python: Tensor Decompositions

```python
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac

# Create a 3rd-order tensor with known low-rank structure
np.random.seed(42)
shape = (50, 40, 30)
rank_cp = 5

# Ground truth: sum of rank-1 tensors + noise
factors_true = [np.random.randn(s, rank_cp) for s in shape]
T_clean = tl.cp_to_tensor((np.ones(rank_cp), factors_true))
T_noisy = T_clean + 0.1 * np.random.randn(*shape)

# CP decomposition
cp_result = parafac(T_noisy, rank=rank_cp)
T_cp = tl.cp_to_tensor(cp_result)
print(f"CP rank-{rank_cp} error: {tl.norm(T_noisy - T_cp) / tl.norm(T_noisy):.4f}")

# Tucker decomposition with multilinear rank (5, 5, 5)
core, factors = tucker(T_noisy, rank=[5, 5, 5])
T_tucker = tl.tucker_to_tensor((core, factors))
print(f"Tucker (5,5,5) error: {tl.norm(T_noisy - T_tucker) / tl.norm(T_noisy):.4f}")

# Multi-head attention as batched tensor contraction
import torch
B, T_len, d, H = 2, 10, 64, 8
d_k = d // H
X = torch.randn(B, T_len, d)
W_Q = torch.randn(d, d_k, H)  # stacked query projections

Q = torch.einsum('btd,dkh->btkh', X, W_Q)  # all heads at once
print(f"\nQ shape: {Q.shape}  (batch, seq, head_dim, heads)")
print(f"Params in stacked tensor: {W_Q.numel()}")
print(f"Params in H separate matrices: {H * d * d_k}")  # same count
```

This script demonstrates CP and Tucker decompositions on a synthetic tensor with known rank structure, then shows how multi-head attention projections are naturally expressed as tensor contractions via `einsum`.

:::quiz
question: "A 4th-order tensor of shape (100, 100, 100, 100) requires 10^8 entries to store. With a Tucker decomposition of multilinear rank (5, 5, 5, 5), approximately how many parameters are needed?"
options:
  - "625 (just the core tensor)"
  - "2,625 (core tensor + factor matrices: 5^4 + 4*100*5)"
  - "500,000 (not much compression)"
  - "10,000 (just the factor matrices)"
correct: 1
explanation: "The core tensor has 5^4 = 625 entries. Each of the 4 factor matrices has 100 x 5 = 500 entries, totaling 4 x 500 = 2000. Total = 625 + 2000 = 2625 parameters, compared to 10^8 — a compression ratio of about 38,000x. This dramatic compression is why Tucker decomposition is so effective for model compression."
:::

:::quiz
question: "What is the fundamental difference between computing the rank of a matrix and computing the CP rank of a tensor?"
options:
  - "Matrix rank is always 1, while tensor rank can be any positive integer"
  - "Matrix rank can be computed in polynomial time (e.g., via SVD), while computing CP rank is NP-hard"
  - "They are equivalent — CP rank reduces to matrix rank for 2nd-order tensors"
  - "Matrix rank is defined over real numbers while CP rank requires complex numbers"
correct: 1
explanation: "Matrix rank is computable in O(mn*min(m,n)) via SVD. But for tensors of order 3 and above, determining the minimum number of rank-1 terms needed (the CP rank) is NP-hard. There is also no Eckart-Young theorem for tensors — the best rank-R CP approximation may not even exist. This fundamental computational gap between order-2 and order-3+ is one of the deepest results in tensor algebra."
:::

:::quiz
question: "In the tensor train decomposition T_{i1...iN} = G^(1)_{i1} G^(2)_{i2} ... G^(N)_{iN}, what role do the TT-ranks r_k play?"
options:
  - "They determine the physical dimensions of the original tensor"
  - "They control the amount of correlation captured between modes k and k+1, analogous to the number of singular values retained in a truncated SVD"
  - "They must all be equal for the decomposition to be valid"
  - "They represent the number of training iterations needed for convergence"
correct: 1
explanation: "The TT-rank r_k controls the expressiveness of the bond between cores G^(k) and G^(k+1). If r_k = 1, modes 1..k and k+1..N are independent (a product state). Larger r_k captures more correlations across that partition — exactly analogous to retaining more singular values in SVD. The TT decomposition applies SVD sequentially across mode partitions, and r_k is the number of singular values retained at step k."
:::
