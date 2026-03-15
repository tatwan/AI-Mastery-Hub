---
title: "Random Matrix Theory"
estimatedMinutes: 30
tags: ["random-matrix-theory", "Marchenko-Pastur", "weight-initialization", "generalization", "spectral-analysis"]
prerequisites: ["l1-svd-low-rank", "l2-eigendecomposition"]
---

# Random Matrix Theory

Neural networks begin their lives as collections of random matrices. The spectral properties of these random initializations determine whether gradients flow or vanish, how much capacity the network has before seeing any data, and — remarkably — how well the trained model will generalize. Random matrix theory (RMT) provides the mathematical framework for understanding these phenomena, offering precise predictions about eigenvalue distributions that serve as baselines for analyzing learned representations.

## Why Random Matrices Matter for ML

At initialization, a neural network's weight matrices are drawn from specified distributions (typically Gaussian). Their singular value spectra control the Jacobian of the network, which in turn determines gradient magnitudes through the chain rule. If singular values are too large, gradients explode; too small, they vanish. The goal of initialization schemes like Kaiming/He is to place the spectrum in a "just right" regime — and RMT tells us exactly what that spectrum looks like.

After training, comparing a weight matrix's spectrum to the random baseline reveals what the network has learned. Directions where singular values deviate from the random bulk correspond to learned features; directions that remain in the bulk are effectively noise.

> **Key insight:** RMT provides a null hypothesis for weight matrices. The Marchenko-Pastur distribution tells you what a matrix that has learned *nothing* looks like. Deviations from this baseline are the signature of learning.

## The Wigner Semicircle Law

> **Intuition:** The Wigner semicircle law reveals surprising order in randomness. If you fill a large symmetric matrix with independent random entries, the histogram of eigenvalues doesn't look random at all — it converges to a clean semicircular shape with sharp edges at $\pm 2\sigma$. No eigenvalue escapes this bulk. The semicircle is not specific to Gaussian entries; any distribution with finite fourth moment gives the same shape. This universality is what makes the law useful as a baseline: it tells you what "pure noise" looks like spectrally.

The simplest RMT result concerns symmetric random matrices. Let $M = \frac{1}{\sqrt{n}}(A + A^T)/2$ where $A$ has i.i.d. $\mathcal{N}(0, 1)$ entries. As $n \to \infty$, the empirical spectral distribution (the histogram of eigenvalues) converges to the **semicircle distribution**:

$$\rho(\lambda) = \frac{1}{2\pi\sigma^2}\sqrt{4\sigma^2 - \lambda^2}, \quad |\lambda| \leq 2\sigma$$

where $\sigma^2$ is the variance of the entries.

In words: the eigenvalues spread out into a smooth semicircular shape, with sharp edges at $\pm 2\sigma$. No eigenvalue escapes this bulk (with probability 1 in the limit). This universality result holds not just for Gaussian entries but for any distribution with finite fourth moment — the shape of the distribution is determined only by the variance.

The semicircle law applies to symmetric matrices, which appear as Hessians, covariance matrices at initialization, and Gram matrices. It provides the baseline spectrum against which to measure the effect of training on these quantities.

## The Marchenko-Pastur Law

> **Refresher:** The Marchenko-Pastur distribution is the null distribution for PCA on random data. In PCA, you compute the covariance matrix $X^T X / n$ and look for large eigenvalues as evidence of structure. But even for a completely random data matrix, the eigenvalues are not all equal — they spread out. The MP law tells you exactly how they spread as a function of the aspect ratio $\gamma = m/n$. Any eigenvalue above $\lambda_+$ is a genuine signal; anything inside the bulk $[\lambda_-, \lambda_+]$ is indistinguishable from noise.

For rectangular matrices — the case directly relevant to weight matrices — the Marchenko-Pastur (MP) law is the fundamental result.

Let $X \in \mathbb{R}^{m \times n}$ have i.i.d. entries $X_{ij} \sim \mathcal{N}(0, \sigma^2/n)$. Consider the sample covariance-like matrix $S = X X^T / m$. As $m, n \to \infty$ with aspect ratio $\gamma = m/n$ held constant, the eigenvalue distribution of $S$ converges to:

$$\rho(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\gamma\sigma^2\lambda}, \quad \lambda_- \leq \lambda \leq \lambda_+$$

with edges:

$$\lambda_{\pm} = \sigma^2(1 \pm \sqrt{\gamma})^2$$

When $\gamma < 1$ (more columns than rows), there is also a point mass at $\lambda = 0$ with weight $1 - \gamma$.

The MP distribution has several properties that matter for ML:

- The **bulk** is the region $[\lambda_-, \lambda_+]$. Eigenvalues of a purely random matrix stay within this interval.
- The **ratio** $\lambda_+/\lambda_- = ((1 + \sqrt{\gamma})/(1 - \sqrt{\gamma}))^2$ determines the condition number of the random matrix.
- The **mean** eigenvalue is $\sigma^2$, independent of $\gamma$.

## Reading a Spectrum: Random vs. Learned

Given a trained weight matrix $W$, compute its singular values and form the eigenvalue distribution of $WW^T$. Compare this to the MP distribution predicted by the matrix dimensions and entry variance:

- **Eigenvalues inside the MP bulk**: these directions are indistinguishable from random — the network has not learned useful features along them.
- **Eigenvalues exceeding $\lambda_+$ (outliers)**: these are **signal eigenvalues** — directions along which the network has learned structured representations. The number of outliers roughly indicates the effective rank of the learned component.
- **Eigenvalues below $\lambda_-$**: these can indicate regularization effects or structured suppression of certain directions.

Martin and Mahoney (2021) developed this into a systematic diagnostic: by fitting the tail of the eigenvalue distribution to a power law $\rho(\lambda) \propto \lambda^{-\alpha}$, one can assess the quality of training. Well-trained layers show $\alpha \approx 2$–$4$ (heavy-tailed but structured), while poorly trained or overregularized layers show spectra close to the MP bulk.

> **Key insight:** The MP distribution is the spectral fingerprint of randomness. Outlier eigenvalues above the MP edge are the mathematical signature of what a neural network has learned — each one represents a direction in weight space that has moved away from its random initialization in a structured way.

## Free Probability: The Mathematics of Random Matrix Sums

> **Intuition:** Free probability is probability theory for non-commuting matrices. In classical probability, knowing the distributions of independent variables $X$ and $Y$ tells you the distribution of $X + Y$. For matrices, independence is replaced by "freeness" — a condition roughly meaning the eigenbases of $A$ and $B$ are in generic position relative to each other. When $A$ and $B$ are free, the spectral distribution of $A + B$ is determined by those of $A$ and $B$ alone, via free convolution. This is the tool that lets you predict how spectra compose across the layers of a deep network.

Classical probability studies sums of independent scalar random variables (central limit theorem). **Free probability theory** (Voiculescu, 1991) is the analogous framework for sums and products of large random matrices, where "freeness" replaces classical independence.

The key result is **free convolution**: if $A$ and $B$ are large random matrices that are "free" (roughly: drawn from rotationally invariant ensembles), the empirical spectral distribution of $A + B$ is the **free additive convolution** $\mu_A \boxplus \mu_B$, computable via $R$-transforms. Similarly for products via the $S$-transform.

Why this matters for deep learning:
- **Residual networks**: the spectrum of $W + I$ (residual connection) is not simply a shift of $W$'s spectrum — it's a free convolution. Free probability explains why residual connections preserve spectral bulk near 1.
- **Gram matrices of trained networks**: the spectrum of $\frac{1}{n}X^TX$ for deep representations follows free convolution of the individual layer spectra.
- Martin & Mahoney (2021) use free probability to explain power-law tails in trained weight matrices as evidence of "heavy-tailed self-regularization."

> **Key insight:** Free probability is the right language for understanding how spectra compose across layers. Classical addition $\mu_{A+B} \neq \mu_A * \mu_B$ for matrices, but free convolution $\mu_{A+B} = \mu_A \boxplus \mu_B$ holds when $A$ and $B$ are free.

## Weight Initialization: Kaiming/He

> **Remember:** Kaiming initialization sets $\text{Var}(W_{ij}) = 2/n_{\text{in}}$ for ReLU networks. The goal is to keep activation variance constant across layers: without the factor of 2, each ReLU layer would halve the variance (since ReLU zeroes out roughly half its inputs), and after $L$ layers the signal would shrink by $(1/2)^L$. The factor of 2 exactly cancels this attenuation.

The Kaiming (He) initialization sets:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

where $n_{\text{in}}$ is the fan-in (number of input units). The factor of 2 accounts for ReLU activations (which zero out roughly half the units).

The derivation follows from variance propagation. For a layer $y = Wx$ followed by ReLU, assuming $x$ has components with variance $\text{Var}(x_i)$:

$$\text{Var}(y_j) = n_{\text{in}} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_i) \cdot \frac{1}{2}$$

The factor $\frac{1}{2}$ comes from ReLU zeroing out negative values — this assumes inputs are symmetrically distributed around zero at initialization, so ReLU zeroes approximately half. The approximation holds at initialization but drifts during training as the distribution of pre-activations shifts. Setting $\text{Var}(W_{ij}) = 2/n_{\text{in}}$ gives $\text{Var}(y_j) = \text{Var}(x_i)$ — variance is preserved across layers.

From the RMT perspective, this initialization places the singular values of each weight matrix near 1, ensuring that the product of Jacobians through many layers neither grows nor shrinks. The MP distribution for this initialization has bulk edges at $\sigma^2(1 \pm \sqrt{\gamma})^2$ with $\sigma^2 = 2/n_{\text{in}}$. For a square layer ($\gamma = 1$), the bulk spans $[0, 4\sigma^2]$, and the mean singular value squared is $\sigma^2 \cdot n_{\text{in}} = 2$ — giving a mean singular value of approximately $\sqrt{2}$, which compensates for the $1/\sqrt{2}$ factor from ReLU. (Strictly, Jensen's inequality gives $\mathbb{E}[\sigma_i] \leq \sqrt{\mathbb{E}[\sigma_i^2]}$, so $\sqrt{2}$ is an upper bound on the mean singular value, not the exact value. For practical purposes of initialization the approximation is sufficient.)

## Spectral Norm and Lipschitz Constants

The spectral norm $\|W\|_2 = \sigma_{\max}(W)$ is the largest singular value — the maximum factor by which $W$ can stretch a vector. For a linear map $y = Wx$:

$$\frac{\|y\|}{\|x\|} \leq \|W\|_2$$

with equality achieved by the top right singular vector. This means $\|W\|_2$ is the **Lipschitz constant** of the linear map.

For a composition of layers $f = f_L \circ \cdots \circ f_1$, the global Lipschitz constant is bounded by the product of per-layer Lipschitz constants. If each layer has spectral norm $s_l$, then $\text{Lip}(f) \leq \prod_l s_l$.

**Spectral normalization** (Miyato et al., 2018) divides each weight matrix by its spectral norm at every training step:

$$\bar{W} = \frac{W}{\|W\|_2}$$

This ensures each layer is 1-Lipschitz, making the entire network $L$-Lipschitz (where $L$ is the number of layers, accounting for activation functions). Spectral normalization was introduced for stabilizing GAN training — it prevents the discriminator from being too sensitive to small input perturbations, which causes mode collapse.

## Double Descent and the Interpolation Threshold

> **Intuition:** Double descent breaks the classical U-shaped bias-variance tradeoff. In the classical picture, adding model capacity past the interpolation threshold (where parameters = data points) causes overfitting. But for modern overparameterized models, test error decreases again beyond this threshold — forming a second descent. The key: when a model has far more capacity than needed, it can interpolate the training data while still generalizing, because the excess capacity allows it to find a smooth, low-norm solution. The classical curve was correct for its regime; it just doesn't extend to the overparameterized regime that deep learning operates in.

One of the most striking predictions of RMT concerns generalization. Classical learning theory says test error follows a U-shaped curve as model complexity increases: first decreasing (underfitting to good fit) then increasing (overfitting). But modern over-parameterized models violate this — test error decreases *again* after the interpolation threshold (where the number of parameters equals the number of data points).

RMT explains this through the spectral structure of the kernel matrix $K = X X^T / n$. At the interpolation threshold, the smallest eigenvalue of $K$ approaches zero, causing the condition number to diverge. The model is forced to use very large weights to interpolate, amplifying noise. Past this threshold, the model has excess capacity to interpolate *smoothly*, keeping weights small via implicit regularization.

The MP distribution predicts exactly where this threshold occurs: at $\gamma = m/n = 1$, where $\lambda_- \to 0$. The peak in test error at $\gamma = 1$ is predicted by the divergence of $\text{tr}(K^{-1})$, which governs the variance term in the bias-variance decomposition.

> **Key insight:** Double descent is not a mystery — it is a direct consequence of the spectral properties of random matrices at the interpolation threshold. RMT predicts the exact location and severity of the peak, and explains why overparameterized models generalize well: they interpolate with spectrally regular solutions.

## Python: Marchenko-Pastur in Practice

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m, n = 500, 1000  # gamma = m/n = 0.5
sigma2 = 1.0 / n
X = np.random.randn(m, n) * np.sqrt(sigma2)
S = X @ X.T  # m x m (unnormalized by m; MP edges below are scaled accordingly)
eigs = np.linalg.eigvalsh(S)

# Theoretical MP distribution
gamma = m / n
lam_plus = sigma2 * n * (1 + np.sqrt(gamma))**2 / m  # = (1+sqrt(gamma))^2
lam_minus = sigma2 * n * (1 - np.sqrt(gamma))**2 / m
# Simplified: for S = XX^T with Var(X_ij) = sigma2
# eigenvalues of S/m have MP with sigma2*n/m scaling
lam_grid = np.linspace(max(lam_minus * 0.9, 0), lam_plus * 1.1, 500)
mp_density = np.sqrt(np.maximum((lam_plus - lam_grid) * (lam_grid - lam_minus), 0))
mp_density /= (2 * np.pi * gamma * (sigma2 * n / m) * lam_grid + 1e-10)

# Compare: simulate a neural network layer before and after "training"
W_init = np.random.randn(256, 512) * np.sqrt(2.0 / 512)  # Kaiming init
# Simulate training: add low-rank learned component
U_learn = np.random.randn(256, 5)
V_learn = np.random.randn(5, 512)
W_trained = W_init + 0.5 * U_learn @ V_learn  # rank-5 signal added

svals_init = np.linalg.svd(W_init, compute_uv=False)
svals_trained = np.linalg.svd(W_trained, compute_uv=False)

print(f"MP bulk edges: [{lam_minus:.4f}, {lam_plus:.4f}]")
print(f"Init: max sval^2 = {svals_init[0]**2:.3f}, expected MP edge ~ {2*(1+np.sqrt(0.5))**2/512*512:.3f}")
print(f"Trained: top 5 svals = {svals_trained[:5].round(3)}")
print(f"Trained: svals 6-10  = {svals_trained[5:10].round(3)}")
print(f"Outliers above init max: {np.sum(svals_trained > svals_init[0])}")
```

This code generates a random matrix, computes its eigenvalue distribution, and compares to the MP prediction. It then simulates a neural network layer before and after training (with a rank-5 learned signal), showing how trained singular values develop outliers above the random bulk.

:::quiz
question: "A 512x512 weight matrix is initialized with Kaiming initialization (variance 2/512). After training, you compute its singular values and find that 8 singular values exceed the Marchenko-Pastur upper edge while the rest remain in the bulk. What does this tell you?"
options:
  - "The network has diverged and needs a lower learning rate"
  - "The layer has learned an approximately rank-8 structured component on top of the random initialization"
  - "The initialization variance was too high"
  - "The layer has memorized exactly 8 training examples"
correct: 1
explanation: "Singular values above the MP edge are signal eigenvalues — directions along which the weight matrix has moved away from its random initialization in a structured way. Eight outliers suggest the learned component is approximately rank-8. The bulk eigenvalues remaining in the MP distribution indicates those directions are still effectively random."
:::

:::quiz
question: "The Marchenko-Pastur distribution for a 100x1000 matrix (gamma = 0.1) has bulk edges at lambda_- = sigma^2(1-sqrt(0.1))^2 and lambda_+ = sigma^2(1+sqrt(0.1))^2. What happens to the condition number of this bulk as gamma approaches 1?"
options:
  - "It decreases, making the matrix better conditioned"
  - "It stays constant regardless of gamma"
  - "It diverges because lambda_- approaches 0"
  - "It approaches 1 because both edges converge"
correct: 2
explanation: "As gamma -> 1, lambda_- = sigma^2(1-sqrt(gamma))^2 -> 0 while lambda_+ = sigma^2(1+sqrt(gamma))^2 -> 4*sigma^2. The condition number lambda_+/lambda_- diverges. This is exactly the interpolation threshold where double descent occurs — the minimum eigenvalue vanishing causes numerical instability and the peak in test error."
:::

:::quiz
question: "Why does spectral normalization (dividing W by its spectral norm) stabilize GAN training?"
options:
  - "It makes the weight matrix orthogonal"
  - "It constrains each layer to be 1-Lipschitz, preventing the discriminator from being overly sensitive to small input changes"
  - "It ensures the weight matrix has full rank"
  - "It minimizes the Frobenius norm of the weight matrix"
correct: 1
explanation: "Spectral normalization divides W by sigma_max(W), ensuring the largest singular value is 1. This makes the linear map 1-Lipschitz: ||Wx|| <= ||x|| for all x. For the discriminator, this prevents it from having very steep gradients (high Lipschitz constant), which would cause the generator to receive uninformative gradient signals and collapse to a single mode."
:::
