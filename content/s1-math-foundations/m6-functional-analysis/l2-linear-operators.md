---
title: "Linear Operators & the Spectral Theorem"
estimatedMinutes: 35
tags: ["linear-operators", "spectral-theorem", "compact-operators", "adjoint", "eigenfunction-expansion"]
prerequisites: ["l1-banach-hilbert-spaces", "l2-eigendecomposition from m1"]
---

# Linear Operators & the Spectral Theorem

In finite dimensions, a linear map between vector spaces is completely described by a matrix. In infinite dimensions — function spaces, sequence spaces, the $L^2$ settings of signal processing and machine learning — the analogue is a **linear operator**, and the passage from finite to infinite dimension is not merely a change in scale. New phenomena appear: operators can be bounded or unbounded, compact or not, and the "matrix" may not even have a finite representation. The payoff is a spectral theorem that guarantees orthonormal bases of eigenfunctions for the right class of operators, directly generalizing finite-dimensional diagonalization to the function spaces where modern ML operates.

## Bounded Linear Operators

> **Refresher:** In L1 we constructed Hilbert spaces $H$ — complete inner product spaces equipped with a norm $\|x\| = \sqrt{\langle x, x \rangle}$. The key examples are $\mathbb{R}^n$ (finite-dimensional), $\ell^2$ (square-summable sequences), and $L^2(\Omega)$ (square-integrable functions on a domain $\Omega$). A linear operator $T: H \to H$ is just a map that respects addition and scalar multiplication: $T(\alpha x + \beta y) = \alpha Tx + \beta Ty$.

A linear operator $T: H \to H$ is **bounded** if there exists $C < \infty$ such that

$$\|Tx\| \leq C \|x\| \quad \text{for all } x \in H.$$

The smallest such $C$ is the **operator norm**:

$$\|T\| = \sup_{\|x\| = 1} \|Tx\| = \sup_{x \neq 0} \frac{\|Tx\|}{\|x\|}.$$

> **Key insight:** Boundedness is the correct infinite-dimensional analogue of "the matrix has finite entries." An unbounded operator can send unit vectors to arbitrarily large outputs — it is "infinitely large" in some direction, which destroys continuity.

**Bounded $\Leftrightarrow$ Continuous.** This equivalence is fundamental: $T$ is bounded if and only if it is continuous at every point, equivalently, if and only if it is continuous at $0$. The proof is short: if $T$ is bounded, then $\|Tx_n - Tx\| = \|T(x_n - x)\| \leq \|T\| \cdot \|x_n - x\| \to 0$, so $T$ is Lipschitz continuous. Conversely, if $T$ is continuous at $0$ but not bounded, one can construct unit vectors $x_n$ with $\|Tx_n\| \to \infty$, contradicting continuity.

**Examples.**

- **Multiplication operators.** On $L^2([0,1])$, fix $m \in L^\infty([0,1])$ and define $(M_m f)(x) = m(x) f(x)$. Then $\|M_m\| = \|m\|_{L^\infty}$, which is finite. The eigenvalue equation $M_m f = \lambda f$ becomes $m(x)f(x) = \lambda f(x)$, so $f$ must be supported on the level set $\{m = \lambda\}$ — in general, there are no $L^2$ eigenfunctions, only a **continuous spectrum**.

The **spectrum** $\sigma(T)$ of a bounded operator $T$ is $\{\lambda : T - \lambda I \text{ is not invertible}\}$, decomposed into three parts:
- **Point spectrum** $\sigma_p(T)$: $\lambda$ is an eigenvalue ($\ker(T - \lambda I) \neq 0$).
- **Continuous spectrum** $\sigma_c(T)$: $T - \lambda I$ is injective with dense range but is not surjective.
- **Residual spectrum** $\sigma_r(T)$: $T - \lambda I$ is injective but does not have dense range.

For compact self-adjoint operators, $\sigma(T) = \sigma_p(T) \cup \{0\}$ — the continuous and residual spectra are empty except possibly at $\lambda = 0$. This is precisely why the spectral theorem for compact operators is so clean: every nonzero spectral value is an eigenvalue with a genuine eigenvector, unlike the multiplication operator where most spectral values have no eigenvector.

- **Integral operators.** Define $(Tf)(x) = \int_\Omega K(x,y) f(y) \, dy$ for a kernel $K: \Omega \times \Omega \to \mathbb{R}$. Under mild conditions on $K$ (e.g., $K \in L^2(\Omega \times \Omega)$), $T$ is bounded on $L^2(\Omega)$. The kernel $K(x,y) = \min(x,y)$ on $[0,1]$ is the covariance function of Brownian motion and defines a canonical bounded integral operator.

- **Differentiation.** The operator $T = d/dx$ on $L^2([0,1])$ is **unbounded**: the functions $f_n(x) = \sin(n\pi x)/\sqrt{2}$ are unit vectors with $\|f_n'\|^2 = n^2\pi^2/2 \to \infty$. Differential operators are the principal example of unbounded operators; they require careful domain specification and are the subject of spectral theory for PDEs.

- **The shift operator.** On $\ell^2$, the right shift $S(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, \ldots)$ satisfies $\|S\| = 1$ (it is an isometry) but has no eigenvectors in $\ell^2$.

**The Banach algebra $B(H)$.** The space of all bounded linear operators $T: H \to H$ is denoted $B(H)$. It is a **Banach algebra**: a Banach space under the operator norm in which multiplication (composition) is associative and satisfies $\|ST\| \leq \|S\| \cdot \|T\|$. This submultiplicativity is elementary: $\|STx\| \leq \|S\| \cdot \|Tx\| \leq \|S\| \cdot \|T\| \cdot \|x\|$. The identity operator $I$ is the multiplicative unit with $\|I\| = 1$.

## The Adjoint Operator

For any bounded operator $T \in B(H)$ on a Hilbert space, the inner product allows us to define a companion operator $T^*$ called the **adjoint** via the relation

$$\langle Tx, y \rangle = \langle x, T^*y \rangle \quad \text{for all } x, y \in H.$$

**Existence and uniqueness of $T^*$.** For each fixed $y$, the map $x \mapsto \langle Tx, y \rangle$ is a bounded linear functional on $H$ (with norm $\leq \|T\| \cdot \|y\|$). By the Riesz representation theorem (proved in L1), there is a unique vector, which we call $T^*y$, such that $\langle Tx, y \rangle = \langle x, T^* y \rangle$. The assignment $y \mapsto T^*y$ is linear and bounded with $\|T^*\| = \|T\|$, so $T^* \in B(H)$.

> **Intuition:** In finite dimensions, the adjoint of a real matrix $A$ is just its transpose $A^T$: the relation $\langle Ax, y \rangle = x^T A^T y = \langle x, A^T y \rangle$ holds by the symmetry of the dot product. Over $\mathbb{C}$, the adjoint is the conjugate transpose $A^* = \bar{A}^T$. The operator adjoint extends this to infinite dimensions using the Riesz theorem in place of explicit matrix entries.

**Key properties.** For $S, T \in B(H)$ and $\alpha \in \mathbb{C}$:

- $(T^*)^* = T$
- $(\alpha T)^* = \bar{\alpha} T^*$
- $(S + T)^* = S^* + T^*$
- $(ST)^* = T^* S^*$ (order reverses, as with matrix transposes)
- $\|T^* T\| = \|T\|^2$ (the $C^*$-identity, foundational in $C^*$-algebra theory)

**Special classes of operators.**

- **Self-adjoint (Hermitian):** $T = T^*$, equivalently $\langle Tx, y \rangle = \langle x, Ty \rangle$ for all $x, y$. All eigenvalues of a self-adjoint operator are real, and eigenvectors corresponding to distinct eigenvalues are orthogonal — exactly as for symmetric matrices. The covariance operator and all integral operators with symmetric kernels $K(x,y) = K(y,x)$ are self-adjoint.

- **Unitary:** $T^* T = T T^* = I$, equivalently $T$ is a bijective isometry ($\|Tx\| = \|x\|$ for all $x$). Unitary operators preserve the inner product: $\langle Tx, Ty \rangle = \langle x, T^*Ty \rangle = \langle x, y \rangle$. Examples include the Fourier transform on $L^2(\mathbb{R})$ (Plancherel's theorem) and the shift operator on $\ell^2$ restricted to an invariant subspace.

- **Normal:** $T^* T = T T^*$. Both self-adjoint and unitary operators are normal. The spectral theorem in its most general form (for bounded operators) applies to normal operators on Hilbert spaces, producing a spectral measure decomposition. We will focus on the more tractable compact case.

**Adjoint of an integral operator.** If $(Tf)(x) = \int K(x,y) f(y) \, dy$, then $(T^*g)(y) = \int \overline{K(x,y)} g(x) \, dx$. The kernel of $T^*$ is $K^*(y,x) = \overline{K(x,y)}$. Self-adjointness requires $K(x,y) = \overline{K(y,x)}$, i.e., the kernel is Hermitian-symmetric; for real kernels, this is simply $K(x,y) = K(y,x)$.

> **Key insight:** The adjoint is the mechanism by which the geometry of the Hilbert space (the inner product) interacts with the algebra of operators. Self-adjoint operators are the infinite-dimensional analogue of symmetric matrices: they have real spectra, orthogonal eigenspaces, and are the natural setting for quantum observables, covariance operators, and kernel methods.

## Compact Operators

Boundedness is a coarse condition. The class of **compact operators** is the right infinite-dimensional analogue of finite-dimensional matrices.

An operator $T \in B(H)$ is **compact** if for every bounded sequence $(x_n)$ in $H$, the image sequence $(Tx_n)$ has a convergent subsequence. Equivalently, $T$ maps every bounded set to a **precompact** set (a set whose closure is compact).

> **Intuition:** In infinite-dimensional Hilbert spaces, the closed unit ball is not compact — bounded sequences need not have convergent subsequences (this fails in $\ell^2$: the standard basis $e_n$ is bounded but has no convergent subsequence since $\|e_n - e_m\| = \sqrt{2}$ for all $n \neq m$). A compact operator "collapses" the infinite-dimensional unit ball into something compact — something that behaves like a finite-dimensional object. This is why compact operators are the right class: they are the bounded operators for which a spectral theorem analogous to the finite-dimensional case holds.

**Examples.**

- **Every finite-rank operator is compact.** If $T$ has finite-dimensional range (i.e., $\text{rank}(T) < \infty$), then bounded sets map to bounded sets in a finite-dimensional space, which are precompact by the Heine-Borel theorem.

- **Hilbert-Schmidt integral operators are compact.** If $K \in L^2(\Omega \times \Omega)$, then the integral operator $(Tf)(x) = \int_\Omega K(x,y) f(y) \, dy$ is compact on $L^2(\Omega)$. This is the primary example in ML and functional data analysis.

- **Compact operators form a closed two-sided ideal.** If $T$ is compact and $S$ is bounded, then $ST$ and $TS$ are compact. If $T_n \to T$ in operator norm and each $T_n$ is compact, then $T$ is compact.

**Approximation by finite-rank operators.** Every compact operator $T$ on a Hilbert space is the norm limit of finite-rank operators. Specifically, if $\{e_n\}$ is an orthonormal basis of $H$ and $P_n$ is the projection onto $\text{span}(e_1, \ldots, e_n)$, then $T_n = P_n T \to T$ in operator norm. This is the precise sense in which compact operators are "infinite-dimensional limits of matrices."

> **Key insight:** The passage from finite-rank to compact operator is the infinite-dimensional analogue of taking a sequence of finite matrices with increasing size. Compact operators inherit the best structural properties of matrices — in particular, a spectral theorem with a discrete spectrum — while living in a genuinely infinite-dimensional setting.

**Non-compact bounded operators.** The identity operator $I$ on an infinite-dimensional Hilbert space is bounded but not compact: the image of the unit ball is the unit ball itself, which is not compact. Multiplication operators $M_m$ with $m \in L^\infty$ that are not in $L^2$ are typically not compact. This distinction is crucial: a kernel matrix in ML is the finite-dimensional compression of an integral operator; the operator itself may be compact (when the kernel is $L^2$), but the identity "operator" that underlies naive function interpolation is not.

## Spectral Theorem for Compact Self-Adjoint Operators

The spectral theorem is the central result of this lesson. It asserts that compact self-adjoint operators behave exactly like symmetric matrices in the sense that they have an orthonormal basis of eigenvectors — but now the basis consists of functions and the eigenvalues decay to zero.

**Theorem (Spectral Theorem for Compact Self-Adjoint Operators).** Let $T: H \to H$ be a compact self-adjoint operator on a separable Hilbert space $H$. Then:

1. All eigenvalues of $T$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3. There exists a countable orthonormal set $\{\phi_n\}_{n=1}^\infty \subset H$ and real scalars $\lambda_1, \lambda_2, \ldots$ (not necessarily distinct) such that $T\phi_n = \lambda_n \phi_n$.
4. The eigenvalues satisfy $|\lambda_1| \geq |\lambda_2| \geq \cdots \to 0$.
5. The orthonormal set $\{\phi_n\}$ spans all of $H$ except the kernel: every $f \in H$ decomposes as
$$f = \sum_{n=1}^\infty \langle f, \phi_n \rangle \phi_n + P_{\ker T} f$$
where $P_{\ker T}$ is the orthogonal projection onto $\ker T$. If $\ker T = \{0\}$, then $\{\phi_n\}$ is an orthonormal basis of $H$.
6. The operator acts by:
$$Tf = \sum_{n=1}^\infty \lambda_n \langle f, \phi_n \rangle \phi_n.$$

> **Key insight:** Statement (4) — eigenvalues decay to zero — has no finite-dimensional analogue (in finite dimensions, eigenvalues can be anything). It is forced by compactness: if $|\lambda_n| \geq \varepsilon > 0$ for infinitely many $n$, then $\{T\phi_n / \lambda_n\} = \{\phi_n\}$ would be a bounded sequence with no convergent subsequence in the image, violating compactness.

**Proof sketch (Rayleigh quotient argument).**

*Step 1: Existence of a first eigenvector.* Define the Rayleigh quotient $R(f) = \langle Tf, f \rangle / \|f\|^2$ and set $\lambda_1 = \sup_{\|f\|=1} \langle Tf, f \rangle$ (the spectral radius equals $\|T\|$ for self-adjoint operators). Take a maximizing sequence $\|f_n\| = 1$ with $\langle Tf_n, f_n \rangle \to \lambda_1$. Since $T$ is compact, pass to a subsequence so $Tf_n \to g$ for some $g \in H$. A calculation shows $Tf_n - \lambda_1 f_n \to 0$ in norm, so $f_n \to g/\lambda_1 =: \phi_1$ and $T\phi_1 = \lambda_1\phi_1$. The infimum similarly yields an eigenvector with eigenvalue $\lambda_0 = \inf_{\|f\|=1} \langle Tf, f \rangle$; the first eigenvalue extracted is whichever of $\lambda_1, \lambda_0$ has larger absolute value.

> **Remember:** The attainment of the Rayleigh quotient supremum requires compactness. Take a maximizing sequence $\|f_n\| = 1$ with $\langle Tf_n, f_n\rangle \to \lambda_1 = \|T\|$. By compactness of $T$, $\{Tf_n\}$ has a convergent subsequence. Using the identity $\|Tf - \lambda_1 f\|^2 = \|Tf\|^2 - 2\lambda_1\langle Tf, f\rangle + \lambda_1^2$, one can show the limit point is an eigenvector with eigenvalue $\lambda_1$. This is one of two places in the proof where compactness is genuinely used.

*Step 2: Induction on the orthogonal complement.* Let $H_1 = (\text{span}\{\phi_1\})^\perp$. Since $T$ is self-adjoint, $T$ maps $H_1$ to $H_1$ (self-adjoint operators preserve orthogonal complements of their eigenspaces). The restriction $T|_{H_1}$ is again compact and self-adjoint on $H_1$. Apply Step 1 to obtain $\phi_2 \perp \phi_1$ with $T\phi_2 = \lambda_2 \phi_2$ and $|\lambda_2| \leq |\lambda_1|$.

*Step 3: Eigenvalue decay.* If $|\lambda_n| \geq \varepsilon$ for all $n$, then $T\phi_n = \lambda_n \phi_n$ gives $\|T\phi_n\| \geq \varepsilon$, but $\{\phi_n\}$ is an orthonormal sequence with $\phi_n \rightharpoonup 0$ (weakly). Compactness of $T$ would require $T\phi_n \to 0$ in norm, contradiction. Hence $\lambda_n \to 0$.

*Step 4: Completeness.* The remaining piece $f - \sum_n \langle f, \phi_n \rangle \phi_n$ lies in $\bigcap_n H_n$ and maps to $0$ under $T$ (the inductive process exhausts all nonzero eigenvalues), hence belongs to $\ker T$.

**The eigenfunction expansion.** The decomposition $Tf = \sum_n \lambda_n \langle f, \phi_n \rangle \phi_n$ is the infinite-dimensional analogue of $A = Q\Lambda Q^T$. The eigenfunctions $\phi_n$ play the role of eigenvectors; $\langle f, \phi_n \rangle$ plays the role of the coordinate of $f$ in the eigenbasis; and the eigenvalues $\lambda_n$ are real and decay to zero.

**Connection to the finite-dimensional spectral theorem.** If $H = \mathbb{R}^n$ and $T$ is a symmetric matrix, then every linear operator is compact (finite-dimensional spaces are locally compact), and the theorem reduces exactly to the spectral theorem from M1: $n$ real eigenvalues, orthonormal eigenvectors, diagonalization $A = Q\Lambda Q^T$.

## Operator SVD (Compact Operators)

For non-self-adjoint compact operators, the eigendecomposition is replaced by the **singular value decomposition**. For a compact operator $T: H \to H$, the operator $T^*T$ is compact, self-adjoint, and positive semidefinite ($\langle T^*Tx, x \rangle = \|Tx\|^2 \geq 0$). By the spectral theorem, $T^*T$ has an orthonormal basis of eigenfunctions $\{u_n\}$ with non-negative eigenvalues $\sigma_n^2 \geq 0$ decaying to zero.

The **singular values** of $T$ are $\sigma_n = \sqrt{\lambda_n(T^*T)} \geq 0$, ordered $\sigma_1 \geq \sigma_2 \geq \cdots \to 0$. Setting $v_n = Tu_n / \sigma_n$ (for $\sigma_n > 0$) yields an orthonormal set $\{v_n\}$ in $H$, and the **singular value decomposition** is:

$$Tf = \sum_{n=1}^\infty \sigma_n \langle f, u_n \rangle v_n.$$

The vectors $\{u_n\}$ are the right singular functions (input modes), $\{v_n\}$ are the left singular functions (output modes), and $\{\sigma_n\}$ are the singular values. The operator norm satisfies $\|T\| = \sigma_1$.

> **Intuition:** The SVD decomposes $T$ into pure "stretching" operations: input $u_n$ is mapped to $\sigma_n v_n$. No rotation is mixed with the scaling. This is exact the same structure as the finite-dimensional SVD from M1, now holding for any compact operator on a Hilbert space.

**Eckart-Young theorem in infinite dimensions.** The best rank-$k$ approximation to $T$ in operator norm is:

$$T_k = \sum_{n=1}^k \sigma_n \langle \cdot, u_n \rangle v_n, \quad \|T - T_k\| = \sigma_{k+1}.$$

This is the operator-theoretic Eckart-Young theorem: truncating to the top $k$ singular components gives the nearest rank-$k$ operator. The approximation error is exactly the $(k+1)$-th singular value — and since $\sigma_n \to 0$ for compact operators, this error vanishes as $k \to \infty$. In the self-adjoint case, $u_n = v_n = \phi_n$ and $\sigma_n = |\lambda_n|$, recovering the eigendecomposition truncation.

> **Key insight:** Eckart-Young in infinite dimensions tells us that compact operators can be approximated to arbitrary precision by finite-rank operators (matrices), with the approximation quality controlled by the decay rate of singular values. Fast-decaying singular values mean the operator is well-approximated by a small matrix — a finite-dimensional computation is adequate.

## Hilbert-Schmidt Operators

Among all compact operators, the **Hilbert-Schmidt operators** form the most tractable subclass — they carry a natural Hilbert space structure themselves.

**Definition.** Let $\{e_n\}$ be any orthonormal basis of $H$. A bounded operator $T$ is **Hilbert-Schmidt** if

$$\|T\|_{HS}^2 = \sum_{n=1}^\infty \|Te_n\|^2 < \infty.$$

This sum is independent of the choice of orthonormal basis (verified by expanding in two different bases and applying Parseval's identity). The quantity $\|T\|_{HS}$ is the **Hilbert-Schmidt norm**.

**Properties.**

- Every Hilbert-Schmidt operator is compact.
- $\|T\|_{HS}^2 = \sum_n \sigma_n^2$ (sum of squared singular values), so Hilbert-Schmidt operators are those with $\sum_n \sigma_n^2 < \infty$. Compare: for **trace-class** (nuclear) operators, $\sum_n \sigma_n < \infty$.
- $\|T\| \leq \|T\|_{HS}$ (operator norm is bounded by Hilbert-Schmidt norm).
- The set of Hilbert-Schmidt operators on $H$ is itself a Hilbert space under the inner product $\langle S, T \rangle_{HS} = \sum_n \langle Se_n, Te_n \rangle = \text{tr}(S^*T)$.

**Hilbert-Schmidt integral operators.** On $H = L^2(\Omega)$, the operator $(Tf)(x) = \int_\Omega K(x,y) f(y) \, dy$ is Hilbert-Schmidt if and only if $K \in L^2(\Omega \times \Omega)$:

$$\|T\|_{HS}^2 = \int_\Omega \int_\Omega |K(x,y)|^2 \, dy \, dx < \infty.$$

**Proof.** Expand in an ONB $\{e_n\}$ of $L^2(\Omega)$:

$$\|Te_n\|^2 = \int \left|\int K(x,y) e_n(y) \, dy\right|^2 dx.$$

Summing and applying Parseval in $y$ (the functions $e_n(y)$ form an ONB): $\sum_n \|Te_n\|^2 = \int \int |K(x,y)|^2 \, dy \, dx = \|K\|_{L^2(\Omega^2)}^2$.

> **Key insight:** The Hilbert-Schmidt norm of an integral operator equals the $L^2$ norm of its kernel. This is the infinite-dimensional analogue of the Frobenius norm of a matrix: $\|A\|_F^2 = \sum_{i,j} A_{ij}^2$, since the matrix entries are precisely the kernel evaluated at the grid points.

**Connection to kernel methods.** In a reproducing kernel Hilbert space (RKHS) with kernel $k(x,y)$, the integral operator $(T_k f)(x) = \int k(x,y) f(y) d\mu(y)$ defines the **kernel integral operator**. It is Hilbert-Schmidt when $\int \int k(x,y)^2 \, d\mu(x) \, d\mu(y) < \infty$ — satisfied by all standard kernels (RBF, polynomial, Matern) on compact domains with finite measures. The eigenfunctions and eigenvalues of $T_k$ are exactly the Mercer eigenfunctions: $T_k \phi_n = \mu_n \phi_n$, and Mercer's theorem states $k(x,y) = \sum_n \mu_n \phi_n(x) \phi_n(y)$ with convergence in $L^2$.

## ML Connections

The spectral theory of compact operators is not an abstract curiosity — it underlies several core structures in modern machine learning.

**1. Kernel matrices as finite-dimensional approximations.**

Given $n$ data points $x_1, \ldots, x_n$ drawn from a distribution $\mu$ on $\Omega$, the kernel matrix $K \in \mathbb{R}^{n \times n}$ with $K_{ij} = k(x_i, x_j)$ is a finite-rank approximation to the Hilbert-Schmidt integral operator $T_k$ on $L^2(\Omega, \mu)$. Specifically, if we form the empirical measure $\hat{\mu}_n = \frac{1}{n}\sum_i \delta_{x_i}$, then $(1/n) K$ is the matrix of $T_k$ in the basis of point evaluation functionals. The eigenvalues of $(1/n) K$ converge to the eigenvalues $\mu_n$ of $T_k$ as $n \to \infty$ (by the law of large numbers applied to the empirical spectral measure). This is why kernel PCA with $n$ points gives $n$ eigenvalues that approximate the infinite spectrum $\{\mu_n\}$.

**2. PCA via the covariance operator.**

Let $X$ be a random variable in $L^2(\Omega)$ with $\mathbb{E}[X] = 0$. The **covariance operator** is:

$$C: f \mapsto \mathbb{E}[\langle X, f \rangle X].$$

This is a compact self-adjoint positive semidefinite operator on $L^2(\Omega)$. Its eigenfunctions $\{\phi_n\}$ are the functional principal components and its eigenvalues $\lambda_n$ are the explained variances in each direction. Functional PCA (FPCA) — used in longitudinal data analysis, EEG, fMRI, climate modeling — is exactly the spectral theorem applied to $C$. The finite-dimensional PCA of discretized data approximates this spectral decomposition.

**3. The neural tangent kernel (NTK).**

For a neural network $f_\theta: \mathbb{R}^d \to \mathbb{R}$ trained by gradient descent, the NTK is:

$$K_{\text{NTK}}(x, x') = \nabla_\theta f_\theta(x)^T \nabla_\theta f_\theta(x').$$

In the infinite-width limit, $K_{\text{NTK}}$ converges to a deterministic kernel that remains constant during training (Jacot et al., 2018). The associated integral operator $T_{K_{\text{NTK}}}$ is Hilbert-Schmidt on $L^2$ of the data distribution, and its eigendecomposition determines training dynamics: components of the target function along eigenfunctions with large eigenvalues are learned quickly, while those along eigenfunctions with small eigenvalues are learned slowly or not at all. This **spectral bias** (Rahaman et al., 2019) — the tendency of neural networks to learn low-frequency components first — is a direct consequence of the eigenvalue decay of $T_{K_{\text{NTK}}}$.

**4. Truncated SVD as optimal low-rank approximation.**

The Eckart-Young theorem for compact operators shows that the best rank-$k$ approximation to a Hilbert-Schmidt operator $T$ is obtained by truncating its SVD to the top $k$ singular values. In ML this manifests in: (a) **low-rank kernel approximations** (Nyström method, random Fourier features) approximate $T_k$ with a rank-$m$ operator, incurring error $\sigma_{m+1}$; (b) **weight matrix compression** in neural networks via SVD truncation; (c) **attention matrices** in transformers, which can be interpreted as low-rank operators in function space when the query/key dimension is smaller than the sequence length.

> **Key insight:** The unified theme is that compactness + self-adjointness forces the operator to behave like a diagonal matrix in the eigenbasis, with diagonal entries (eigenvalues) decaying to zero. This decay rate governs approximation quality, learning dynamics, and computational cost of any spectral method.

## Python: Integral Operators and Spectral Decomposition

The following code implements the full pipeline for the Brownian motion covariance kernel $K(x,y) = \min(x,y)$ on $[0,1]$: discretize the Hilbert-Schmidt integral operator, compute its spectrum numerically, verify against the theoretical eigenvalues $\lambda_n = 4/((2n-1)^2\pi^2)$, visualize the eigenfunctions, and demonstrate the Eckart-Young approximation.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# 1. Discretize the Hilbert-Schmidt integral operator with kernel K(x,y) = min(x,y)
#    This is the covariance kernel of standard Brownian motion on [0,1].
#    The integral operator (Tf)(x) = int_0^1 min(x,y) f(y) dy
#    is discretized on a uniform grid using the midpoint rule:
#       T_ij ~ K(x_i, x_j) * delta_x
# -----------------------------------------------------------------------
N = 512  # grid resolution
x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N  # midpoints of N subintervals
dx = 1.0 / N

# Build the kernel matrix: K[i,j] = min(x_i, x_j)
X, Y = np.meshgrid(x, x, indexing='ij')
K_mat = np.minimum(X, Y)  # shape (N, N)

# The discretized operator matrix: T_mat = K_mat * dx
# Eigenvalue equation: T_mat @ v = lambda * v  (v approximates eigenfunction at grid pts)
T_mat = K_mat * dx

# -----------------------------------------------------------------------
# 2. Compute eigenvalues and eigenvectors of the symmetric matrix T_mat
#    numpy.linalg.eigh exploits symmetry and returns real eigenvalues in ascending order.
# -----------------------------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eigh(T_mat)

# eigh returns eigenvalues in ascending order; reverse to get descending
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Normalize eigenvectors as functions: the L^2 norm of a discretized function f_vec is
#   ||f||_{L^2} ~ sqrt(sum(f_vec^2) * dx)
# Normalize so that each eigenfunction has L^2 norm 1.
norms = np.sqrt(np.sum(eigenvectors**2 * dx, axis=0))
eigenvectors = eigenvectors / norms[np.newaxis, :]

# Fix sign: ensure each eigenfunction is positive at its first large-amplitude grid point
for n in range(eigenvectors.shape[1]):
    idx = np.argmax(np.abs(eigenvectors[:, n]))
    if eigenvectors[idx, n] < 0:
        eigenvectors[:, n] *= -1

# -----------------------------------------------------------------------
# 3. Theoretical spectrum: lambda_n = 4 / ((2n-1)^2 * pi^2)
#    These are the exact eigenvalues of the Brownian motion covariance operator.
#    The corresponding eigenfunctions are phi_n(x) = sqrt(2) * sin((2n-1)*pi*x/2).
# -----------------------------------------------------------------------
n_compare = 20  # compare first 20 eigenvalues
ns = np.arange(1, n_compare + 1)
lambda_theory = 4.0 / ((2 * ns - 1)**2 * np.pi**2)
lambda_numerical = eigenvalues[:n_compare]

print("Eigenvalue comparison (first 10):")
print(f"{'n':>4} {'Theory':>12} {'Numerical':>12} {'Rel. Error':>12}")
for n in range(10):
    rel_err = abs(lambda_numerical[n] - lambda_theory[n]) / lambda_theory[n]
    print(f"{n+1:>4} {lambda_theory[n]:>12.6f} {lambda_numerical[n]:>12.6f} {rel_err:>12.2e}")

# -----------------------------------------------------------------------
# 4. Visualize eigenfunctions and compare to theory
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_spec = fig.add_subplot(gs[0, :2])
ax_err = fig.add_subplot(gs[0, 2])

# Eigenvalue spectrum: numerical vs theoretical
ax_spec.semilogy(ns, lambda_theory, 'o-', color='steelblue', label='Theory: $4/((2n-1)^2\\pi^2)$', markersize=5)
ax_spec.semilogy(ns, lambda_numerical, 's--', color='firebrick', label='Numerical', markersize=5, alpha=0.8)
ax_spec.set_xlabel('$n$', fontsize=12)
ax_spec.set_ylabel('Eigenvalue $\\lambda_n$', fontsize=12)
ax_spec.set_title('Spectrum of the Brownian Motion Covariance Operator', fontsize=12)
ax_spec.legend(fontsize=10)
ax_spec.grid(True, which='both', alpha=0.3)

# Relative errors
rel_errors = np.abs(lambda_numerical - lambda_theory) / lambda_theory
ax_err.semilogy(ns, rel_errors, 'o-', color='darkorange', markersize=5)
ax_err.set_xlabel('$n$', fontsize=12)
ax_err.set_ylabel('Relative error', fontsize=12)
ax_err.set_title('Eigenvalue Relative Error\n(theory vs. numerical)', fontsize=12)
ax_err.grid(True, which='both', alpha=0.3)

# Plot eigenfunctions n=1,2,3 and compare to theoretical phi_n(x) = sqrt(2)*sin((2n-1)*pi*x/2)
for i, n in enumerate([1, 2, 3]):
    ax = fig.add_subplot(gs[1, i])
    phi_theory = np.sqrt(2) * np.sin((2 * n - 1) * np.pi * x / 2)
    phi_num = eigenvectors[:, n - 1]
    # Align sign to theory
    if np.dot(phi_num, phi_theory) < 0:
        phi_num = -phi_num
    ax.plot(x, phi_theory, '-', color='steelblue', lw=2, label='Theory', alpha=0.8)
    ax.plot(x, phi_num, '--', color='firebrick', lw=1.5, label='Numerical', alpha=0.9)
    ax.set_title(f'Eigenfunction $n={n}$', fontsize=11)
    ax.set_xlabel('$x$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hilbert-Schmidt Integral Operator: Brownian Motion Kernel', fontsize=13, y=1.01)
plt.savefig('brownian_spectrum.png', dpi=120, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------
# 5. Eckart-Young: reconstruct the operator from top-k eigencomponents
#    T_k(x,y) = sum_{n=1}^{k} lambda_n * phi_n(x) * phi_n(y)
#    Reconstruction error in the operator norm = lambda_{k+1}
# -----------------------------------------------------------------------
ks = [1, 3, 5, 10, 20, 50]
print("\nEckart-Young reconstruction error ||T - T_k||:")
print(f"{'k':>5} {'||T-T_k|| (theory=lambda_{k+1})':>35} {'Frobenius error':>18}")

for k in ks:
    # Rank-k approximation: T_k = V @ diag(lambda) @ V^T where V = first k eigenvectors
    V_k = eigenvectors[:, :k]                        # (N, k)
    Lambda_k = eigenvalues[:k]                        # (k,)
    T_k = (V_k * Lambda_k[np.newaxis, :]) @ V_k.T    # (N, N)
    T_k_scaled = T_k  # already scaled by dx in eigendecomposition

    frob_err = np.sqrt(np.sum((T_mat - T_k_scaled)**2)) * dx
    op_norm_err = eigenvalues[k] if k < len(eigenvalues) else 0.0
    print(f"{k:>5} {op_norm_err:>35.6e} {frob_err:>18.6e}")

# Visual: show reconstruction quality for k=1, 5, 20
fig2, axes = plt.subplots(1, 4, figsize=(16, 4))

im = axes[0].imshow(T_mat, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', aspect='auto')
axes[0].set_title('True operator $T$', fontsize=11)
axes[0].set_xlabel('$y$'); axes[0].set_ylabel('$x$')
plt.colorbar(im, ax=axes[0], fraction=0.046)

for ax, k in zip(axes[1:], [1, 5, 20]):
    V_k = eigenvectors[:, :k]
    T_k = (V_k * eigenvalues[:k][np.newaxis, :]) @ V_k.T
    im = ax.imshow(T_k, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', aspect='auto',
                   vmin=T_mat.min(), vmax=T_mat.max())
    ax.set_title(f'Rank-{k} approximation $T_{{{k}}}$', fontsize=11)
    ax.set_xlabel('$y$')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Eckart-Young: Low-rank reconstruction of the Brownian motion operator', fontsize=12)
plt.tight_layout()
plt.savefig('eckart_young_reconstruction.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Expected output.** The relative errors in the eigenvalue comparison should be at the level of $10^{-3}$ to $10^{-4}$ for the first 10 eigenvalues, improving with grid resolution $N$. The Eckart-Young errors confirm that $\|T - T_k\| \approx \lambda_{k+1} \approx 4/((2k+1)^2\pi^2)$, which decays as $O(k^{-2})$.

:::quiz
question: "An operator T on an infinite-dimensional Hilbert space is bounded with ||T|| = 1. Which of the following is guaranteed?"
options:
  - "T is compact"
  - "T maps every bounded sequence to a sequence with a convergent subsequence"
  - "T is continuous: if x_n -> x in norm, then Tx_n -> Tx in norm"
  - "T has at least one eigenvalue"
correct: 2
explanation: "Boundedness is equivalent to continuity for linear operators on Banach spaces: ||T(x_n - x)|| <= ||T|| * ||x_n - x|| -> 0. Options (1) and (2) describe compactness, which is strictly stronger than boundedness — the identity on an infinite-dimensional space is bounded but not compact. Option (4) is false: bounded operators on infinite-dimensional spaces need not have any eigenvalues (e.g., the shift operator on l^2)."
:::

:::quiz
question: "Let T be a compact self-adjoint operator on L^2([0,1]) with eigenfunctions {phi_n} and eigenvalues {lambda_n} -> 0. You want to solve the integral equation Tf = g for an unknown f, given g in L^2. Under what condition does a solution exist, and what is it?"
options:
  - "A solution always exists; it is f = sum_n (1/lambda_n) <g, phi_n> phi_n"
  - "A solution exists if and only if g is orthogonal to ker(T) and sum_n |<g, phi_n>|^2 / lambda_n^2 < infinity; the solution is f = sum_n (1/lambda_n) <g, phi_n> phi_n + f_0 for any f_0 in ker(T)"
  - "A solution exists only if all eigenvalues are nonzero and bounded away from zero"
  - "A solution never exists because T is not invertible"
correct: 1
explanation: "Expanding in the eigenbasis: Tf = sum_n lambda_n <f, phi_n> phi_n = g requires lambda_n <f, phi_n> = <g, phi_n>, so <f, phi_n> = <g, phi_n>/lambda_n for lambda_n != 0. For f to lie in L^2, we need sum_n |<g, phi_n>|^2 / lambda_n^2 < infinity (Parseval). The component of g in ker(T) must be zero (since T maps ker(T) to 0), and the component of f in ker(T) is free. This is the Fredholm alternative for compact operators."
:::

:::quiz
question: "A kernel k(x,y) = exp(-||x-y||^2 / (2*sigma^2)) (RBF/Gaussian kernel) on a compact domain Omega with the uniform measure defines an integral operator T_k on L^2(Omega). Which statement is most precise?"
options:
  - "T_k is bounded but not compact, since the feature map is infinite-dimensional"
  - "T_k is compact and Hilbert-Schmidt, with Hilbert-Schmidt norm equal to the L^2(Omega x Omega) norm of the kernel"
  - "T_k is compact but not Hilbert-Schmidt, because the RBF kernel is not square-integrable"
  - "T_k is unitary because the Gaussian kernel is a probability density"
correct: 1
explanation: "On a compact domain with uniform measure, the RBF kernel satisfies int int exp(-||x-y||^2/sigma^2) dx dy < infinity (bounded domain, bounded integrand), so k in L^2(Omega x Omega). By the theory of Hilbert-Schmidt integral operators, T_k is Hilbert-Schmidt with ||T_k||_{HS}^2 = int int k(x,y)^2 dx dy. Hilbert-Schmidt implies compact. The infinite-dimensionality of the RKHS feature map is a property of the kernel function, not of the operator's compactness — these are different objects."
:::
