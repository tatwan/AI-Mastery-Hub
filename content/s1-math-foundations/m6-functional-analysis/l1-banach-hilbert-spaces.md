---
title: "Banach Spaces & Hilbert Spaces"
estimatedMinutes: 35
tags: ["Banach-spaces", "Hilbert-spaces", "L2-spaces", "completeness", "Riesz-representation"]
prerequisites: ["l1-svd-low-rank from m1", "l1-entropy from m3"]
---

# Banach Spaces & Hilbert Spaces

The tools you built in M1 through M5 — SVD, measure theory, convex analysis, optimal transport — all live in spaces with rich algebraic and geometric structure. Most of those arguments worked in $\mathbb{R}^n$, where every Cauchy sequence converges and intuition about angles and distances is reliable. In machine learning, however, the objects we care about — probability densities, kernel-induced feature maps, neural network function classes, sample paths of Gaussian processes — live in infinite-dimensional spaces. This lesson builds the framework that makes infinite-dimensional geometry rigorous and usable.

---

## 1. From Finite to Infinite Dimensions

### The Problem with Infinite Dimensions

In $\mathbb{R}^n$, bounded and closed implies compact (Heine-Borel). Every Cauchy sequence converges. The unit ball is compact. These facts underlie virtually every convergence argument in finite-dimensional optimization and statistics.

None of these facts survive the passage to infinite dimensions.

**Example: a sequence that "wants" to converge but has no limit in the space.**

Consider the space $C([0,1])$ of continuous functions on $[0,1]$, equipped with the $L^1$ norm $\|f\| = \int_0^1 |f(x)|\,dx$. Define:

$$f_n(x) = \begin{cases} 0 & 0 \le x \le \tfrac{1}{2} - \tfrac{1}{n} \\ n\bigl(x - \tfrac{1}{2} + \tfrac{1}{n}\bigr) & \tfrac{1}{2} - \tfrac{1}{n} < x \le \tfrac{1}{2} \\ 1 & \tfrac{1}{2} < x \le 1 \end{cases}$$

Each $f_n$ is continuous. The sequence is Cauchy in $L^1$: $\|f_n - f_m\|_{L^1} \to 0$. But the pointwise limit is the Heaviside step function $\mathbf{1}_{[1/2, 1]}$, which is discontinuous. The limit exists in $L^1([0,1])$ but not in $C([0,1])$.

The space $C([0,1])$ with the $L^1$ norm has a "hole" at the step function.

> **Key insight:** Completeness — the requirement that every Cauchy sequence converges within the space — is not automatic. It is a structural property that must be verified or imposed. Without it, limiting arguments fail at the last step.

### Why This Matters for ML

Machine learning is full of approximation arguments: we approximate a target function by a sequence of neural networks, or a distribution by a sequence of finite mixtures, or an integral operator by a finite-rank matrix. Each of these is a limit argument. To conclude that the limit exists and lies in a useful space, we need completeness.

The passage from finite to infinite dimensions forces us to:
1. Carefully specify which norm we are using (different norms can make the same space complete or incomplete).
2. Distinguish between algebraic properties (linearity, inner products) and topological properties (convergence, compactness).
3. Recognize that the dual space — the space of bounded linear functionals — has its own structure that does not trivially generalize.

---

## 2. Normed Spaces and Banach Spaces

### Norms: the Axioms

Let $V$ be a vector space over $\mathbb{R}$ (or $\mathbb{C}$). A **norm** on $V$ is a map $\|\cdot\|: V \to [0, \infty)$ satisfying:

1. **Positive definiteness:** $\|v\| = 0 \iff v = 0$
2. **Absolute homogeneity:** $\|\alpha v\| = |\alpha|\,\|v\|$ for all scalars $\alpha$
3. **Triangle inequality:** $\|u + v\| \le \|u\| + \|v\|$

A vector space equipped with a norm is a **normed space**. The norm induces a metric $d(u, v) = \|u - v\|$, so every normed space is a metric space.

> **Refresher:** In M1 you worked with the Frobenius norm $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ and the operator norm $\|A\|_2 = \sigma_1(A)$. Both are norms on the vector space of matrices. The Frobenius norm comes from an inner product; the operator norm does not.

### The $L^p$ Spaces

Fix a measure space $(\Omega, \mathcal{F}, \mu)$. For $1 \le p < \infty$, define:

$$L^p(\Omega, \mu) = \left\{ f: \Omega \to \mathbb{R} \;\Big|\; \int_\Omega |f|^p \, d\mu < \infty \right\} \Big/ \sim$$

where $f \sim g$ iff $f = g$ $\mu$-almost everywhere. The $L^p$ norm is:

$$\|f\|_p = \left(\int_\Omega |f|^p \, d\mu\right)^{1/p}$$

For $p = \infty$:

$$\|f\|_\infty = \operatorname{ess\,sup}_{\omega \in \Omega} |f(\omega)|$$

The identification up to null sets is essential: without it, positive definiteness fails ($\|f\|_p = 0$ would only imply $f = 0$ almost everywhere, not everywhere).

**Holder's inequality:** For $\frac{1}{p} + \frac{1}{q} = 1$:

$$\int |fg|\,d\mu \le \|f\|_p \|g\|_q$$

**Minkowski's inequality** (triangle inequality for $L^p$):

$$\|f + g\|_p \le \|f\|_p + \|g\|_p$$

### The $\ell^p$ Spaces

For sequences $(a_1, a_2, \ldots)$:

$$\ell^p = \left\{ (a_n)_{n \ge 1} \;\Big|\; \sum_{n=1}^\infty |a_n|^p < \infty \right\}, \quad \|(a_n)\|_{\ell^p} = \left(\sum_{n=1}^\infty |a_n|^p\right)^{1/p}$$

These are the discrete analogs of $L^p$ spaces, corresponding to counting measure on $\mathbb{N}$.

**Key example:** $\ell^2$ is the space of square-summable sequences. Its inner product structure makes it the canonical infinite-dimensional analog of $\mathbb{R}^n$.

### Banach Spaces: Completeness

**Definition.** A sequence $(v_n)$ in a normed space $V$ is **Cauchy** if for every $\varepsilon > 0$ there exists $N$ such that $\|v_n - v_m\| < \varepsilon$ for all $n, m > N$.

Every convergent sequence is Cauchy. The converse — every Cauchy sequence converges — defines completeness.

**Definition.** A normed space $V$ is a **Banach space** if it is complete: every Cauchy sequence converges to an element of $V$.

| Space | Norm | Banach? |
|---|---|---|
| $L^p(\Omega, \mu)$, $1 \le p \le \infty$ | $\|\cdot\|_p$ | Yes (Riesz-Fischer) |
| $\ell^p$, $1 \le p \le \infty$ | $\|\cdot\|_{\ell^p}$ | Yes |
| $C([a,b])$ | $\|f\|_\infty = \sup|f|$ | Yes |
| $C([a,b])$ | $\|f\|_1 = \int|f|$ | No |
| Polynomials on $[0,1]$ | $\|\cdot\|_\infty$ | No |
| $\mathbb{R}^n$ | any norm | Yes |

> **Intuition:** Completeness means the space has no missing "limit points." Think of the rational numbers $\mathbb{Q}$: the sequence $3, 3.1, 3.14, 3.141, \ldots$ is Cauchy in $\mathbb{Q}$ but its limit $\pi$ is not in $\mathbb{Q}$. The reals $\mathbb{R}$ are the completion of $\mathbb{Q}$. Banach spaces are the function-space analog of $\mathbb{R}$.

**Theorem (Riesz-Fischer).** $L^p(\Omega, \mu)$ is a Banach space for all $1 \le p \le \infty$.

The proof for $1 \le p < \infty$ works by showing that if $\sum_n \|f_n\|_p < \infty$ then $\sum_n f_n$ converges in $L^p$ (absolutely convergent series converge), then using this to extract limits of Cauchy sequences. The key step uses the monotone convergence theorem from measure theory.

> **Key insight:** The Riesz-Fischer theorem is why we quotient by null sets in the definition of $L^p$. Without this identification, the "norm" fails positive definiteness (a function that is zero a.e. but not everywhere would have zero norm), and the completeness proof breaks.

---

## 3. Hilbert Spaces

### Inner Products

An **inner product** on a vector space $V$ over $\mathbb{R}$ is a map $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

1. **Symmetry:** $\langle u, v \rangle = \langle v, u \rangle$
2. **Linearity in first argument:** $\langle \alpha u + \beta w, v \rangle = \alpha \langle u, v \rangle + \beta \langle w, v \rangle$
3. **Positive definiteness:** $\langle v, v \rangle \ge 0$, with equality iff $v = 0$

Every inner product induces a norm: $\|v\| = \sqrt{\langle v, v \rangle}$.

**Not every norm comes from an inner product.** The $L^p$ norm for $p \ne 2$ does not arise from an inner product. The characterization is the **parallelogram law**:

$$\|u + v\|^2 + \|u - v\|^2 = 2\|u\|^2 + 2\|v\|^2$$

A norm satisfies the parallelogram law if and only if it arises from an inner product. One checks that $\|\cdot\|_p$ for $p \ne 2$ violates this law (try $u = e_1$, $v = e_2$ in $\ell^p$).

> **Remember:**
> **Parallelogram law:** $\|u+v\|^2 + \|u-v\|^2 = 2\|u\|^2 + 2\|v\|^2$
> **Polarization identity:** $\langle u, v \rangle = \tfrac{1}{4}\bigl(\|u+v\|^2 - \|u-v\|^2\bigr)$

The polarization identity recovers the inner product from the norm, proving that the inner product is uniquely determined by the norm (when it exists).

### The Cauchy-Schwarz Inequality

**Theorem.** For any inner product space:

$$|\langle u, v \rangle| \le \|u\|\,\|v\|$$

with equality iff $u$ and $v$ are linearly dependent.

**Proof sketch.** For any $t \in \mathbb{R}$: $0 \le \|u - tv\|^2 = \|u\|^2 - 2t\langle u, v\rangle + t^2\|v\|^2$. This quadratic in $t$ is non-negative, so its discriminant is non-positive: $4\langle u,v\rangle^2 - 4\|u\|^2\|v\|^2 \le 0$.

### Hilbert Spaces: Definition

**Definition.** A **Hilbert space** is an inner product space that is complete with respect to the norm induced by the inner product.

Equivalently: a Hilbert space is a Banach space whose norm satisfies the parallelogram law.

### $L^2$ as the Canonical Hilbert Space

The most important Hilbert space in analysis and ML is $L^2(\Omega, \mu)$ with the inner product:

$$\langle f, g \rangle = \int_\Omega f(x) g(x) \, d\mu(x)$$

**Verification:** Symmetry and linearity are immediate. Positive definiteness requires the measure-theoretic fact from M2 that $\int f^2 \, d\mu = 0 \Rightarrow f = 0$ a.e. The Cauchy-Schwarz inequality for $L^2$ is:

$$\left|\int fg \, d\mu\right| \le \|f\|_2 \|g\|_2$$

which is the integral form of Cauchy-Schwarz and follows from Holder's inequality with $p = q = 2$.

> **Key insight:** $L^2$ is special among all $L^p$ spaces: it is the only one with an inner product, the only one isometric to its own dual (as we will see via Riesz representation), and the only one in which orthogonality — a geometric concept — makes rigorous sense. This is why Gaussian processes, RKHS, and Fourier analysis all live naturally in $L^2$.

**Important example:** $L^2([-\pi, \pi])$ with Lebesgue measure. The inner product is:

$$\langle f, g \rangle = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x) g(x) \, dx$$

(the $\frac{1}{2\pi}$ normalization is conventional and ensures the trigonometric functions have unit norm). This is the home of classical Fourier analysis.

---

## 4. Orthogonality and Projections

### Orthogonality

Two elements $u, v \in H$ are **orthogonal**, written $u \perp v$, if $\langle u, v \rangle = 0$.

The **Pythagorean theorem** holds in any inner product space: if $u \perp v$ then $\|u + v\|^2 = \|u\|^2 + \|v\|^2$.

The **orthogonal complement** of a set $S \subseteq H$ is:

$$S^\perp = \{v \in H : \langle v, s \rangle = 0 \text{ for all } s \in S\}$$

$S^\perp$ is always a closed subspace, regardless of whether $S$ is closed or even a subspace. Moreover $(S^\perp)^\perp = \overline{\text{span}(S)}$.

### The Projection Theorem

This is the central geometric theorem of Hilbert space theory.

**Theorem (Projection Theorem / Best Approximation).** Let $M$ be a closed subspace of a Hilbert space $H$, and let $x \in H$. Then:

1. There exists a unique element $\hat{x} \in M$ satisfying $\|x - \hat{x}\| = \inf_{m \in M} \|x - m\|$.
2. $\hat{x}$ is characterized by: $\hat{x} \in M$ and $x - \hat{x} \perp M$ (i.e., $\langle x - \hat{x}, m \rangle = 0$ for all $m \in M$).
3. The map $P_M : x \mapsto \hat{x}$ is a bounded linear operator with $P_M^2 = P_M$ (idempotent), $\|P_M\| = 1$, and $H = M \oplus M^\perp$.

**Proof sketch (existence and uniqueness):** Let $d = \inf_{m \in M}\|x - m\|$ and pick $m_n \in M$ with $\|x - m_n\| \to d$. Apply the parallelogram law to $(x - m_n)$ and $(x - m_k)$:

$$\|(x-m_n) + (x-m_k)\|^2 + \|(x-m_n) - (x-m_k)\|^2 = 2\|x-m_n\|^2 + 2\|x-m_k\|^2$$

Since $M$ is a subspace, $\frac{m_n + m_k}{2} \in M$, so $\|x - \frac{m_n+m_k}{2}\| \ge d$, giving:

$$\|m_n - m_k\|^2 = \|(x-m_n)-(x-m_k)\|^2 \le 2\|x-m_n\|^2 + 2\|x-m_k\|^2 - 4d^2 \to 0$$

So $(m_n)$ is Cauchy, hence converges (completeness!) to some $\hat{x} \in M$ (closedness). Uniqueness follows from strict convexity of the norm: if $\hat{x}_1 \ne \hat{x}_2$ both minimize, their midpoint is in $M$ and is strictly closer to $x$.

> **Key insight:** This proof uses completeness and the parallelogram law — both Hilbert space properties. In a Banach space that is not a Hilbert space (say $L^1$), nearest-point projections onto closed subspaces need not exist or be unique. The inner product is not a luxury; it is what makes projection well-behaved.

> **Intuition:** In $\mathbb{R}^3$, the projection of a point onto a plane is the foot of the perpendicular from the point to the plane. The characterization $x - \hat{x} \perp M$ is exactly this: the error vector is perpendicular to the subspace. The projection theorem says this geometric picture is valid in any Hilbert space, including infinite-dimensional ones.

### Gram-Schmidt in Infinite Dimensions

Given a countable sequence $\{v_1, v_2, \ldots\}$ of linearly independent vectors in $H$, the Gram-Schmidt procedure produces an orthonormal sequence $\{e_1, e_2, \ldots\}$ with $\text{span}(e_1, \ldots, e_n) = \text{span}(v_1, \ldots, v_n)$ for each $n$:

$$u_1 = v_1, \quad e_1 = \frac{u_1}{\|u_1\|}$$

$$u_{n+1} = v_{n+1} - \sum_{k=1}^n \langle v_{n+1}, e_k\rangle e_k, \quad e_{n+1} = \frac{u_{n+1}}{\|u_{n+1}\|}$$

The procedure is identical to the finite-dimensional case; the infinite-dimensional case simply runs indefinitely.

---

## 5. Orthonormal Bases and Fourier Expansions

### Orthonormal Systems

A set $\{e_\alpha\}_{\alpha \in A}$ in a Hilbert space $H$ is **orthonormal** if $\langle e_\alpha, e_\beta \rangle = \delta_{\alpha\beta}$.

For a finite or countable orthonormal set $\{e_1, e_2, \ldots\}$ and any $x \in H$, the numbers $\langle x, e_n \rangle$ are the **Fourier coefficients** of $x$ with respect to this system.

### Bessel's Inequality

**Theorem.** For any orthonormal sequence $\{e_n\}$ and any $x \in H$:

$$\sum_{n=1}^\infty |\langle x, e_n \rangle|^2 \le \|x\|^2$$

**Proof.** For any $N$, let $S_N = \sum_{n=1}^N \langle x, e_n\rangle e_n$. By orthonormality:

$$0 \le \left\|x - S_N\right\|^2 = \|x\|^2 - 2\sum_{n=1}^N |\langle x,e_n\rangle|^2 + \sum_{n=1}^N |\langle x, e_n\rangle|^2 = \|x\|^2 - \sum_{n=1}^N |\langle x, e_n\rangle|^2$$

So $\sum_{n=1}^N |\langle x, e_n\rangle|^2 \le \|x\|^2$ for all $N$; take $N \to \infty$.

Bessel's inequality says that the Fourier coefficients are square-summable — they define an element of $\ell^2$. It does not say the expansion converges to $x$.

### Complete Orthonormal Systems (Orthonormal Bases)

**Definition.** An orthonormal set $\{e_n\}$ in $H$ is **complete** (or an **orthonormal basis**) if any of the following equivalent conditions hold:

1. $\overline{\text{span}\{e_n\}} = H$ (the closed linear span is all of $H$)
2. $x \perp e_n$ for all $n$ implies $x = 0$
3. For every $x \in H$: $x = \sum_{n=1}^\infty \langle x, e_n \rangle e_n$ (convergence in norm)
4. **Parseval's identity:** $\|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2$

> **Remember:**
> **Parseval's identity:** $\|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2$
> This is the infinite-dimensional Pythagorean theorem: the squared norm equals the sum of squared Fourier coefficients.

The equivalence of (3) and (4) follows because $\|x - S_N\|^2 = \|x\|^2 - \sum_{n=1}^N |\langle x, e_n\rangle|^2$, which goes to zero iff Parseval's identity holds.

**Theorem.** Every separable Hilbert space has a countable orthonormal basis.

(A Hilbert space is **separable** if it has a countable dense subset. All Hilbert spaces appearing in ML — $L^2$ with standard measures, $\ell^2$, RKHS of most kernels — are separable.)

### Fourier Series in $L^2([-\pi, \pi])$

The functions

$$e_0(x) = \frac{1}{\sqrt{2\pi}}, \quad e_n^c(x) = \frac{\cos(nx)}{\sqrt{\pi}}, \quad e_n^s(x) = \frac{\sin(nx)}{\sqrt{\pi}}, \quad n = 1, 2, \ldots$$

form a complete orthonormal system in $L^2([-\pi, \pi])$ (with Lebesgue measure, no normalization factor). Equivalently, using complex exponentials:

$$\phi_n(x) = \frac{e^{inx}}{\sqrt{2\pi}}, \quad n \in \mathbb{Z}$$

form a complete orthonormal basis. For any $f \in L^2([-\pi, \pi])$:

$$f = \sum_{n=-\infty}^\infty \hat{f}(n)\, \phi_n \quad \text{(convergence in } L^2\text{)}$$

where $\hat{f}(n) = \langle f, \phi_n \rangle = \frac{1}{\sqrt{2\pi}}\int_{-\pi}^\pi f(x) e^{-inx} dx$ are the Fourier coefficients.

Parseval's identity becomes:

$$\|f\|_{L^2}^2 = \frac{1}{2\pi}\int_{-\pi}^\pi |f(x)|^2 dx = \sum_{n=-\infty}^\infty |\hat{f}(n)|^2$$

> **Key insight:** The Fourier expansion is not just a computational tool; it is the statement that the trigonometric system is an orthonormal basis for $L^2$. The convergence is in $L^2$ norm, not pointwise. Pointwise convergence requires additional regularity (Dirichlet's theorem, Dini conditions). In ML we almost always care about $L^2$ convergence, which is guaranteed for all $f \in L^2$.

**Warning on $C([-\pi,\pi])$ vs $L^2([-\pi,\pi])$:** Continuous functions form a dense subspace of $L^2$, but there exist continuous functions whose Fourier series diverge pointwise at some point (Kolmogorov's example, extended to $L^2$ by du Bois-Reymond). $L^2$ convergence and pointwise convergence are genuinely different.

---

## 6. The Riesz Representation Theorem

### Bounded Linear Functionals

A **linear functional** on $H$ is a linear map $\phi: H \to \mathbb{R}$.

A linear functional is **bounded** if there exists $C < \infty$ such that $|\phi(x)| \le C\|x\|$ for all $x \in H$. Equivalently, $\phi$ is continuous. The **operator norm** of $\phi$ is:

$$\|\phi\| = \sup_{\|x\| \le 1} |\phi(x)|$$

**Examples:**
- Fix $v \in H$. The map $\phi_v(x) = \langle x, v \rangle$ is a bounded linear functional with $\|\phi_v\| = \|v\|$ (by Cauchy-Schwarz, with equality at $x = v/\|v\|$).
- Fix $t_0 \in [a,b]$. The evaluation functional $\delta_{t_0}(f) = f(t_0)$ is a linear functional on $C([a,b])$ but is NOT bounded on $L^2([a,b])$ (elements of $L^2$ are equivalence classes and need not be pointwise defined).

The **dual space** $H^*$ is the Banach space of all bounded linear functionals on $H$, with norm $\|\phi\|_{H^*}$.

### The Riesz Representation Theorem

**Theorem (Riesz, 1907).** Let $H$ be a Hilbert space. For every bounded linear functional $\phi \in H^*$, there exists a unique $v \in H$ such that:

$$\phi(x) = \langle x, v \rangle \quad \text{for all } x \in H$$

Moreover, $\|\phi\|_{H^*} = \|v\|_H$.

Consequently, the map $v \mapsto \phi_v$ (where $\phi_v(x) = \langle x, v \rangle$) is an isometric isomorphism $H \xrightarrow{\sim} H^*$.

**Proof.** If $\phi = 0$, take $v = 0$. Otherwise, let $M = \ker(\phi) = \{x: \phi(x) = 0\}$. Since $\phi$ is bounded, $M$ is a closed subspace. Since $\phi \ne 0$, $M \ne H$, so $M^\perp \ne \{0\}$.

Pick any $w \in M^\perp$ with $w \ne 0$. For any $x \in H$, write:

$$x = \underbrace{\left(x - \frac{\phi(x)}{\phi(w)} w\right)}_{\in M} + \frac{\phi(x)}{\phi(w)} w$$

(The first term is in $M$ because applying $\phi$ gives $\phi(x) - \frac{\phi(x)}{\phi(w)}\phi(w) = 0$.)

Since $w \perp M$, taking the inner product of $x$ with $w$:

$$\langle x, w \rangle = \frac{\phi(x)}{\phi(w)}\|w\|^2$$

Therefore $\phi(x) = \langle x, \frac{\phi(w)}{\|w\|^2} w \rangle$. Set $v = \frac{\overline{\phi(w)}}{\|w\|^2} w$ (with conjugation for complex case).

**Uniqueness:** If $\langle x, v \rangle = \langle x, v' \rangle$ for all $x$, then $\langle x, v - v' \rangle = 0$ for all $x$, so taking $x = v - v'$ gives $\|v - v'\|^2 = 0$.

> **Key insight:** The Riesz representation theorem says $H \cong H^*$ isometrically as Banach spaces. Hilbert spaces are **self-dual**. This is what makes Hilbert spaces so computationally convenient: there is no difference between "vectors" and "linear measurements of vectors." In contrast, for $L^p$ with $p \ne 2$, the dual is $L^q$ with $q \ne p$, and the spaces are genuinely different.

> **Intuition:** The theorem says every way of linearly measuring elements of $H$ (every bounded linear functional) is secretly just "taking an inner product with some fixed vector $v$." There is no exotic way to linearly measure things in a Hilbert space; everything reduces to inner products. This is not true in general Banach spaces.

### Consequences

1. **Weak convergence in $H$:** $x_n \rightharpoonup x$ weakly iff $\langle x_n, v \rangle \to \langle x, v \rangle$ for all $v \in H$.

2. **Adjoints:** For any bounded linear operator $T: H \to H$, the Riesz theorem guarantees existence of a unique adjoint $T^*: H \to H$ satisfying $\langle Tx, y \rangle = \langle x, T^*y \rangle$.

3. **Reproducing kernels (RKHS):** If evaluation functionals $\delta_t: f \mapsto f(t)$ are bounded on a Hilbert space $\mathcal{H}$, then Riesz gives $\delta_t(f) = \langle f, k_t \rangle$ for some $k_t \in \mathcal{H}$. The function $k(s,t) = \langle k_s, k_t \rangle$ is the reproducing kernel. This is the entire foundation of kernel methods.

---

## 7. ML Connections

### 1. Kernel Methods and RKHS

A **reproducing kernel Hilbert space (RKHS)** $\mathcal{H}$ is a Hilbert space of functions on some set $\mathcal{X}$ in which point-evaluation functionals are bounded. By Riesz representation, each evaluation is an inner product: $f(x) = \langle f, k(\cdot, x) \rangle_\mathcal{H}$.

The **feature map** $\phi: \mathcal{X} \to \mathcal{H}$ defined by $\phi(x) = k(\cdot, x)$ embeds data points into $\mathcal{H}$ (possibly infinite-dimensional, as with the Gaussian RBF kernel $k(x,y) = e^{-\|x-y\|^2/2\sigma^2}$ whose RKHS is a Sobolev-type space). The kernel trick computes inner products in this space without explicitly constructing $\phi(x)$: $\langle \phi(x), \phi(y) \rangle_\mathcal{H} = k(x,y)$.

Support vector machines, Gaussian process regression, and kernel principal component analysis all perform linear algebra in $\mathcal{H}$, using the Riesz theorem and projection theorem in an essential way.

### 2. PCA as Hilbert Space Projection

In finite dimensions, PCA computes the projection of data onto the top-$k$ eigenspace of the covariance matrix. In the infinite-dimensional setting, this generalizes to the **Karhunen-Loeve expansion**: for a mean-zero stochastic process $X_t$ with covariance kernel $K(s,t) = \mathbb{E}[X_s X_t]$, the integral operator $T_K f(s) = \int K(s,t) f(t) \, dt$ is a compact self-adjoint operator on $L^2$. Its eigenfunctions $\{e_n\}$ form an orthonormal basis and $X_t = \sum_n \xi_n e_n(t)$ where $\xi_n = \langle X, e_n \rangle_{L^2}$ are uncorrelated. This is PCA in $L^2$.

Functional data analysis (FDA) uses this framework to do regression and classification with inputs in $L^2$.

### 3. Gaussian Processes

A Gaussian process $f \sim \mathcal{GP}(0, k)$ has sample paths that (generically) do not lie in the RKHS $\mathcal{H}_k$, but do lie in a larger Hilbert space. Precisely, $f \in L^2(\mathcal{X}, \mu)$ almost surely (under appropriate conditions on $k$ and $\mu$). The posterior update in GP regression is a projection in $\mathcal{H}_k$: the posterior mean is the projection of $f$ onto the subspace spanned by the kernel functions at the training points, $\{k(\cdot, x_i)\}_{i=1}^n$.

The Riesz theorem underlies the connection: $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}_k}$ is the Riesz representation of the evaluation functional in the RKHS, which is why $k$ encodes both the inner product structure and the covariance of the process.

### 4. Neural Network Function Spaces

Neural networks define function classes $\mathcal{F}_{W,\sigma}$ parameterized by weights. An active area of research asks: what Hilbert (or Banach) space does $\mathcal{F}$ approximate as width/depth grow?

For shallow networks with a single hidden layer, the **Barron space** is a Hilbert space with norm related to the spectral decay of the Fourier transform of the target function. Functions in Barron space are approximable by width-$n$ networks with $O(1/n)$ $L^2$ error, free of the curse of dimensionality. The $L^2$ structure is essential for the approximation-theory arguments.

For deep networks, the relevant spaces are less understood; current research (neural tangent kernel, mean-field limits) uses $L^2$ and RKHS theory as the primary analytical tools.

### 5. Wasserstein Space is NOT a Hilbert Space

In M5 you studied the Wasserstein distance $W_2(\mu, \nu)$ on probability measures. The space $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ is a metric space but is emphatically not a Hilbert space:

- There is no inner product on $\mathcal{P}_2$ that induces the $W_2$ metric.
- The parallelogram law fails: for $\mu = \delta_0$, $\nu = \delta_1$: $W_2(\delta_0 + \delta_1, 0)$ and related combinations do not satisfy the identity.
- The "geodesics" in Wasserstein space (displacement interpolations) behave like geodesics on a curved manifold, not like straight lines in a Hilbert space.

This has concrete consequences: optimization algorithms designed for Hilbert spaces (gradient descent, proximal methods with $L^2$ prox) do not directly apply to $\mathcal{P}_2$. One must use Wasserstein gradient flows, which require the theory of optimal transport (M5) rather than Hilbert space geometry.

> **Key insight:** The rich geometry of $L^2$ — inner products, orthogonal projections, Riesz representation — is not generic. It is special to $p = 2$. Many spaces in ML (Wasserstein, $L^1$, total variation) lack this geometry, which is why new tools (optimal transport, subgradient methods, total variation regularization) are needed for them.

---

## 8. Python: Fourier Bases and $L^2$ Approximation

The following code demonstrates four things: (1) truncated Fourier series approximation of functions in $L^2([-\pi,\pi])$; (2) visualization of $L^2$ projection onto increasing subspaces spanned by Fourier basis functions; (3) numerical verification of Parseval's identity; (4) convergence of the $L^2$ approximation error as the number of terms grows.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------
# Setup: grid and target functions
# -----------------------------------------------------------------------
N_grid = 2048
x = np.linspace(-np.pi, np.pi, N_grid, endpoint=False)
dx = x[1] - x[0]

def l2_norm(f):
    """L2 norm on [-pi, pi] with Lebesgue measure (no 1/2pi factor)."""
    return np.sqrt(np.sum(f**2) * dx)

def l2_inner(f, g):
    """L2 inner product on [-pi, pi]."""
    return np.sum(f * g) * dx

# Target functions to approximate
def target_square_wave(x):
    return np.sign(np.sin(x))

def target_sawtooth(x):
    # f(x) = x/pi on (-pi, pi), period 2pi
    return x / np.pi

def target_smooth(x):
    return np.exp(np.cos(x))

# -----------------------------------------------------------------------
# Fourier basis: orthonormal system in L2([-pi, pi])
# Using convention: e_0 = 1/sqrt(2pi),
#                   cos basis: cos(nx)/sqrt(pi), n >= 1
#                   sin basis: sin(nx)/sqrt(pi), n >= 1
# -----------------------------------------------------------------------
def fourier_basis_functions(x, N_terms):
    """
    Returns list of (N_terms*2 + 1) basis functions.
    Order: constant, cos(x), sin(x), cos(2x), sin(2x), ...
    """
    basis = [np.ones_like(x) / np.sqrt(2 * np.pi)]
    for n in range(1, N_terms + 1):
        basis.append(np.cos(n * x) / np.sqrt(np.pi))
        basis.append(np.sin(n * x) / np.sqrt(np.pi))
    return basis

def fourier_projection(f, x, N_terms):
    """
    Project f onto span of first N_terms Fourier basis functions.
    Returns: approximation, list of Fourier coefficients.
    """
    basis = fourier_basis_functions(x, N_terms)
    coeffs = [l2_inner(f, e) for e in basis]
    approx = sum(c * e for c, e in zip(coeffs, basis))
    return approx, coeffs

# -----------------------------------------------------------------------
# Figure 1: Fourier approximations of the square wave
# -----------------------------------------------------------------------
f = target_square_wave(x)
fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle(r"Fourier Approximation of Square Wave in $L^2([-\pi, \pi])$",
              fontsize=14, fontweight='bold')

term_counts = [1, 3, 5, 10, 20, 50]
for ax, N in zip(axes.flat, term_counts):
    approx, _ = fourier_projection(f, x, N)
    err = l2_norm(f - approx)
    ax.plot(x, f, 'k--', lw=1.5, alpha=0.5, label='Target')
    ax.plot(x, approx, 'steelblue', lw=2,
            label=f'$S_{{{N}}}f$')
    ax.set_title(f'$N = {N}$ terms, $\\|f - S_N f\\|_{{L^2}} = {err:.4f}$',
                 fontsize=10)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_approximation.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure 1 saved: fourier_approximation.png")

# -----------------------------------------------------------------------
# Figure 2: L2 approximation error vs number of terms (3 functions)
# -----------------------------------------------------------------------
Ns = list(range(0, 51))
functions = {
    'Square wave': target_square_wave(x),
    'Sawtooth': target_sawtooth(x),
    r'$e^{\cos x}$': target_smooth(x),
}

errors = {name: [] for name in functions}
for N in Ns:
    for name, f in functions.items():
        approx, _ = fourier_projection(f, x, N)
        errors[name].append(l2_norm(f - approx))

fig2, ax2 = plt.subplots(figsize=(9, 5))
for name, errs in errors.items():
    ax2.semilogy(Ns, errs, marker='o', markersize=3, label=name, lw=2)

ax2.set_xlabel('Number of Fourier terms $N$', fontsize=12)
ax2.set_ylabel(r'$\|f - S_N f\|_{L^2}$', fontsize=12)
ax2.set_title(r'$L^2$ Approximation Error vs. Number of Fourier Terms', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fourier_convergence.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure 2 saved: fourier_convergence.png")

# -----------------------------------------------------------------------
# Parseval's Identity: numerical verification
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("PARSEVAL'S IDENTITY VERIFICATION")
print("="*60)

N_parseval = 200  # use enough terms to approach completeness

for name, f_arr in functions.items():
    norm_sq = l2_norm(f_arr)**2  # true ||f||^2 in L2

    basis = fourier_basis_functions(x, N_parseval)
    coeffs = [l2_inner(f_arr, e) for e in basis]
    parseval_sum = sum(c**2 for c in coeffs)

    print(f"\nFunction: {name}")
    print(f"  ||f||_L2^2 (direct):          {norm_sq:.8f}")
    print(f"  sum of |<f,e_n>|^2 (N={N_parseval}): {parseval_sum:.8f}")
    print(f"  Relative error:               {abs(norm_sq - parseval_sum)/norm_sq:.2e}")

# -----------------------------------------------------------------------
# Figure 3: Orthonormal basis verification
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("ORTHONORMALITY VERIFICATION (first 6 basis functions)")
print("="*60)

N_check = 5
basis = fourier_basis_functions(x, N_check)[:6]
names_basis = [r'$1/\sqrt{2\pi}$', r'$\cos(x)/\sqrt{\pi}$',
               r'$\sin(x)/\sqrt{\pi}$', r'$\cos(2x)/\sqrt{\pi}$',
               r'$\sin(2x)/\sqrt{\pi}$', r'$\cos(3x)/\sqrt{\pi}$']

gram_matrix = np.array([[l2_inner(ei, ej) for ej in basis] for ei in basis])

fig3, ax3 = plt.subplots(figsize=(7, 6))
im = ax3.imshow(gram_matrix, cmap='RdBu_r', vmin=-0.02, vmax=1.02)
ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels(names_basis, fontsize=9, rotation=30, ha='right')
ax3.set_yticklabels(names_basis, fontsize=9)
ax3.set_title('Gram Matrix $G_{ij} = \\langle e_i, e_j \\rangle_{L^2}$\n'
              '(should be identity matrix)', fontsize=11)
plt.colorbar(im, ax=ax3)
for i in range(6):
    for j in range(6):
        ax3.text(j, i, f'{gram_matrix[i,j]:.3f}', ha='center', va='center',
                 fontsize=8, color='black')
plt.tight_layout()
plt.savefig('gram_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure 3 saved: gram_matrix.png")

# -----------------------------------------------------------------------
# Figure 4: L2 projection as sequence of subspace approximations
# -----------------------------------------------------------------------
f_saw = target_sawtooth(x)
fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
fig4.suptitle(r'$L^2$ Projection onto Increasing Fourier Subspaces (Sawtooth)',
              fontsize=13, fontweight='bold')

for ax, N in zip(axes4, [2, 8, 30]):
    approx, coeffs = fourier_projection(f_saw, x, N)
    err = l2_norm(f_saw - approx)
    parseval_partial = sum(c**2 for c in coeffs)
    ax.fill_between(x, f_saw, approx, alpha=0.25, color='red',
                    label=r'$f - P_N f$ (error)')
    ax.plot(x, f_saw, 'k-', lw=2, label=r'$f$ (target)')
    ax.plot(x, approx, 'steelblue', lw=2, label=r'$P_N f$')
    ax.set_title(f'$N = {N}$\n'
                 f'$\\|f - P_N f\\|^2 = {err**2:.4f}$, '
                 f'$\\sum_{{|n| \\le N}} |\\hat{{f}}_n|^2 = {parseval_partial:.4f}$',
                 fontsize=9)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('l2_projection.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure 4 saved: l2_projection.png")
```

**Expected output:**

Running this code produces:
- `fourier_approximation.png`: Six panels showing truncated Fourier series of the square wave with $N = 1, 3, 5, 10, 20, 50$ terms. Gibbs phenomenon (overshoot near discontinuities) is visible and does not vanish as $N \to \infty$; the $L^2$ error still goes to zero because the overshoot region shrinks.
- `fourier_convergence.png`: Log-scale plot of $\|f - S_N f\|_{L^2}$ vs $N$. The smooth function $e^{\cos x}$ converges exponentially (spectral accuracy for analytic functions); the sawtooth and square wave converge at polynomial rate $O(1/N^{1/2})$ and $O(1/N)$ respectively, consistent with their regularity.
- Parseval verification: relative errors of order $10^{-5}$ or smaller for $N = 200$, confirming the identity numerically.
- `gram_matrix.png`: Gram matrix near the $6 \times 6$ identity matrix, confirming numerical orthonormality.
- `l2_projection.png`: Visual decomposition of the sawtooth into projection and error component, with the partial Parseval sum approaching $\|f\|^2$.

---

## 9. Quiz

:::quiz
question: "Which of the following is a Hilbert space (with its natural norm/inner product)?"
options:
  - "$L^1([0,1])$ with the norm $\\|f\\|_1 = \\int_0^1 |f(x)|\\,dx$"
  - "$C([0,1])$ with the norm $\\|f\\|_\\infty = \\sup_{x \\in [0,1]} |f(x)|$"
  - "$L^2([0,1])$ with the inner product $\\langle f, g\\rangle = \\int_0^1 f(x)g(x)\\,dx$"
  - "$\\ell^1$ with the norm $\\|(a_n)\\|_{\\ell^1} = \\sum_{n=1}^\\infty |a_n|$"
correct: 2
explanation: "$L^2([0,1])$ is a Hilbert space: the given map is a genuine inner product, and $L^2$ is complete (Riesz-Fischer). $L^1([0,1])$ is a Banach space but fails the parallelogram law — its norm does not arise from any inner product (check with $f = \\mathbf{1}_{[0,1/2]}$ and $g = \\mathbf{1}_{[1/2,1]}$: the parallelogram law gives $2 + 2 = 4$ but $\\|f+g\\|_1^2 + \\|f-g\\|_1^2 = 1 + 1 = 2 \\ne 4$). $C([0,1])$ with the sup-norm is a Banach space (complete), but again fails the parallelogram law and has no inner product. $\\ell^1$ is a Banach space but not a Hilbert space for the same reason."
:::

:::quiz
question: "Let $H = L^2([0,1])$ and let $M = \\{f \\in H : \\int_0^{1/2} f(x)\\,dx = 0\\}$. A function $g \\in H$ has $\\|g\\|_{L^2} = 1$ and $g \\notin M$. By the projection theorem, there is a unique $\\hat{g} \\in M$ minimizing $\\|g - \\hat{g}\\|_{L^2}$. Which condition characterizes $\\hat{g}$?"
options:
  - "$\\hat{g}$ minimizes $\\|\\hat{g}\\|_{L^2}$ subject to $\\hat{g} \\in M$"
  - "$g - \\hat{g} \\in M^\\perp$, i.e., $\\langle g - \\hat{g},\\, m\\rangle = 0$ for all $m \\in M$"
  - "$\\hat{g}$ is the element of $M$ closest to the origin"
  - "$\\langle g,\\, \\hat{g}\\rangle = \\|\\hat{g}\\|^2$"
correct: 1
explanation: "The projection theorem states that $\\hat{g} = \\arg\\min_{m \\in M} \\|g - m\\|$ if and only if $g - \\hat{g} \\perp M$, i.e., $\\langle g - \\hat{g}, m \\rangle = 0$ for all $m \\in M$. This is the orthogonality condition that characterizes the nearest point. Option (A) describes projecting $g$ onto $\\{0\\}$ (the zero function), not onto $M$ from $g$. Option (C) finds the element of $M$ closest to 0, not to $g$. Option (D) is the condition for $g$ to lie in $M$ (it would say $\\hat{g} = g$), which contradicts $g \\notin M$."
:::

:::quiz
question: "The Riesz representation theorem states that every bounded linear functional $\\phi$ on a Hilbert space $H$ has the form $\\phi(x) = \\langle x, v \\rangle$ for a unique $v \\in H$. Which of the following is a direct consequence of this theorem?"
options:
  - "Every Hilbert space is finite-dimensional"
  - "The evaluation functional $\\delta_t: f \\mapsto f(t)$ is bounded on $L^2([0,1])$ for every $t \\in [0,1]$"
  - "Every bounded linear functional on $H$ attains its supremum on the unit ball at $v / \\|v\\|$"
  - "The parallelogram law holds in $H$"
correct: 2
explanation: "For the functional $\\phi(x) = \\langle x, v \\rangle$ with $\\|\\phi\\| = \\|v\\|$, the supremum $\\sup_{\\|x\\|=1} |\\phi(x)|$ is attained at $x = v/\\|v\\|$ by Cauchy-Schwarz (equality when $x \\parallel v$). This follows directly from the Riesz representation combined with the equality conditions in Cauchy-Schwarz. Option (A) is false: $L^2$ and $\\ell^2$ are infinite-dimensional Hilbert spaces. Option (B) is actually FALSE — the evaluation functional is NOT bounded on $L^2([0,1])$: functions in $L^2$ are equivalence classes and need not be pointwise defined. The Riesz theorem applies to bounded functionals; $\\delta_t$ is not bounded on $L^2$, which is precisely why evaluation of $L^2$ functions at a point is ill-defined. Option (D) is a property used to DEFINE Hilbert spaces (the converse of the inner product characterization), not a consequence of Riesz."
:::
