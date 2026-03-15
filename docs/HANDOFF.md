# Handoff Notes — AI Mastery Hub

**Last updated:** 2026-03-16
**Status:** S1 complete (all 6 modules written + feedback-corrected). S2 structure locked. Ready to write S2/M1 content.

---

## Current State

### What exists

- `frontend/` + `backend/` — fully implemented (Phase 1–3 complete per original spec)
- `content/curriculum.json` — full 8-semester outline, S1 all available, S2 modules defined (coming-soon)
- `content/s1-math-foundations/` — 6 modules, 30 lessons, all written and feedback-corrected
- `content/s2-classical-ml/` — 6 module directories + module.json stubs, no lesson .md files yet
- `feedback/` — reviewer feedback files used after each module/semester

### What is NOT done yet

- S2 lesson content (30 lessons, 6 modules) — **start here next session**
- S3–S8 content — not started, not structured

---

## Immediate Next Step

**Write S2/M1: Statistical Learning Theory** — 5 lessons, in order:
1. `l1-pac-learning.md`
2. `l2-vc-dimension.md`
3. `l3-rademacher-complexity.md`
4. `l4-algorithmic-stability.md`
5. `l5-pac-bayes.md`

Files go in: `content/s2-classical-ml/m1-statistical-learning-theory/`

After all 5 lessons are written → send to feedback review → apply fixes → commit → move to M2.

---

## Content Writing Process (follow exactly)

### Step 1: Write lessons

Write all 5 lessons in parallel using background agents (one per lesson). Each lesson is a standalone `.md` file.

**Lesson structure (every lesson must have all of these):**

```
# [Lesson Title]

## [Section 1 — foundational concept]
## [Section 2 — core theory with definitions/theorems]
## [Section 3 — proofs/derivations]
## [Section N — ...]

## ML Connections
[How this topic appears in modern ML practice — 4–6 named applications
 with specific model names, papers, or algorithms. Not generic. This section
 is mandatory in every lesson.]

## Python Implementation
[Runnable code demonstrating the key concept. Prefer numpy/scipy/sklearn.
 Include printed output showing what the code produces.]

:::quiz
question: "..."
options:
  - "..."
  - "..."
  - "..."
  - "..."
correct: N   # 0-indexed
explanation: "..."
:::
```

**Quality standard:** PhD/graduate level. Assume the reader has completed S1 (measure theory, linear algebra, functional analysis, optimization, optimal transport). Do not re-derive S1 material — reference it.

### Step 2: Callout boxes

Use these blockquote-based boxes throughout every lesson (not just at the end):

| Box type | Syntax | Purpose |
|----------|--------|---------|
| Key insight | `> **Key insight:**` | Non-obvious conceptual observations |
| Intuition | `> **Intuition:**` | Geometric or informal mental model before/after a hard result |
| Refresher | `> **Refresher:**` | Remind reader of S1 prerequisite without re-teaching it |
| Remember | `> **Remember:**` | Critical formula or fact to internalize |

Rules:
- No emojis in callout boxes or anywhere in content
- Place boxes at natural teaching moments — before a hard theorem (Intuition/Refresher), after a proof (Key insight/Remember)
- 3–6 boxes per lesson is typical; fewer is fine if the content flows naturally

### Step 3: Math formatting

- All math in KaTeX-compatible LaTeX: `$inline$` and `$$display$$`
- For intimidating formulas, add a **Remember** or **Intuition** box immediately after explaining what each symbol means and what the formula "does" in plain language
- Theorem/Definition/Proof blocks use bold labels: `**Theorem.**`, `**Proof.**`, `**Definition.**`

### Step 4: Feedback review

After writing a full module (all 5 lessons), create a feedback file at:
`feedback/S[N]_M[N].md`

Then apply the fixes — must-fix items first (correctness errors), then should-fix (precision/pedagogy).

**Must-fix = factual errors** — wrong sign, wrong theorem statement, wrong claim. Fix immediately.
**Should-fix = precision gaps** — imprecise phrasing, missing caveat, better example. Apply selectively.

### Step 5: Commit

After feedback fixes are applied, get a commit message from the assistant and commit.

Commit convention:
- Task-focused only, no author names, no `Co-Authored-By` lines
- Separate commits for: (a) initial module write, (b) feedback fixes

---

## S2 Finalized Structure

All 6 modules and 30 lessons are locked. IDs match `curriculum.json` exactly.

### M1: Statistical Learning Theory
| ID | Title | Min |
|----|-------|-----|
| l1-pac-learning | PAC Learning & Sample Complexity | 35 |
| l2-vc-dimension | VC Dimension & Shattering | 35 |
| l3-rademacher-complexity | Rademacher Complexity & Uniform Convergence | 40 |
| l4-algorithmic-stability | Algorithmic Stability & Uniform Stability | 35 |
| l5-pac-bayes | PAC-Bayes Bounds | 40 |

### M2: Kernel Methods
| ID | Title | Min |
|----|-------|-----|
| l1-svm-primal-dual | SVMs: Primal, Dual & KKT Conditions | 35 |
| l2-soft-margin-hinge-loss | Soft Margin, Hinge Loss & Regularization Paths | 35 |
| l3-structured-svms | Structured SVMs & Output Kernels | 40 |
| l4-nystrom-random-features | Nyström Approximation & Random Features | 35 |
| l5-kernel-pca-spectral | Kernel PCA, Spectral Clustering & Kernel CCA | 40 |

### M3: Gaussian Processes
| ID | Title | Min |
|----|-------|-----|
| l1-gp-regression-bo | GP Regression, Marginal Likelihood & Bayesian Optimization | 40 |
| l2-covariance-spectral | Covariance Function Design, Stationarity & Spectral Methods | 35 |
| l3-gp-classification | GP Classification & Approximate Inference | 40 |
| l4-sparse-gps | Sparse GPs & Inducing Point Methods | 40 |
| l5-deep-gps-ntk | Deep GPs, NTK Correspondence & Neural Network Priors | 45 |

### M4: Probabilistic Graphical Models
| ID | Title | Min |
|----|-------|-----|
| l1-directed-graphical-models | Directed Graphical Models & d-Separation | 35 |
| l2-undirected-mrfs | Undirected Models & Markov Random Fields | 35 |
| l3-exact-inference | Exact Inference: Variable Elimination & Junction Tree | 40 |
| l4-belief-propagation | Belief Propagation & Loopy BP | 40 |
| l5-em-hmms-crfs | EM Algorithm, HMMs & CRF Training | 45 |

### M5: Bayesian Inference & Monte Carlo
| ID | Title | Min |
|----|-------|-----|
| l1-bayesian-computation | Bayesian Computation & Conjugate Analysis | 35 |
| l2-mcmc-mh-gibbs | MCMC: Metropolis-Hastings & Gibbs Sampling | 40 |
| l3-hmc-nuts | Hamiltonian Monte Carlo & NUTS | 40 |
| l4-variational-inference | Variational Inference, ELBO & Amortized Inference | 40 |
| l5-smc-particle-filters | Sequential Monte Carlo & Particle Filters | 40 |

### M6: Nonparametric Bayesian Methods
| ID | Title | Min |
|----|-------|-----|
| l1-dirichlet-process | Dirichlet Process & Chinese Restaurant Process | 40 |
| l2-hierarchical-dp | Hierarchical Dirichlet Processes & Topic Models | 40 |
| l3-indian-buffet-process | Indian Buffet Process & Latent Feature Models | 40 |
| l4-gplvm | GP Latent Variable Models & Dimensionality Reduction | 40 |
| l5-bnp-regression | Bayesian Nonparametric Regression & Density Estimation | 40 |

---

## S1 Completion Record

All 6 modules written and feedback-corrected:

| Module | Written | Feedback applied |
|--------|---------|-----------------|
| m3-information-theory | ✅ | ✅ |
| m1-linear-algebra | ✅ | ✅ |
| m2-probability-theory | ✅ | ✅ |
| m4-optimization-theory | ✅ | ✅ |
| m5-optimal-transport | ✅ | ✅ |
| m6-functional-analysis | ✅ | ✅ |

---

## S1 → S2 Dependency Map

Use these when writing S2 lessons — reference S1 material by module/lesson ID rather than re-deriving:

| S2 topic | References from S1 |
|----------|--------------------|
| M1 PAC/VC/Rademacher | S1-M2 concentration inequalities (l3), S1-M3 information theory |
| M1 PAC-Bayes | S1-M3 KL divergence (l2), S1-M2 measure theory |
| M2 SVMs / RKHS | S1-M6 RKHS (l3), S1-M4 optimization duality (l1) |
| M2 Nyström/RFF | S1-M6 Mercer's theorem (l3), S1-M1 SVD (l1) |
| M2 Kernel PCA / Spectral | S1-M1 eigendecomposition (l2), S1-M6 compact operators (l2) |
| M3 GP regression | S1-M6 RKHS/GP connection (l3), S1-M2 Gaussian measures |
| M3 Covariance/Spectral | S1-M6 Bochner's theorem (l3), S1-M1 Fourier analysis |
| M3 Sparse GPs / ELBO | S1-M4 optimization (l1–l2) |
| M3 Deep GPs / NTK | S1-M6 NTK (l5) |
| M4 PGMs | S1-M2 conditional probability, independence |
| M4 Belief propagation | S1-M3 information theory (l3 mutual information) |
| M4 EM algorithm | S1-M3 KL divergence, S1-M4 coordinate ascent |
| M5 MCMC | S1-M2 Markov chains (l4 martingales), S1-M2 measure theory |
| M5 HMC | S1-M4 Hamiltonian dynamics, S1-M2 SDEs (l5) |
| M5 VI / ELBO | S1-M3 KL divergence (l2), S1-M4 optimization |
| M5 SMC | S1-M2 conditional expectation, filtering |
| M6 Dirichlet Process | S1-M2 measure theory, S1-M3 entropy |
| M6 GP-LVM | S1-M6 RKHS (l3), S2-M3 GP regression |
| M6 BNP regression | S1-M4 RKHS regularization, S2-M5 VI |

---

## Key Technical Details (do not change)

- **Math rendering:** KaTeX via react-markdown + remark-math + rehype-katex + remark-gfm
- **Quiz syntax:** `:::quiz` YAML blocks, `correct` field is 0-indexed integer
- **Callout detection:** blockquote keyword prefix — `Key insight`, `Intuition`, `Refresher`, `Remember`
- **No emojis** anywhere in content or callout boxes
- **Content path:** `content/s2-classical-ml/<module-id>/<lesson-id>.md`
- **Status flow:** lesson written → `"status": "available"` in both `curriculum.json` and `module.json`

---

## Design Decisions (locked — do not re-open)

| Decision | Choice |
|----------|--------|
| Visual personality | Bold & Modern — dark slate-900, indigo/violet gradient accents |
| Navigation | Persistent 280px sidebar |
| Lesson layout | Deep Reader (wide content column, theory → math → code) |
| Content storage | Markdown files on disk |
| Content IDs | Kebab-case slugs matching directory names |
| Session auth | Anonymous — localStorage UUID, `X-Session-Id` header |
| API envelope | `{ ok, data }` / `{ ok: false, error: { code, message } }` |
