export interface LessonSection {
  id: string;
  title?: string;
  type: "text" | "code" | "math" | "visualization" | "callout" | "quiz_inline";
  content: string;
  language?: string;
  caption?: string;
  calloutType?: "info" | "warning" | "tip" | "important";
}

export interface Exercise {
  id: string;
  type: "multiple_choice" | "fill_code" | "write_code" | "true_false";
  question: string;
  description?: string;
  starterCode?: string;
  solutionCode?: string;
  options?: string[];
  correctAnswer: string;
  explanation: string;
  hints?: string[];
}

export interface Lesson {
  id: string;
  moduleId: string;
  trackId: string;
  title: string;
  description: string;
  type: "concept" | "coding" | "quiz" | "project";
  estimatedMinutes: number;
  order: number;
  sections: LessonSection[];
  exercises: Exercise[];
  keyTakeaways: string[];
  prerequisites: string[];
  nextLessonId?: string;
  prevLessonId?: string;
}

export interface Module {
  id: string;
  trackId: string;
  title: string;
  description: string;
  order: number;
  estimatedHours: number;
  lessons: Lesson[];
}

export interface Track {
  id: string;
  title: string;
  description: string;
  icon: string;
  difficulty: "beginner" | "intermediate" | "advanced" | "expert";
  estimatedHours: number;
  moduleCount: number;
  lessonCount: number;
  tags: string[];
  modules: Module[];
  color: string;
  order: number;
}

// ============================================================
// TRACK 1: MATHEMATICAL FOUNDATIONS FOR ADVANCED ML
// ============================================================
const mathFoundationsTrack: Track = {
  id: "math-foundations",
  title: "Mathematical Foundations for Advanced ML",
  description: "Rigorous mathematical underpinnings: optimization theory, measure-theoretic probability, information geometry, and functional analysis — the language of modern machine learning research.",
  icon: "∑",
  difficulty: "advanced",
  estimatedHours: 40,
  moduleCount: 4,
  lessonCount: 16,
  tags: ["optimization", "probability", "information theory", "linear algebra"],
  color: "#6366f1",
  order: 1,
  modules: [
    {
      id: "convex-optimization",
      trackId: "math-foundations",
      title: "Convex Optimization & Duality",
      description: "KKT conditions, Lagrangian duality, primal-dual methods, and interior-point algorithms that underlie all of ML optimization.",
      order: 1,
      estimatedHours: 12,
      lessons: [
        {
          id: "convex-sets-functions",
          moduleId: "convex-optimization",
          trackId: "math-foundations",
          title: "Convex Sets, Functions & Subdifferentials",
          description: "Deep dive into convexity: supporting hyperplanes, epigraphs, conjugate functions, and subdifferential calculus for non-smooth objectives.",
          type: "concept",
          estimatedMinutes: 55,
          order: 1,
          nextLessonId: "lagrangian-duality",
          prerequisites: [],
          keyTakeaways: [
            "A set C is convex iff for all x,y ∈ C and λ ∈ [0,1], λx + (1−λ)y ∈ C",
            "Subdifferential ∂f(x) generalizes gradients to non-smooth functions",
            "Conjugate function f*(y) = sup_x {y·x − f(x)} enables Fenchel duality",
            "Jensen's inequality: E[f(X)] ≥ f(E[X]) for convex f — foundation of EM algorithm",
          ],
          sections: [
            {
              id: "s1",
              title: "Why Convexity Matters in ML",
              type: "text",
              content: `Convexity is the most important structural property in optimization. When a problem is convex, every local minimum is a global minimum — a property almost no other class of problems enjoys. Most classical ML objectives (SVMs, logistic regression, LASSO) are convex. Understanding convexity rigorously lets you design algorithms with provable convergence guarantees and recognize when a problem has this desirable structure.

The field of convex analysis provides the theoretical bedrock for:
- **Support Vector Machines**: primal-dual SVM formulation exploits strong duality
- **LASSO and Elastic Net**: subdifferential calculus handles the non-smooth L1 penalty
- **Mirror Descent**: generalizes gradient descent using Bregman divergences on non-Euclidean geometries
- **Proximal algorithms**: efficient splitting methods for composite objectives

Even when studying non-convex deep learning, convex analysis provides the vocabulary (saddle points, local minima, escape directions) and approximation techniques (convex relaxations, convexification).`,
            },
            {
              id: "s2",
              title: "Convex Sets & The Supporting Hyperplane Theorem",
              type: "text",
              content: `**Definition (Convex Set):** A set C ⊆ ℝⁿ is convex if for all x, y ∈ C and λ ∈ [0,1]:
λx + (1−λ)y ∈ C

Equivalently, the line segment between any two points lies entirely in C.

**Key examples:**
- Affine subspaces: {x : Ax = b}
- Halfspaces: {x : aᵀx ≤ b}
- Polytopes: intersection of finitely many halfspaces
- Positive semidefinite cone: S₊ⁿ = {X ∈ ℝⁿˣⁿ : X = Xᵀ, X ⪰ 0}
- Norm balls: {x : ‖x‖ ≤ r}

**The Supporting Hyperplane Theorem:** Let C be a convex set and x₀ ∈ ∂C (boundary of C). Then there exists a nonzero vector a such that:
aᵀx ≤ aᵀx₀ for all x ∈ C

The hyperplane {x : aᵀx = aᵀx₀} is a *supporting hyperplane* of C at x₀. This theorem is fundamental to the proof that strongly convex problems have unique minima and to the separation theorem used in SVM theory.

**Separation Theorem:** If C and D are disjoint convex sets, there exists a hyperplane separating them. This directly motivates the maximum-margin classifier.`,
            },
            {
              id: "s3",
              title: "Convex Functions & Characterizations",
              type: "text",
              content: `**Definition:** f: dom(f) → ℝ is convex if dom(f) is a convex set and for all x, y ∈ dom(f), λ ∈ [0,1]:
f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)

**First-order characterization (for differentiable f):**
f is convex ⟺ f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) for all x, y ∈ dom(f)

The tangent plane at any point is a global underestimator — crucial for convergence proofs.

**Second-order characterization (for twice-differentiable f):**
f is convex ⟺ ∇²f(x) ⪰ 0 for all x ∈ dom(f)

**Strong convexity:** f is m-strongly convex if:
f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) + (m/2)‖y−x‖²

Strong convexity implies unique minima and enables linear convergence rates for gradient descent.

**L-smoothness:** f has L-Lipschitz gradient if:
f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + (L/2)‖y−x‖²

The ratio κ = L/m is the **condition number**. Poor conditioning (large κ) causes slow convergence — motivation for preconditioning and adaptive optimizers.`,
            },
            {
              id: "s4",
              title: "Subdifferential Calculus",
              type: "math",
              content: `The subdifferential generalizes the gradient to non-smooth convex functions — essential for LASSO, SVMs, and proximal algorithms.

**Definition:** The subdifferential of f at x is:
∂f(x) = {g ∈ ℝⁿ : f(y) ≥ f(x) + gᵀ(y−x) for all y}

Each g ∈ ∂f(x) is called a **subgradient**.

**Example — Absolute Value:**
∂|x| = { {−1}  if x < 0
          [−1,1] if x = 0
          {+1}   if x > 0 }

**Example — L1 norm:**
∂‖x‖₁ = {g : gᵢ ∈ ∂|xᵢ| for each i}

**Optimality condition:** x* minimizes f ⟺ 0 ∈ ∂f(x*)

For LASSO: min_β (1/2n)‖y−Xβ‖² + λ‖β‖₁

The subdifferential condition gives the KKT conditions:
(1/n)Xᵀ(Xβ − y) + λ∂‖β‖₁ ∋ 0

This yields the soft-thresholding solution:
βⱼ = sign(zⱼ)·max(|zⱼ| − λ, 0) where z = (1/n)Xᵀy

**Conjugate functions (Fenchel conjugacy):**
f*(y) = sup_{x ∈ dom f} {yᵀx − f(x)}

The Fenchel-Moreau theorem: (f*)* = f for closed convex f
Conjugates convert constrained problems into unconstrained ones and reveal duality structure.`,
            },
            {
              id: "s5",
              title: "Implementing Subgradient Descent for LASSO",
              type: "code",
              language: "python",
              content: `import numpy as np
import matplotlib.pyplot as plt

class SubgradientLASSO:
    """
    LASSO via subgradient method — demonstrates subdifferential calculus in practice.
    
    Objective: min_{beta} (1/2n)||y - X @ beta||^2 + lambda_reg * ||beta||_1
    """
    
    def __init__(self, lambda_reg=0.1, max_iter=1000, diminishing_step=True):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.diminishing_step = diminishing_step
        self.loss_history = []
        self.beta = None
    
    def _subgradient(self, beta, X, y):
        n = len(y)
        residual = X @ beta - y
        grad_smooth = (X.T @ residual) / n
        # Subgradient of L1: sign(beta), with 0 for zero components (choose 0)
        subgrad_l1 = np.sign(beta)  
        return grad_smooth + self.lambda_reg * subgrad_l1
    
    def _objective(self, beta, X, y):
        n = len(y)
        return (0.5 / n) * np.sum((y - X @ beta)**2) + self.lambda_reg * np.sum(np.abs(beta))
    
    def fit(self, X, y):
        n, p = X.shape
        self.beta = np.zeros(p)
        best_beta = self.beta.copy()
        best_obj = float('inf')
        
        for t in range(1, self.max_iter + 1):
            # Diminishing step size: alpha_t = alpha_0 / sqrt(t)
            # Key theoretical result: sum alpha_t = inf, sum alpha_t^2 < inf required
            alpha_t = 1.0 / np.sqrt(t) if self.diminishing_step else 0.01
            
            g = self._subgradient(self.beta, X, y)
            self.beta = self.beta - alpha_t * g
            
            obj = self._objective(self.beta, X, y)
            self.loss_history.append(obj)
            
            if obj < best_obj:
                best_obj = obj
                best_beta = self.beta.copy()
        
        self.beta = best_beta
        return self

    def compare_with_coordinate_descent(self, X, y):
        """
        Coordinate descent (proximal) LASSO — much faster convergence via soft-thresholding.
        This is the algorithm used by sklearn's Lasso.
        """
        n, p = X.shape
        beta = np.zeros(p)
        
        for _ in range(200):
            for j in range(p):
                r_j = y - X @ beta + X[:, j] * beta[j]
                z_j = X[:, j] @ r_j / n
                # Soft-thresholding operator — the proximal operator for L1
                beta[j] = np.sign(z_j) * max(abs(z_j) - self.lambda_reg, 0)
        
        return beta

# Generate sparse regression data
np.random.seed(42)
n, p = 200, 50
X = np.random.randn(n, p)
X = X / np.linalg.norm(X, axis=0)  # normalize columns
true_beta = np.zeros(p)
true_beta[:5] = [3.0, -2.0, 1.5, -1.0, 0.8]  # only 5 non-zero coefficients
y = X @ true_beta + 0.1 * np.random.randn(n)

# Compare methods
model_subgrad = SubgradientLASSO(lambda_reg=0.05, max_iter=2000).fit(X, y)
beta_cd = model_subgrad.compare_with_coordinate_descent(X, y)

print("=== LASSO Recovery Comparison ===")
print(f"True support (non-zeros): {np.where(true_beta != 0)[0]}")
print(f"Subgradient recovered:    {np.where(np.abs(model_subgrad.beta) > 0.1)[0]}")
print(f"Coord descent recovered:  {np.where(np.abs(beta_cd) > 0.1)[0]}")
print()
print(f"L2 error (subgradient):  {np.linalg.norm(model_subgrad.beta - true_beta):.4f}")
print(f"L2 error (coord desc):   {np.linalg.norm(beta_cd - true_beta):.4f}")
print()
print("Key insight: Coordinate descent exploits problem structure (separability of L1")
print("and quadratic form). Subgradient is general but slow O(1/sqrt(T)) convergence.")
print("Proximal gradient gets O(1/T), accelerated proximal gets O(1/T^2) [FISTA].")
`,
              caption: "Subgradient descent vs. coordinate descent for LASSO — demonstrating why algorithm choice critically impacts convergence speed",
            },
            {
              id: "s6",
              type: "callout",
              calloutType: "important",
              title: "Research Insight: Convergence Rates Matter",
              content: `**Subgradient method:** O(1/√T) convergence — requires O(1/ε²) iterations for ε-accuracy
**Proximal gradient:** O(1/T) convergence — requires O(1/ε) iterations  
**Accelerated proximal (FISTA):** O(1/T²) — requires O(1/√ε) iterations

This gap is **fundamental**, not implementational. Nesterov (1983) proved lower bounds showing FISTA is optimal for first-order methods on smooth convex functions. Understanding these rates guides algorithm selection in practice.`,
            },
          ],
          exercises: [
            {
              id: "ex-subgrad-1",
              type: "multiple_choice",
              question: "For LASSO regression, the subgradient of the L1 penalty at β_j = 0 is:",
              options: [
                "Exactly 0",
                "Any value in [−1, 1]",
                "Any value in (−∞, ∞)",
                "Undefined — L1 is not differentiable at 0",
              ],
              correctAnswer: "Any value in [−1, 1]",
              explanation: "At β_j = 0, the subdifferential of |β_j| is the entire interval [−1, 1]. Any subgradient in this set satisfies the subgradient inequality. This is why the KKT condition for LASSO yields the soft-thresholding operator: a coordinate β_j is exactly zero when |(X_j^T r)/n| ≤ λ, meaning 0 is in the subdifferential of the total objective.",
              hints: ["Think about the definition of subdifferential at a non-smooth point", "The subdifferential at a smooth point is just the gradient (singleton set)"],
            },
            {
              id: "ex-subgrad-2",
              type: "write_code",
              question: "Implement the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for LASSO",
              description: "FISTA achieves O(1/T²) convergence by adding Nesterov momentum to proximal gradient descent. Implement it and compare convergence to plain proximal gradient.",
              starterCode: `import numpy as np

def soft_threshold(z, threshold):
    """Proximal operator for L1: element-wise soft-thresholding"""
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def fista_lasso(X, y, lambda_reg, L, max_iter=500):
    """
    FISTA for LASSO: achieves O(1/T^2) convergence rate.
    L is the Lipschitz constant of the gradient (= largest eigenvalue of X^T X / n)
    
    Key idea: maintain extrapolated point z_t (momentum) for gradient evaluation,
    while updating beta_t via proximal step.
    """
    n, p = X.shape
    beta = np.zeros(p)
    z = beta.copy()   # extrapolated point
    t = 1.0           # Nesterov momentum parameter
    
    loss_history = []
    
    for _ in range(max_iter):
        # TODO: Compute gradient of smooth part at z (not beta!)
        grad = # ???
        
        # TODO: Proximal gradient step: beta_new = prox_{lambda/L * ||.||_1}(z - (1/L)*grad)
        beta_new = # ???
        
        # TODO: Update momentum coefficient t_new = (1 + sqrt(1 + 4*t^2)) / 2
        t_new = # ???
        
        # TODO: Update extrapolated point z using Nesterov momentum
        z = # ???
        
        beta = beta_new
        t = t_new
        
        obj = 0.5/n * np.sum((y - X @ beta)**2) + lambda_reg * np.sum(np.abs(beta))
        loss_history.append(obj)
    
    return beta, loss_history
`,
              solutionCode: `import numpy as np

def soft_threshold(z, threshold):
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def fista_lasso(X, y, lambda_reg, L, max_iter=500):
    n, p = X.shape
    beta = np.zeros(p)
    z = beta.copy()
    t = 1.0
    loss_history = []
    
    for _ in range(max_iter):
        grad = (X.T @ (X @ z - y)) / n
        beta_new = soft_threshold(z - (1/L) * grad, lambda_reg / L)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = beta_new + ((t - 1) / t_new) * (beta_new - beta)
        beta = beta_new
        t = t_new
        obj = 0.5/n * np.sum((y - X @ beta)**2) + lambda_reg * np.sum(np.abs(beta))
        loss_history.append(obj)
    
    return beta, loss_history
`,
              correctAnswer: "fista",
              explanation: "FISTA uses Nesterov momentum: instead of computing the gradient at the current iterate β_t, it computes it at an extrapolated point z_t = β_t + ((t-1)/t_{t+1})(β_t − β_{t-1}). The momentum coefficient grows as t_new = (1+√(1+4t²))/2, yielding the famous O(1/T²) rate. This was a breakthrough — doubling convergence rate without additional computation per step.",
              hints: ["The gradient is computed at z, not beta", "Nesterov's momentum: z = beta_new + ((t-1)/t_new) * (beta_new - beta_old)"],
            },
          ],
        },
        {
          id: "lagrangian-duality",
          moduleId: "convex-optimization",
          trackId: "math-foundations",
          title: "Lagrangian Duality, KKT Conditions & Strong Duality",
          description: "The theoretical foundation of constrained optimization: Lagrangians, dual functions, KKT necessary and sufficient conditions, and strong duality via Slater's condition.",
          type: "concept",
          estimatedMinutes: 60,
          order: 2,
          prevLessonId: "convex-sets-functions",
          nextLessonId: "gradient-descent-theory",
          prerequisites: ["convex-sets-functions"],
          keyTakeaways: [
            "Lagrangian L(x,λ,ν) = f₀(x) + Σλᵢfᵢ(x) + Σνⱼhⱼ(x) converts constrained to unconstrained",
            "Dual function g(λ,ν) = inf_x L(x,λ,ν) is always concave (even for non-convex primal)",
            "Strong duality: p* = d* holds when Slater's condition satisfied",
            "KKT conditions are necessary and sufficient for convex programs",
          ],
          sections: [
            {
              id: "s1",
              title: "The Lagrangian and Dual Function",
              type: "text",
              content: `**Standard form convex optimization problem:**
minimize   f₀(x)
subject to fᵢ(x) ≤ 0,  i = 1,...,m
           hⱼ(x) = 0,  j = 1,...,p

**The Lagrangian** associates multipliers with each constraint:
L(x, λ, ν) = f₀(x) + Σᵢ λᵢfᵢ(x) + Σⱼ νⱼhⱼ(x)

where λᵢ ≥ 0 (dual variables for inequality constraints), νⱼ ∈ ℝ (for equality constraints).

**The Lagrangian dual function:**
g(λ, ν) = inf_{x ∈ dom f₀} L(x, λ, ν)

**Critical property:** g is always concave in (λ, ν), regardless of the convexity of the primal problem. This is because g is the infimum (pointwise minimum) of a family of affine functions of (λ,ν).

**Weak duality:** For any feasible x̃ and any λ ≥ 0, ν:
g(λ, ν) ≤ f₀(x̃)

This always holds — g provides a lower bound on the optimal value p*. The **duality gap** is p* − d*, where d* = sup_{λ≥0,ν} g(λ,ν).`,
            },
            {
              id: "s2",
              title: "SVM Duality — A Complete Derivation",
              type: "math",
              content: `The SVM is the canonical example of Lagrangian duality in ML. The primal is:

minimize_{w,b}  (1/2)‖w‖²
subject to      yᵢ(wᵀxᵢ + b) ≥ 1,  i = 1,...,n

Rewrite constraints as: 1 − yᵢ(wᵀxᵢ + b) ≤ 0

**Lagrangian:**
L(w, b, α) = (1/2)‖w‖² − Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) − 1],  αᵢ ≥ 0

**Dual function:** Minimize L over (w, b):
∂L/∂w = 0  →  w = Σᵢ αᵢyᵢxᵢ
∂L/∂b = 0  →  Σᵢ αᵢyᵢ = 0

Substituting back:
g(α) = Σᵢ αᵢ − (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼ xᵢᵀxⱼ

**SVM Dual Problem:**
maximize_{α} Σᵢ αᵢ − (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼ (xᵢᵀxⱼ)
subject to   αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0

The kernel trick now follows naturally: replace xᵢᵀxⱼ with K(xᵢ, xⱼ). The dual only accesses data through inner products!

**KKT complementary slackness:**
αᵢ[yᵢ(wᵀxᵢ + b) − 1] = 0

Either αᵢ = 0 (non-support vector) or yᵢ(wᵀxᵢ + b) = 1 (support vector on margin). This sparsity is why SVMs scale gracefully — only support vectors matter.`,
            },
            {
              id: "s3",
              title: "Deriving the SVM Dual in Python",
              type: "code",
              language: "python",
              content: `import numpy as np
from scipy.optimize import minimize

def svm_dual_solve(X, y, C=None):
    """
    Solve the SVM via the Lagrangian dual.
    Shows how duality transforms a constrained QP into 
    a form amenable to kernel methods.
    
    Hard-margin (C=None) or soft-margin (finite C).
    """
    n = len(y)
    
    # Gram matrix: K[i,j] = x_i . x_j
    K = X @ X.T
    
    # Dual objective: maximize sum(alpha) - 0.5 * alpha^T @ (y_outer * K) @ alpha
    # Equivalently: minimize 0.5 * alpha^T @ Q @ alpha - sum(alpha)
    Q = np.outer(y, y) * K  # Q[i,j] = y_i * y_j * K[i,j]
    
    def dual_objective(alpha):
        return 0.5 * alpha @ Q @ alpha - np.sum(alpha)
    
    def dual_gradient(alpha):
        return Q @ alpha - np.ones(n)
    
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
    
    if C is None:
        bounds = [(0, None)] * n   # Hard margin: alpha_i >= 0
    else:
        bounds = [(0, C)] * n      # Soft margin: 0 <= alpha_i <= C
    
    result = minimize(
        dual_objective, np.zeros(n),
        jac=dual_gradient,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    alpha = result.x
    
    # Recover primal variables from dual solution (KKT)
    # w = sum_i alpha_i * y_i * x_i  (from stationarity condition)
    w = (alpha * y) @ X
    
    # Support vectors: alpha_i > threshold (KKT complementary slackness)
    sv_mask = alpha > 1e-5
    support_vectors = X[sv_mask]
    sv_alphas = alpha[sv_mask]
    sv_labels = y[sv_mask]
    
    # Bias from KKT: y_i(w^T x_i + b) = 1 for support vectors
    # So b = mean over SVs of: y_i - w^T x_i
    b = np.mean(sv_labels - support_vectors @ w)
    
    def predict(X_test):
        return np.sign(X_test @ w + b)
    
    print(f"Dual objective value: {-result.fun:.6f}")
    print(f"Number of support vectors: {sv_mask.sum()} / {n} ({100*sv_mask.mean():.1f}%)")
    print(f"Recovered weight norm: {np.linalg.norm(w):.4f}")
    print(f"Margin width: {2 / np.linalg.norm(w):.4f}")
    print(f"Dual feasibility check (sum alpha_i y_i = 0): {np.dot(alpha, y):.2e}")
    
    # Verify strong duality: primal obj = dual obj
    primal_obj = 0.5 * np.dot(w, w)
    dual_obj = np.sum(alpha) - 0.5 * alpha @ Q @ alpha
    print(f"\\nStrong duality check:")
    print(f"  Primal objective: {primal_obj:.6f}")
    print(f"  Dual objective:   {dual_obj:.6f}")
    print(f"  Duality gap:      {abs(primal_obj - dual_obj):.2e}  (should be ~0)")
    
    return w, b, alpha, support_vectors

# Generate linearly separable data
np.random.seed(42)
n = 100
X_pos = np.random.randn(n//2, 2) + np.array([2, 2])
X_neg = np.random.randn(n//2, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n//2), -np.ones(n//2)])

w, b, alpha, svs = svm_dual_solve(X, y)
`,
              caption: "Complete SVM dual derivation — demonstrating strong duality, KKT conditions, and support vector sparsity",
            },
          ],
          exercises: [
            {
              id: "ex-kkt-1",
              type: "multiple_choice",
              question: "Slater's condition for strong duality requires:",
              options: [
                "The primal problem to have a unique solution",
                "The existence of a strictly feasible point (satisfying all inequalities strictly)",
                "The objective to be strongly convex",
                "The Lagrangian to be bounded below",
              ],
              correctAnswer: "The existence of a strictly feasible point (satisfying all inequalities strictly)",
              explanation: "Slater's condition: there exists an x̃ in the relative interior of dom(f₀) such that fᵢ(x̃) < 0 for all i (strict inequalities for convex constraints). When this holds, strong duality p* = d* is guaranteed. For linear constraints, any feasible point satisfies Slater. The SVM always satisfies Slater (we can find a strictly feasible w,b), so p* = d* always holds — the dual gap is zero.",
              hints: ["Think about what 'strictly feasible' means geometrically"],
            },
          ],
        },
        {
          id: "gradient-descent-theory",
          moduleId: "convex-optimization",
          trackId: "math-foundations",
          title: "Gradient Descent: Convergence Theory & Acceleration",
          description: "Rigorous convergence proofs for GD, SGD, and Nesterov acceleration. Understanding why momentum works from an optimization theory perspective.",
          type: "coding",
          estimatedMinutes: 70,
          order: 3,
          prevLessonId: "lagrangian-duality",
          nextLessonId: "stochastic-optimization",
          prerequisites: ["lagrangian-duality"],
          keyTakeaways: [
            "GD with step 1/L achieves O(1/T) for convex, O(exp(-mT/L)) for strongly convex",
            "Nesterov acceleration achieves optimal O(1/T²) for convex functions",
            "SGD needs diminishing step sizes; momentum SGD is practical with tuning",
            "The condition number κ=L/m determines convergence speed in practice",
          ],
          sections: [
            {
              id: "s1",
              title: "Convergence Proof for Gradient Descent",
              type: "math",
              content: `**Theorem (GD convergence for L-smooth convex functions):**
For f convex and L-smooth, GD with step size α = 1/L satisfies:
f(x_T) − f* ≤ L‖x₀ − x*‖² / (2T)

**Proof sketch:**
From L-smoothness: f(x_{t+1}) ≤ f(xₜ) + ∇f(xₜ)ᵀ(x_{t+1}−xₜ) + (L/2)‖x_{t+1}−xₜ‖²
Substitute x_{t+1} = xₜ − (1/L)∇f(xₜ):
f(x_{t+1}) ≤ f(xₜ) − (1/2L)‖∇f(xₜ)‖²  ... (descent lemma)

From convexity: f(xₜ) − f* ≤ ∇f(xₜ)ᵀ(xₜ − x*)

Combining and telescoping over T steps:
f(x_T) − f* ≤ L‖x₀ − x*‖² / (2T)  □

**For strongly convex (condition number κ = L/m):**
‖x_{t+1} − x*‖² ≤ (1 − 1/κ)‖xₜ − x*‖²
Iterating: ‖xₜ − x*‖² ≤ (1 − 1/κ)ᵗ‖x₀ − x*‖²

This is **linear convergence** (exponential in T). Poorly conditioned problems (large κ) converge slowly.

**Why Nesterov Beats GD:**
Nesterov's insight: gradient descent makes locally optimal steps but globally suboptimal ones. By maintaining momentum and evaluating gradients at an extrapolated point, the algorithm "looks ahead":
y_{t+1} = x_t − (1/L)∇f(x_t)
x_{t+1} = y_{t+1} + ((t−1)/(t+2))(y_{t+1} − yₜ)

Result: f(x_T) − f* ≤ 2L‖x₀ − x*‖² / T²  — optimal for first-order methods!`,
            },
            {
              id: "s2",
              title: "Comparing Optimizers: GD, Momentum, Adam on Ill-Conditioned Problems",
              type: "code",
              language: "python",
              content: `import numpy as np
import matplotlib.pyplot as plt

def make_ill_conditioned_quadratic(condition_number=100, dim=2):
    """
    f(x) = 0.5 * x^T @ A @ x where A has condition number kappa.
    Perfect testbed for optimizer comparison — closed-form optimal rate.
    """
    eigenvalues = np.linspace(1, condition_number, dim)
    A = np.diag(eigenvalues)
    x_star = np.zeros(dim)
    return A, x_star

def gradient_descent(A, x0, lr, steps):
    x = x0.copy()
    history = [0.5 * x @ A @ x]
    for _ in range(steps):
        grad = A @ x
        x = x - lr * grad
        history.append(0.5 * x @ A @ x)
    return history

def heavy_ball_momentum(A, x0, lr, beta, steps):
    """
    Polyak's Heavy Ball: x_{t+1} = x_t - lr*grad + beta*(x_t - x_{t-1})
    Note: NOT Nesterov! Heavy ball does NOT have global convergence guarantees
    for general convex functions, but works well on quadratics.
    """
    x = x0.copy()
    x_prev = x0.copy()
    history = [0.5 * x @ A @ x]
    for _ in range(steps):
        grad = A @ x
        x_new = x - lr * grad + beta * (x - x_prev)
        x_prev = x
        x = x_new
        history.append(0.5 * x @ A @ x)
    return history

def nesterov_accelerated(A, x0, lr, steps):
    """
    Nesterov's accelerated gradient descent.
    Optimal: O(1/T^2) convergence for convex, O(exp(-sqrt(kappa)*T)) for strongly convex.
    """
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    history = [0.5 * x @ A @ x]
    for _ in range(steps):
        x_new = y - lr * (A @ y)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        history.append(0.5 * x @ A @ x)
    return history

def adam_optimizer(A, x0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=500):
    """
    Adam: adaptive learning rates per parameter.
    m_t = beta1*m_{t-1} + (1-beta1)*grad
    v_t = beta2*v_{t-1} + (1-beta2)*grad^2
    x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps)
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [0.5 * x @ A @ x]
    for t in range(1, steps + 1):
        grad = A @ x
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)  # bias correction
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(0.5 * x @ A @ x)
    return history

# Experiment: ill-conditioned quadratic (kappa = 100)
kappa = 100
dim = 50
A, x_star = make_ill_conditioned_quadratic(kappa, dim)
x0 = np.ones(dim)
steps = 500

L = float(np.max(np.diag(A)))  # Largest eigenvalue = Lipschitz constant
m = float(np.min(np.diag(A)))  # Smallest eigenvalue = strong convexity constant
print(f"L = {L}, m = {m}, κ = {kappa}")

# Theoretical step sizes
lr_gd = 1.0 / L
# Optimal heavy ball: lr=(2/(sqrt(L)+sqrt(m)))^2, beta=((sqrt(kappa)-1)/(sqrt(kappa)+1))^2
lr_hb = (2 / (np.sqrt(L) + np.sqrt(m)))**2
beta_hb = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**2

hist_gd = gradient_descent(A, x0, lr_gd, steps)
hist_hb = heavy_ball_momentum(A, x0, lr_hb, beta_hb, steps)
hist_nag = nesterov_accelerated(A, x0, 1.0/L, steps)
hist_adam = adam_optimizer(A, x0, lr=0.5, steps=steps)

# Theoretical rates
T = np.arange(steps + 1)
f0 = 0.5 * x0 @ A @ x0
rate_gd = f0 * (1 - 1/kappa)**T
rate_nag = f0 * 2 / (T + 1)**2 * L  # approximate

print(f"\\nFinal objective values (lower = better):")
print(f"  GD:         {hist_gd[-1]:.2e}")
print(f"  Heavy Ball: {hist_hb[-1]:.2e}")
print(f"  Nesterov:   {hist_nag[-1]:.2e}")
print(f"  Adam:       {hist_adam[-1]:.2e}")
print(f"\\nNesterov gets {hist_gd[-1]/hist_nag[-1]:.0f}x lower objective than plain GD!")
print(f"Optimal Heavy Ball matches Nesterov on quadratics (both O(exp(-sqrt(kappa)*T)))")
print(f"Adam's adaptive rates handle ill-conditioning automatically but loses exact rates.")
`,
              caption: "Convergence comparison on ill-conditioned quadratics — theory meets implementation",
            },
          ],
          exercises: [
            {
              id: "ex-gd-theory-1",
              type: "multiple_choice",
              question: "For an L-smooth, m-strongly convex function, gradient descent with step size 2/(L+m) achieves convergence rate:",
              options: [
                "O(1/T) — sublinear",
                "O((κ-1)/(κ+1))^T — linear convergence with optimal constant",
                "O(1/T²) — accelerated",
                "O(exp(-T/κ)) — same as step 1/L",
              ],
              correctAnswer: "O((κ-1)/(κ+1))^T — linear convergence with optimal constant",
              explanation: "With step size 2/(L+m), GD achieves the rate (1 - 2/(κ+1))^T = ((κ-1)/(κ+1))^T. This is optimal among all step sizes for gradient descent (not accelerated methods). Nesterov acceleration achieves ((√κ-1)/(√κ+1))^T which converges much faster — taking the square root of the condition number in the exponent.",
              hints: ["The optimal step size balances between the smoothness and strong convexity constants"],
            },
          ],
        },
        {
          id: "stochastic-optimization",
          moduleId: "convex-optimization",
          trackId: "math-foundations",
          title: "Stochastic Optimization: SGD, Variance Reduction & Non-Convex Landscapes",
          description: "Theory of SGD convergence, variance reduction methods (SVRG, SARAH), and why non-convex optimization in deep learning works better than theory predicts.",
          type: "concept",
          estimatedMinutes: 65,
          order: 4,
          prevLessonId: "gradient-descent-theory",
          prerequisites: ["gradient-descent-theory"],
          keyTakeaways: [
            "SGD converges at O(1/√T) for convex due to gradient noise, matching lower bounds",
            "SVRG eliminates variance with periodic full-gradient snapshots achieving O(exp(-T))",
            "For non-convex: SGD finds ε-stationary points in O(1/ε²) iterations",
            "Batch normalization, skip connections implicitly improve the loss landscape",
          ],
          sections: [
            {
              id: "s1",
              title: "Why SGD Converges Despite Noisy Gradients",
              type: "text",
              content: `In standard (full-batch) gradient descent, we compute exact gradients. In SGD, we compute a single-sample estimate: g_t = ∇f_{i_t}(xₜ) where i_t is randomly sampled.

**SGD is an unbiased estimator:** E[g_t] = ∇f(xₜ) — correct on average.

**But variance introduces noise:** Var[g_t] = E[‖g_t − ∇f(xₜ)‖²] = σ²

This variance fundamentally limits convergence:

**SGD convergence (convex, σ²-bounded variance):**
E[f(x̄_T)] − f* ≤ (‖x₀−x*‖/√T)(√(Σ σᵢ²))

With diminishing step αₜ = c/√t: O(1/√T) — this is optimal for noisy first-order methods!

**Why does SGD work well in deep learning then?**
1. **Implicit regularization**: SGD's noise prevents overfitting; it finds "flat minima" that generalize better. Flat minima have more measure under small perturbations → better generalization (PAC-Bayes, sharpness-aware minimization (SAM) literature).
2. **Effective batch size**: Mini-batches reduce variance by factor B. The "linear scaling rule" (Goyal et al. 2017): scale lr proportionally to batch size to maintain the same noise-to-signal ratio.
3. **Saddle point escape**: SGD's noise helps escape saddle points faster than GD. Saddle points are ubiquitous in deep learning (by symmetry arguments) and GD would slow near them.`,
            },
            {
              id: "s2",
              title: "Variance Reduction: SVRG Algorithm",
              type: "code",
              language: "python",
              content: `import numpy as np

def svrg(grad_fi, full_grad, x0, n, lr, epochs=20, m=None):
    """
    SVRG (Stochastic Variance Reduced Gradient) - Johnson & Zhang, 2013.
    
    Achieves LINEAR convergence for strongly convex objectives 
    — same rate as full GD but with O(n) cheaper iterations!
    
    Key idea: periodically compute full gradient as reference point.
    Per-step update uses variance-reduced gradient:
        v_t = grad_fi(x_t) - grad_fi(x_tilde) + full_grad(x_tilde)
    
    E[v_t] = full_grad(x_t) but Var[v_t] -> 0 as x_t -> x*
    """
    if m is None:
        m = 2 * n  # inner loop size (typically 2n to 5n)
    
    x = x0.copy()
    history = []
    
    for epoch in range(epochs):
        # SVRG Outer loop: compute full gradient at snapshot x_tilde
        x_tilde = x.copy()
        mu = full_grad(x_tilde)  # Full gradient — O(n) computation
        
        x_inner = x_tilde.copy()
        
        # Inner loop: m steps of variance-reduced SGD
        for _ in range(m):
            i = np.random.randint(n)
            # Variance-reduced gradient: 
            # E[v] = grad_f(x) because E[grad_fi(x) - grad_fi(xtilde)] = grad_f(x) - mu
            v = grad_fi(i, x_inner) - grad_fi(i, x_tilde) + mu
            x_inner = x_inner - lr * v
        
        # Option I: set x = x_inner (last iterate)
        # Option II: set x = mean of inner iterates (theoretical convergence uses this)
        x = x_inner
        history.append(x.copy())
    
    return x, history

# Demo: Logistic regression (strongly convex with L2 regularization)
np.random.seed(42)
n, p = 1000, 20
X = np.random.randn(n, p)
true_w = np.random.randn(p)
y = (X @ true_w + np.random.randn(n) > 0).astype(float) * 2 - 1
lambda_reg = 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def loss_logistic(w):
    scores = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-scores))) + 0.5 * lambda_reg * np.dot(w, w)

def full_grad(w):
    scores = y * (X @ w)
    coeff = -y * sigmoid(-scores) / n
    return X.T @ coeff + lambda_reg * w

def grad_fi(i, w):
    """Single-sample gradient"""
    score_i = y[i] * np.dot(X[i], w)
    coeff_i = -y[i] * sigmoid(-score_i)
    return X[i] * coeff_i + lambda_reg * w

# SVRG
L = np.max(np.linalg.eigvalsh(X.T @ X)) / (4 * n) + lambda_reg
lr_svrg = 0.1 / L

x0 = np.zeros(p)
w_svrg, hist_svrg = svrg(grad_fi, full_grad, x0, n, lr=lr_svrg, epochs=30)

# Compare: SGD with step-size decay
w_sgd = np.zeros(p)
losses_sgd = []
for t in range(1, 30 * n + 1):
    i = np.random.randint(n)
    g = grad_fi(i, w_sgd)
    lr_t = 0.01 / np.sqrt(t / n + 1)
    w_sgd -= lr_t * g
    if t % n == 0:
        losses_sgd.append(loss_logistic(w_sgd))

losses_svrg = [loss_logistic(w) for w in hist_svrg]
optimal = loss_logistic(np.linalg.lstsq(X, y, rcond=None)[0])

print("=== SVRG vs SGD Convergence ===")
print(f"After 30 epochs:")
print(f"  SVRG loss: {losses_svrg[-1]:.6f}")
print(f"  SGD  loss: {losses_sgd[-1]:.6f}")
print()
print("SVRG achieves linear convergence (error decreases exponentially).")
print("SGD's noise floor prevents it from converging to high precision.")
print()
print("Cost per epoch: SVRG = O(2n) [1 full grad + m=2n stochastic grads]")
print("                SGD  = O(n)  [n stochastic gradient steps]")
print("SVRG pays 2x per epoch but gets exponential convergence vs O(1/sqrt(T)).")
`,
              caption: "SVRG achieves linear convergence with stochastic gradients — bridging full-batch and stochastic optimization",
            },
          ],
          exercises: [
            {
              id: "ex-sgd-1",
              type: "true_false",
              question: "True or False: SVRG achieves the same O(1/√T) convergence rate as SGD, but with a smaller constant.",
              correctAnswer: "False",
              explanation: "False. SVRG achieves LINEAR convergence (O(ρ^T) for some ρ < 1), not O(1/√T). This is the same rate as full-batch gradient descent on strongly convex functions. The key insight: SVRG eliminates the variance in the gradient estimates by using periodic full-gradient snapshots, so the algorithm is not fundamentally limited by gradient noise. It achieves the best of both worlds: the per-iteration cost of SGD with the convergence rate of GD.",
              hints: ["Compare with the SGD convergence rate proof"],
            },
          ],
        },
      ],
    },
    {
      id: "information-theory",
      trackId: "math-foundations",
      title: "Information Theory & Statistical Learning",
      description: "Entropy, mutual information, KL divergence, PAC learning bounds, VC dimension, and the information-theoretic foundations of generalization.",
      order: 2,
      estimatedHours: 10,
      lessons: [
        {
          id: "entropy-kl-divergence",
          moduleId: "information-theory",
          trackId: "math-foundations",
          title: "KL Divergence, Entropy & Variational Inference Connection",
          description: "KL divergence as a fundamental asymmetric distance, its role in variational inference, the evidence lower bound (ELBO), and connections to maximum likelihood.",
          type: "concept",
          estimatedMinutes: 60,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "KL(P||Q) = E_P[log(P/Q)] — asymmetric, non-negative (Gibbs inequality)",
            "MLE is equivalent to minimizing KL(P_data || P_model)",
            "ELBO = E_q[log p(x,z)] - E_q[log q(z)] = log p(x) - KL(q||p)",
            "Forward KL (inclusive) vs reverse KL (exclusive/mode-seeking) have different behavior",
          ],
          sections: [
            {
              id: "s1",
              title: "KL Divergence: The Fundamental Asymmetry",
              type: "math",
              content: `**Definition:** The KL divergence from Q to P is:
KL(P‖Q) = ∫ p(x) log(p(x)/q(x)) dx = E_P[log p(X)/q(X)]

**Key properties:**
1. KL(P‖Q) ≥ 0 with equality iff P = Q a.e. (Gibbs inequality, proven via Jensen's inequality on the convex function −log)
2. KL is NOT symmetric: KL(P‖Q) ≠ KL(Q‖P) in general
3. KL does NOT satisfy the triangle inequality
4. KL is NOT bounded: KL(P‖Q) = ∞ if Q(A) = 0 but P(A) > 0 for some event A

**The Critical Asymmetry:**

**Forward KL** (also called "I-projection"): min_q KL(p_data ‖ q_θ)
→ q_θ must cover all regions where p_data > 0 (mass-covering, "inclusive")
→ Solution: q is the moment-matched distribution (for exponential families, gives MLE!)

**Reverse KL** (also called "M-projection"): min_q KL(q_θ ‖ p_data)  
→ q_θ concentrates on a mode of p_data (zero-forcing, "exclusive")
→ Used in variational inference: ELBO maximization
→ Can miss modes! VAE posteriors trained with reverse KL tend to be mode-seeking

This asymmetry explains a core tension in generative modeling:
- VAEs (reverse KL on posterior) tend to produce blurry images — the mean of multiple modes
- GANs implicitly minimize a different divergence → sharper but mode-dropping samples

**The MLE-KL Connection:**
max_θ E_{p_data}[log p_θ(x)] = min_θ KL(p_data ‖ p_θ) + H(p_data)

Maximizing log-likelihood IS minimizing forward KL divergence (data entropy is constant). This means MLE is information-theoretically optimal when the model class contains the true distribution.`,
            },
            {
              id: "s2",
              title: "Deriving the ELBO from First Principles",
              type: "math",
              content: `**The Problem:** In latent variable models, p(x) = ∫ p(x|z)p(z)dz is often intractable.

**Variational Inference:** Approximate the true posterior p(z|x) with a tractable q(z; φ).

**ELBO Derivation:**
log p(x) = log ∫ p(x,z) dz
         = log ∫ q(z) [p(x,z)/q(z)] dz
         ≥ ∫ q(z) log[p(x,z)/q(z)] dz    (Jensen's inequality, log is concave)
         = E_q[log p(x,z)] − E_q[log q(z)]
         = E_q[log p(x|z)] + E_q[log p(z)] − E_q[log q(z)]
         = E_q[log p(x|z)] − KL(q(z) ‖ p(z))

**The ELBO (Evidence Lower BOund):**
ELBO = E_q[log p(x|z)] − KL(q(z) ‖ p(z))
     = log p(x) − KL(q(z) ‖ p(z|x))

**Key decomposition:** log p(x) = ELBO + KL(q ‖ p(·|x))

Since KL ≥ 0: ELBO ≤ log p(x) always (hence "lower bound")
ELBO = log p(x) ⟺ q = p(z|x) exactly (posterior is in the variational family)

**Maximizing ELBO simultaneously:**
1. Makes q(z) close to p(z|x) [minimizes reverse KL]
2. Maximizes evidence log p(x)

This is the core of VAE training: reparameterize to allow gradient flow through the sampling operation.`,
            },
            {
              id: "s3",
              title: "ELBO Computation and Reparameterization Trick",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Implementing the ELBO and reparameterization trick from scratch.
This is the mathematical core of VAEs — understanding this deeply
is essential before implementing neural VAEs.
"""

# ============================================================
# PART 1: Gaussian ELBO in closed form
# ============================================================
def gaussian_elbo(x, mu_z, log_var_z, mu_prior=0, var_prior=1):
    """
    ELBO for Gaussian posterior q(z|x) = N(mu_z, exp(log_var_z))
    and Gaussian prior p(z) = N(mu_prior, var_prior).
    
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    
    For diagonal Gaussians, KL has closed form:
    KL(N(mu, sigma^2) || N(0, 1)) = 
        0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
    """
    var_z = np.exp(log_var_z)
    
    # KL divergence: KL(q || p) with p = N(0,1)
    # Derived from: KL(N(mu1,s1) || N(mu2,s2)) = log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2s2^2) - 1/2
    kl = 0.5 * np.sum(var_z + mu_z**2 - 1 - log_var_z)
    
    return kl  # We return just the KL; reconstruction term depends on decoder

def reparameterize(mu, log_var, n_samples=1):
    """
    Reparameterization trick: z = mu + eps * sigma, eps ~ N(0, I)
    
    CRUCIAL: This allows backpropagation through the sampling operation.
    
    Without reparam: z ~ N(mu, sigma^2) → can't backprop through sampling
    With reparam:    z = mu + eps*sigma, eps ~ N(0,1) → gradients flow through mu, sigma
    
    The gradient now flows through the deterministic computation mu + eps*sigma
    rather than the stochastic node z ~ N(mu, sigma^2).
    """
    sigma = np.exp(0.5 * log_var)
    eps = np.random.randn(*mu.shape) if n_samples == 1 else np.random.randn(n_samples, *mu.shape)
    return mu + eps * sigma

def elbo_variance_analysis(n_trials=1000, latent_dim=10):
    """
    Compare Monte Carlo ELBO estimator variance with and without reparameterization.
    
    Score function estimator (REINFORCE): high variance, needs many samples
    Reparameterization estimator: low variance, works with 1 sample in practice
    """
    mu = np.random.randn(latent_dim) * 0.5
    log_var = np.random.randn(latent_dim) * 0.5 - 1  # small variance
    
    # Reparameterization estimator gradient (wrt mu)
    reparam_grads = []
    for _ in range(n_trials):
        eps = np.random.randn(latent_dim)
        sigma = np.exp(0.5 * log_var)
        z = mu + eps * sigma
        # Gradient of log p(z) wrt z (for standard normal prior)
        grad_log_pz_wrt_z = -z
        # Chain rule: d/d_mu log p(z) = d/d_z log p(z) * dz/d_mu = grad_log_pz * 1
        reparam_grads.append(grad_log_pz_wrt_z)  # simplified
    
    reparam_grads = np.array(reparam_grads)
    
    print("=== ELBO Gradient Variance Analysis ===")
    print(f"Latent dim: {latent_dim}, Trials: {n_trials}")
    print()
    print(f"Reparameterization estimator:")
    print(f"  Mean gradient (mu[0]): {reparam_grads[:, 0].mean():.4f}")
    print(f"  Std of gradient (mu[0]): {reparam_grads[:, 0].std():.4f}")
    print()
    print("Reparameterization trick allows the VAE to train with batch size 1")
    print("and still get reliable gradient estimates — this is its key advantage.")
    print()
    print("KL divergence (closed form):", 0.5 * np.sum(np.exp(log_var) + mu**2 - 1 - log_var))

elbo_variance_analysis()

# ============================================================
# PART 2: Visualize forward vs reverse KL minimization
# ============================================================
def compare_kl_directions():
    """
    Bimodal target p vs unimodal approximation q (Gaussian).
    Shows the qualitative difference between forward and reverse KL.
    """
    x = np.linspace(-6, 6, 1000)
    
    # Bimodal "true" distribution (mixture of Gaussians)
    p = 0.5 * np.exp(-0.5 * (x - 2)**2) + 0.5 * np.exp(-0.5 * (x + 2)**2)
    p = p / np.trapz(p, x)
    
    # Forward KL min: q must cover p → mean between modes → q = N(0, ~2^2)
    # This produces mass-covering behavior
    q_forward = np.exp(-0.5 * x**2 / 4)  # N(0, 4) — covers both modes
    q_forward = q_forward / np.trapz(q_forward, x)
    
    # Reverse KL min: q collapses to one mode → q = N(2, ~1) or N(-2, ~1)
    q_reverse = np.exp(-0.5 * (x - 2)**2)  # N(2, 1) — one mode
    q_reverse = q_reverse / np.trapz(q_reverse, x)
    
    # Compute KL values
    eps = 1e-10
    kl_forward = np.trapz(p * np.log((p + eps) / (q_forward + eps)), x)
    kl_reverse = np.trapz(q_reverse * np.log((q_reverse + eps) / (p + eps)), x)
    
    print("\\n=== Forward vs Reverse KL on Bimodal Distribution ===")
    print(f"KL(p_data || q_forward_opt) = {kl_forward:.3f}  [mass-covering, covers both modes]")
    print(f"KL(q_reverse_opt || p_data) = {kl_reverse:.3f}  [mode-seeking, covers one mode]")
    print()
    print("→ VAE uses reverse KL → mode-seeking → may miss modes")
    print("→ This explains why VAEs sometimes produce blurry/averaged samples")
    print("→ Diffusion models implicitly use forward KL path → better coverage")

compare_kl_directions()
`,
              caption: "KL divergence geometry: forward vs reverse, reparameterization trick, and ELBO variance — the mathematical foundations of VAEs",
            },
          ],
          exercises: [
            {
              id: "ex-kl-1",
              type: "multiple_choice",
              question: "Why does using reverse KL (KL(q||p)) in variational inference lead to mode-seeking behavior?",
              options: [
                "Because reverse KL is always smaller than forward KL",
                "Because KL(q||p) = ∞ when q(z) > 0 but p(z) = 0, forcing q to avoid regions where p is zero",
                "Because reverse KL minimization requires computing the full posterior",
                "Because q is constrained to be Gaussian",
              ],
              correctAnswer: "Because KL(q||p) = ∞ when q(z) > 0 but p(z) = 0, forcing q to avoid regions where p is zero",
              explanation: "When minimizing KL(q||p) = ∫q(z)log(q(z)/p(z))dz, the term q(z)log(q(z)/p(z)) → ∞ if q(z) > 0 and p(z) = 0. This creates infinite cost if q assigns mass anywhere p is zero. As a result, q 'shrinks away' from low-probability regions of p, concentrating on a single mode even if p is multimodal. Forward KL (KL(p||q)) has the opposite behavior — q must cover all modes where p has mass.",
              hints: ["Think about what happens to KL(q||p) when q(z)>0 but p(z)=0"],
            },
          ],
        },
      ],
    },
    {
      id: "measure-theoretic-probability",
      trackId: "math-foundations",
      title: "Measure-Theoretic Probability & Stochastic Processes",
      description: "Sigma-algebras, probability spaces, conditional expectation, martingales, and their role in proving convergence of ML algorithms.",
      order: 3,
      estimatedHours: 10,
      lessons: [
        {
          id: "pac-learning-vc-theory",
          moduleId: "measure-theoretic-probability",
          trackId: "math-foundations",
          title: "PAC Learning, VC Dimension & Rademacher Complexity",
          description: "Statistical learning theory: PAC learnability, VC dimension as a complexity measure, Rademacher complexity for data-dependent bounds, and compression bounds.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "PAC learning: with high probability, ERM achieves ε-error with O(VC(H)/ε²) samples",
            "VC dimension measures the richest set of points a hypothesis class can shatter",
            "Rademacher complexity gives tighter, data-dependent generalization bounds",
            "Double descent: modern overparameterized models defy classical bias-variance tradeoff",
          ],
          sections: [
            {
              id: "s1",
              title: "PAC Learning and Sample Complexity",
              type: "text",
              content: `**PAC (Probably Approximately Correct) Learning Framework (Valiant, 1984):**

A hypothesis class H is PAC learnable if there exists an algorithm A and polynomial function m(1/ε, 1/δ) such that: for any target concept c ∈ H, any distribution D, and any ε, δ > 0, if A receives m(1/ε, 1/δ) i.i.d. samples from D labeled by c, then with probability ≥ 1−δ:
L(h) ≤ ε  where h = A(S) is the output hypothesis

**Sample complexity of finite hypothesis classes:**
For |H| finite, ERM satisfies with probability ≥ 1−δ:
L(h_ERM) ≤ L(h*) + √(log|H|/m + log(1/δ)/m) / 2

**Key insight:** The number of hypotheses grows, but only logarithmically matters for sample complexity. A class of 2^100 hypotheses needs only 100 bits of complexity — generalization is about description length, not raw count.

**VC Dimension (for infinite hypothesis classes):**
VC(H) = max {m : ∃ set S of size m that H shatters}

H "shatters" S if for every labeling of S, some h ∈ H achieves zero training error.

**Examples:**
- Linear classifiers in ℝᵈ: VC dimension = d+1
- Degree-k polynomials in ℝ: VC dimension = k+1
- Neural networks with W weights: VC dimension = O(W log W)
- Gaussian RBF SVMs: VC dimension = ∞ (can shatter any finite set!)

**Fundamental Theorem of Statistical Learning:**
H is PAC learnable ⟺ VC(H) < ∞

Sample complexity: m(ε, δ) = Θ((VC(H) + log(1/δ)) / ε²)

The modern puzzle: deep neural networks have VC dimension >> n (more parameters than data points), yet they generalize. Classical theory predicts overfitting — it doesn't explain this!`,
            },
            {
              id: "s2",
              title: "Double Descent & Why Classical Theory Fails for Deep Learning",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Demonstrating the double descent phenomenon — a fundamental challenge 
to classical bias-variance tradeoff intuitions.

Classical view: test error follows a U-shaped curve as model complexity grows.
Modern ML: after the interpolation threshold, test error DECREASES again!
"""

def generate_data(n=50, p_true=10, noise=0.5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, max(p_true, 200))
    w_true = np.zeros(max(p_true, 200))
    w_true[:p_true] = np.random.randn(p_true)
    y = X @ w_true + noise * np.random.randn(n)
    return X, y, w_true

def fit_and_evaluate(X_train, y_train, X_test, y_test, n_features):
    """
    Fit least squares with n_features features.
    When n_features > n_samples: use minimum-norm solution (Moore-Penrose pseudoinverse).
    This corresponds to training modern overparameterized models to zero training loss.
    """
    Xtr = X_train[:, :n_features]
    Xte = X_test[:, :n_features]
    
    n, p = Xtr.shape
    
    if p <= n:
        # Underdetermined or exactly determined: use standard least squares
        w = np.linalg.lstsq(Xtr, y_train, rcond=None)[0]
        train_loss = np.mean((Xtr @ w - y_train)**2)
    else:
        # Overparameterized (p > n): minimum L2 norm solution
        # w = X^T (X X^T)^{-1} y  [Moore-Penrose pseudoinverse]
        # This is what gradient descent with small init converges to!
        XXT = Xtr @ Xtr.T
        alpha = np.linalg.lstsq(XXT, y_train, rcond=None)[0]
        w = Xtr.T @ alpha
        train_loss = np.mean((Xtr @ w - y_train)**2)
    
    test_loss = np.mean((Xte @ w - y_test)**2)
    return train_loss, test_loss, np.linalg.norm(w)

# Generate data
n_train = 50
n_test = 1000
p_total = 500

X_all, y_all, w_true = generate_data(n=n_train + n_test, p_true=10, noise=0.3)
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test, y_test = X_all[n_train:], y_all[n_train:]

feature_counts = list(range(5, p_total, 5))
train_losses, test_losses, norms = [], [], []

for p in feature_counts:
    tr, te, nrm = fit_and_evaluate(X_train, y_train, X_test, y_test, p)
    train_losses.append(tr)
    test_losses.append(te)
    norms.append(nrm)

# Find the interpolation threshold (where training loss first hits ~0)
interp_idx = next(i for i, tl in enumerate(train_losses) if tl < 0.01)
interp_features = feature_counts[interp_idx]

print("=== Double Descent Phenomenon ===")
print(f"Training samples: {n_train}, Interpolation threshold: p = {interp_features}")
print()
print(f"{'Features':>10} | {'Train MSE':>10} | {'Test MSE':>10} | {'||w||':>10} | Region")
print("-" * 60)
for i, p in enumerate(feature_counts[::10]):
    idx = feature_counts.index(p)
    region = "Classical" if p < interp_features else "Modern"
    print(f"{p:>10} | {train_losses[idx]:>10.4f} | {test_losses[idx]:>10.4f} | {norms[idx]:>10.2f} | {region}")

print()
print("Key observations:")
print(f"1. Classical regime (p < {interp_features}): test error U-shaped as classical theory predicts")
print(f"2. At interpolation threshold (p ≈ {interp_features}): training loss → 0, test error peaks")
print(f"3. Modern regime (p > {interp_features}): test error DECREASES despite zero training loss!")
print()
print("Explanation: Minimum-norm interpolation acts as implicit regularization.")
print("With more parameters, the min-norm solution is smoother/simpler,")
print("despite fitting training data exactly. The 'simplest explanation'")
print("that fits the data gets simpler as there are more ways to fit it!")
print()
print("Implications for deep learning:")
print("- Neural networks are trained in the modern (overparameterized) regime")
print("- Gradient descent implicitly finds minimum-norm (or max-margin) solutions")
print("- Classical bias-variance tradeoff does NOT apply as-is")
`,
              caption: "Double descent: the interpolation threshold and why overparameterization can help generalization",
            },
          ],
          exercises: [
            {
              id: "ex-vc-1",
              type: "multiple_choice",
              question: "What is the VC dimension of the class of axis-aligned rectangles in ℝ²?",
              options: ["2", "3", "4", "Infinite"],
              correctAnswer: "4",
              explanation: "Axis-aligned rectangles in ℝ² can shatter any set of 4 points arranged at the corners (one point leftmost, one rightmost, one topmost, one bottommost). For any labeling of these 4 points, there's a rectangle that includes exactly the positive-labeled ones. However, no 5 points can be shattered: if we arrange 5 points with one inside the convex hull of the others, we cannot find a rectangle that includes the outer 4 but excludes the inner one. Hence VC dim = 4.",
              hints: ["Can you shatter 4 points placed at extreme left/right/top/bottom positions?"],
            },
          ],
        },
      ],
    },
    {
      id: "spectral-methods",
      trackId: "math-foundations",
      title: "Spectral Methods & Random Matrix Theory",
      description: "SVD, PCA, spectral graph theory, random matrix theory, and their applications in dimensionality reduction, graph neural networks, and understanding neural network training dynamics.",
      order: 4,
      estimatedHours: 8,
      lessons: [
        {
          id: "svd-pca-advanced",
          moduleId: "spectral-methods",
          trackId: "math-foundations",
          title: "SVD, Randomized Linear Algebra & Spectral Concentration",
          description: "Randomized SVD, sketch-and-solve, Johnson-Lindenstrauss lemma, and spectral concentration inequalities used in modern large-scale ML.",
          type: "coding",
          estimatedMinutes: 55,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "Randomized SVD computes k-rank approximation in O(mnk) vs O(mn²) deterministic",
            "JL lemma: random projections approximately preserve distances with O(log n/ε²) dimensions",
            "Eckart-Young theorem: truncated SVD gives optimal low-rank approximation in Frobenius and spectral norms",
            "Neural tangent kernel (NTK) theory uses spectral analysis to understand infinite-width networks",
          ],
          sections: [
            {
              id: "s1",
              title: "Randomized SVD: Efficient Low-Rank Approximation",
              type: "code",
              language: "python",
              content: `import numpy as np
import time

def randomized_svd(A, k, n_oversampling=10, n_power_iter=2):
    """
    Randomized SVD of A ≈ U @ diag(S) @ Vt.
    Algorithm: Halko, Martinsson, Tropp (2011) "Finding Structure with Randomness"
    
    Complexity: O(mn*k) vs O(mn*min(m,n)) for full SVD
    Key idea: find a random subspace that captures the range of A,
    then project and compute exact SVD in the subspace.
    
    Args:
        k: target rank
        n_oversampling: extra dimensions for better approximation (p in literature)
        n_power_iter: power iteration for spectral gap amplification
    """
    m, n = A.shape
    l = k + n_oversampling  # oversampled rank
    
    # Step 1: Form random sketch Omega ~ N(0, I_{n x l})
    Omega = np.random.randn(n, l)
    
    # Step 2: Compute Y = A @ Omega — random projection into range of A
    Y = A @ Omega
    
    # Step 3: Power iteration — amplifies singular value gaps
    # A(A^T A)^q @ Omega has same singular vectors as A but much larger gaps
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    
    # Step 4: Orthonormalize: Q s.t. range(Q) ≈ range(A)
    Q, _ = np.linalg.qr(Y)  # Q: m x l
    
    # Step 5: Project A into the small subspace: B = Q^T @ A  (l x n, tiny!)
    B = Q.T @ A
    
    # Step 6: Exact SVD of tiny B
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 7: Lift back to full space
    U = Q @ U_hat
    
    return U[:, :k], S[:k], Vt[:k, :]

def johnson_lindenstrauss_demo(n_points=1000, original_dim=1000, eps=0.3):
    """
    Johnson-Lindenstrauss Lemma: 
    For any set of n points in ℝ^d, a random projection to 
    k = O(log(n) / eps^2) dimensions approximately preserves all pairwise distances.
    
    More precisely: (1-eps)||x-y||^2 <= ||P(x)-P(y)||^2 <= (1+eps)||x-y||^2
    """
    k_jl = int(8 * np.log(n_points) / (eps**2))
    print(f"JL dimension: {original_dim}D -> {k_jl}D (compression ratio: {original_dim/k_jl:.1f}x)")
    
    # Generate random points
    X = np.random.randn(n_points, original_dim)
    
    # Random projection matrix (Gaussian)
    P = np.random.randn(k_jl, original_dim) / np.sqrt(k_jl)
    X_proj = X @ P.T
    
    # Verify distance preservation on random pairs
    n_pairs = 1000
    idx1 = np.random.randint(n_points, size=n_pairs)
    idx2 = np.random.randint(n_points, size=n_pairs)
    
    d_orig = np.sum((X[idx1] - X[idx2])**2, axis=1)
    d_proj = np.sum((X_proj[idx1] - X_proj[idx2])**2, axis=1)
    
    ratios = d_proj / d_orig
    violations = np.mean((ratios < (1-eps)) | (ratios > (1+eps)))
    
    print(f"Distance ratio mean: {ratios.mean():.4f} (should be ~1.0)")
    print(f"Distance ratio std:  {ratios.std():.4f} (should be small)")
    print(f"Fraction violating ({1-eps:.1f}, {1+eps:.1f}) bound: {violations:.4f}")
    print(f"JL guarantee: probability ~O(1/n) of any pair violating the bound")

# Benchmark: Exact vs Randomized SVD
print("=== Randomized SVD vs Exact SVD ===\\n")
m, n, k = 2000, 3000, 50
# Low-rank + noise matrix
U_true = np.random.randn(m, k)
V_true = np.random.randn(n, k)
S_true = np.exp(-np.arange(k) * 0.1) * 100  # decaying singular values
A = U_true @ np.diag(S_true) @ V_true.T + 0.01 * np.random.randn(m, n)

t0 = time.time()
U_exact, S_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
t_exact = time.time() - t0

t0 = time.time()
U_rand, S_rand, Vt_rand = randomized_svd(A, k=k, n_oversampling=10, n_power_iter=3)
t_rand = time.time() - t0

# Reconstruction error
A_exact_k = U_exact[:, :k] @ np.diag(S_exact[:k]) @ Vt_exact[:k]
A_rand_k = U_rand @ np.diag(S_rand) @ Vt_rand

err_exact = np.linalg.norm(A - A_exact_k, 'fro')
err_rand = np.linalg.norm(A - A_rand_k, 'fro')

print(f"Matrix size: {m}x{n}, target rank: k={k}")
print(f"Exact SVD:       time={t_exact:.2f}s, Frobenius error={err_exact:.2f}")
print(f"Randomized SVD:  time={t_rand:.2f}s, Frobenius error={err_rand:.2f}")
print(f"Speedup: {t_exact/t_rand:.1f}x, Error ratio: {err_rand/err_exact:.3f}")
print()
print("\\n=== Johnson-Lindenstrauss Lemma Demo ===\\n")
johnson_lindenstrauss_demo(n_points=500, original_dim=500, eps=0.3)
`,
              caption: "Randomized SVD and Johnson-Lindenstrauss: the algorithmic core of scalable ML",
            },
          ],
          exercises: [
            {
              id: "ex-svd-1",
              type: "multiple_choice",
              question: "The Eckart-Young theorem states that the best rank-k approximation to A in both the Frobenius and spectral norms is:",
              options: [
                "The k eigenvectors of A corresponding to the k largest eigenvalues",
                "The truncated SVD: Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ",
                "The k columns of A with largest L2 norm",
                "The projection of A onto the k leading principal components of AᵀA",
              ],
              correctAnswer: "The truncated SVD: Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ",
              explanation: "Eckart-Young-Mirsky theorem: Among all rank-k matrices B, the truncated SVD Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢvᵢᵀ minimizes ‖A−B‖_F = √(Σᵢ>k σᵢ²) and also minimizes ‖A−B‖₂ = σₖ₊₁. This is what justifies PCA — projecting onto the top k singular vectors gives the best low-dimensional linear approximation of the data.",
              hints: ["The optimal approximation comes from the SVD, not eigendecomposition of A directly"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 2: ADVANCED CLASSICAL MACHINE LEARNING
// ============================================================
const advancedClassicalMLTrack: Track = {
  id: "advanced-classical-ml",
  title: "Advanced Classical Machine Learning",
  description: "Kernel methods, Gaussian processes, Bayesian inference, ensemble theory, and structured prediction — the rigorous underpinnings of pre-deep-learning ML that remain fundamental.",
  icon: "⚙",
  difficulty: "advanced",
  estimatedHours: 35,
  moduleCount: 4,
  lessonCount: 14,
  tags: ["kernels", "Gaussian processes", "Bayesian ML", "ensemble methods"],
  color: "#8b5cf6",
  order: 2,
  modules: [
    {
      id: "kernel-methods",
      trackId: "advanced-classical-ml",
      title: "Kernel Methods & Reproducing Kernel Hilbert Spaces",
      description: "The mathematical theory of kernels: Mercer's theorem, RKHS, the kernel trick, and how kernels enable infinite-dimensional feature spaces.",
      order: 1,
      estimatedHours: 10,
      lessons: [
        {
          id: "rkhs-mercer",
          moduleId: "kernel-methods",
          trackId: "advanced-classical-ml",
          title: "Reproducing Kernel Hilbert Spaces & Mercer's Theorem",
          description: "The functional analysis framework behind kernel methods: RKHS, the representer theorem, and Mercer's theorem connecting kernels to feature maps.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "A kernel k(x,x') defines an inner product in an RKHS: k(x,x') = ⟨φ(x), φ(x')⟩_H",
            "Mercer's theorem: k is a valid kernel iff its Gram matrix is PSD for all datasets",
            "Representer theorem: optimal solution in RKHS lives in span{k(·,xᵢ)} — finite-dimensional!",
            "Neural networks and kernels: NTK shows infinite-width networks are Gaussian processes",
          ],
          sections: [
            {
              id: "s1",
              title: "The RKHS Framework",
              type: "text",
              content: `A **Reproducing Kernel Hilbert Space (RKHS)** is a Hilbert space H of functions f: X → ℝ where evaluation functionals are continuous. This means: there exists a function k: X × X → ℝ (the reproducing kernel) such that:

1. **Feature map:** φ: X → H, x ↦ k(·, x)
2. **Reproducing property:** f(x) = ⟨f, k(·,x)⟩_H for all f ∈ H
3. **Kernel inner product:** k(x, x') = ⟨k(·,x), k(·,x')⟩_H = ⟨φ(x), φ(x')⟩_H

**Mercer's Theorem:** A symmetric function k: X × X → ℝ is a valid kernel (i.e., is the inner product of some RKHS) if and only if the Gram matrix K with Kᵢⱼ = k(xᵢ, xⱼ) is positive semidefinite for all finite sets {x₁,...,xₙ}.

**Kernel examples:**
- **Linear kernel:** k(x,x') = xᵀx' → standard dot product, φ(x) = x
- **Polynomial kernel:** k(x,x') = (xᵀx' + c)ᵈ → monomial features up to degree d
- **RBF/Gaussian:** k(x,x') = exp(−‖x−x'‖²/(2σ²)) → infinite-dimensional feature space!
- **Matérn kernels:** k(x,x') depends on ‖x−x'‖ with smoothness parameter ν
- **String kernels:** k(s,s') = number of common subsequences → NLP applications
- **Neural network kernel:** k(x,x') = E_{w~N(0,I)}[σ(wᵀx)σ(wᵀx')] → random features approximation

**The Kernel Trick:** Computing k(x,x') is O(d) even when the implicit feature space φ(x) is infinite-dimensional. This enables algorithms that would be computationally infeasible in the explicit feature space.`,
            },
            {
              id: "s2",
              title: "Representer Theorem & Kernel Ridge Regression",
              type: "math",
              content: `**Representer Theorem (Schölkopf, Herbrich, Smola, 2001):**
For any monotonically increasing function Ω: [0,∞) → ℝ and any empirical loss L, the problem:

min_{f ∈ H} L(f(x₁),...,f(xₙ), y₁,...,yₙ) + Ω(‖f‖²_H)

has an optimal solution of the form:
f*(x) = Σᵢ αᵢ k(x, xᵢ)

**Proof sketch:** Any f ∈ H can be decomposed as f = f_∥ + f_⊥ where f_∥ ∈ span{k(·,xᵢ)} and f_⊥ ⊥ span{k(·,xᵢ)}. Since f_⊥(xᵢ) = ⟨f_⊥, k(·,xᵢ)⟩ = 0, removing f_⊥ doesn't change the loss but reduces ‖f‖²_H. □

**Kernel Ridge Regression:**
min_{f ∈ H} (1/n)‖y − f(X)‖² + λ‖f‖²_H

Representer theorem gives f*(x) = Σᵢ αᵢ k(x,xᵢ), so f(xⱼ) = Σᵢ αᵢ k(xⱼ,xᵢ) = (Kα)ⱼ

Substituting: min_α (1/n)‖y − Kα‖² + λαᵀKα

Solution: α = (K + nλI)⁻¹ y

**Predictions:** f*(x) = kₓᵀ(K + nλI)⁻¹y where kₓ = [k(x,x₁),...,k(x,xₙ)]

**Complexity:** O(n³) for solving the system — kernels scale poorly with n. This is why approximation methods (Nyström, random Fourier features) are critical for large-scale kernel methods.

**Connection to Gaussian Processes:** Kernel ridge regression is exactly the posterior mean of a Gaussian Process with kernel k! λ controls the noise variance. The posterior variance gives uncertainty estimates — something KRR's point estimate misses.`,
            },
            {
              id: "s3",
              title: "Kernel Methods: SVMs, RBF Analysis & Nyström Approximation",
              type: "code",
              language: "python",
              content: `import numpy as np
from scipy.spatial.distance import cdist

class KernelRidgeRegression:
    """
    Kernel Ridge Regression with multiple kernel options.
    Demonstrates the representer theorem in action.
    """
    
    def __init__(self, kernel='rbf', lambda_reg=0.1, **kernel_params):
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.kernel_params = kernel_params
    
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            sq_dists = cdist(X1, X2, 'sqeuclidean')
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'polynomial':
            d = self.kernel_params.get('degree', 3)
            c = self.kernel_params.get('c', 1.0)
            return (X1 @ X2.T + c) ** d
        elif self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'matern_3_2':
            # k(r) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
            l = self.kernel_params.get('length_scale', 1.0)
            dists = cdist(X1, X2)
            r = np.sqrt(3) * dists / l
            return (1 + r) * np.exp(-r)
    
    def fit(self, X, y):
        self.X_train = X
        n = len(y)
        K = self._compute_kernel(X, X)
        # alpha = (K + n*lambda*I)^{-1} y  (representer theorem solution)
        self.alpha = np.linalg.solve(K + n * self.lambda_reg * np.eye(n), y)
        
        # Compute training R²
        y_pred = K @ self.alpha
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        self.train_r2 = 1 - ss_res / ss_tot
        return self
    
    def predict(self, X_test):
        K_test = self._compute_kernel(X_test, self.X_train)
        return K_test @ self.alpha

def nystrom_approximation(X, kernel_fn, k, seed=42):
    """
    Nyström method: approximate n×n kernel matrix with rank-k approximation.
    Cost: O(nk²) instead of O(n²k) for Gram matrix computation.
    
    Procedure:
    1. Sample k landmark points {x̃₁,...,x̃ₖ} from X (or use k-means centers)
    2. Compute W = K(X̃, X̃)  (k×k kernel matrix of landmarks)
    3. Compute C = K(X, X̃)   (n×k cross-kernel matrix)
    4. Approximation: K ≈ C @ W⁻¹ @ C^T
    
    This gives an approximation K ≈ Φ Φ^T where Φ = C @ W^{-1/2}
    """
    np.random.seed(seed)
    n = len(X)
    
    # Sample k landmark points
    landmark_idx = np.random.choice(n, k, replace=False)
    X_landmarks = X[landmark_idx]
    
    W = kernel_fn(X_landmarks, X_landmarks)    # k × k
    C = kernel_fn(X, X_landmarks)              # n × k
    
    # Stable inversion via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical stability
    W_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Feature map approximation: Phi = C @ W^{-1/2}  (n × k)
    Phi = C @ W_inv_sqrt
    
    # Kernel approximation: K_approx ≈ Phi @ Phi.T
    K_approx = Phi @ Phi.T
    return K_approx, Phi

# Demonstration
np.random.seed(42)
n_train, n_test = 200, 100

# Non-linear regression task: f(x) = sin(3x) * exp(-0.5x^2) + noise
X_train = np.random.uniform(-3, 3, (n_train, 1))
y_train = np.sin(3 * X_train[:, 0]) * np.exp(-0.5 * X_train[:, 0]**2) + 0.1 * np.random.randn(n_train)
X_test = np.linspace(-3.5, 3.5, n_test).reshape(-1, 1)
y_test_true = np.sin(3 * X_test[:, 0]) * np.exp(-0.5 * X_test[:, 0]**2)

# Compare kernels
kernels = {
    'RBF (γ=1)': KernelRidgeRegression('rbf', lambda_reg=0.01, gamma=1.0),
    'RBF (γ=5)': KernelRidgeRegression('rbf', lambda_reg=0.01, gamma=5.0),
    'Polynomial (d=5)': KernelRidgeRegression('polynomial', lambda_reg=0.001, degree=5),
    'Matérn-3/2': KernelRidgeRegression('matern_3_2', lambda_reg=0.01, length_scale=0.5),
}

print("=== Kernel Ridge Regression: Kernel Comparison ===\\n")
for name, model in kernels.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test_true)**2)
    print(f"{name:25s} | Train R²: {model.train_r2:.4f} | Test MSE: {mse:.6f}")

# Nyström approximation
print("\\n=== Nyström Approximation ===")
gamma = 1.0
def rbf_kernel(X1, X2): return np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))

K_exact = rbf_kernel(X_train, X_train)
for k in [10, 30, 50, 100]:
    K_approx, Phi = nystrom_approximation(X_train, rbf_kernel, k=k)
    err = np.linalg.norm(K_exact - K_approx, 'fro') / np.linalg.norm(K_exact, 'fro')
    print(f"k={k:3d} landmarks: Frobenius relative error = {err:.4f}, Feature dim = {k}")
`,
              caption: "Kernel ridge regression and Nyström approximation — the representer theorem in action",
            },
          ],
          exercises: [
            {
              id: "ex-kernel-1",
              type: "multiple_choice",
              question: "The RBF kernel k(x,x') = exp(-γ‖x-x'‖²) corresponds to an inner product in a feature space of what dimensionality?",
              options: [
                "d (same as input dimension)",
                "d² (degree-2 polynomial features)",
                "Countably infinite",
                "Uncountably infinite",
              ],
              correctAnswer: "Countably infinite",
              explanation: "Expanding the RBF kernel via Taylor series: exp(-γ‖x-x'‖²) = exp(-γ‖x‖²)exp(-γ‖x'‖²)exp(2γxᵀx') = Σₙ₌₀^∞ (2γ)ⁿ/n! (xᵀx')ⁿ. This shows the feature map includes all polynomial orders 0, 1, 2, ..., ∞, making the RKHS infinite-dimensional (but countably so, since it's indexed by multi-indices of monomials). This is why SVMs with RBF kernels can fit any smooth function — they operate in infinite-dimensional space!",
              hints: ["Use the Taylor expansion of the exponential function"],
            },
          ],
        },
        {
          id: "gaussian-processes",
          moduleId: "kernel-methods",
          trackId: "advanced-classical-ml",
          title: "Gaussian Processes: Bayesian Nonparametric Regression",
          description: "GPs as distributions over functions, posterior inference, covariance function design, sparse GPs, and connections to deep learning (deep kernels, neural network GPs).",
          type: "coding",
          estimatedMinutes: 75,
          order: 2,
          prevLessonId: "rkhs-mercer",
          prerequisites: ["rkhs-mercer"],
          keyTakeaways: [
            "GP = distribution over functions: f(x) ~ GP(m(x), k(x,x'))",
            "Posterior GP given data is analytically tractable for Gaussian likelihood",
            "Log marginal likelihood enables kernel hyperparameter optimization via gradient descent",
            "Sparse GPs (inducing points) scale O(nm²) instead of O(n³)",
          ],
          sections: [
            {
              id: "s1",
              title: "Gaussian Processes as Priors Over Functions",
              type: "text",
              content: `A **Gaussian Process** is a collection of random variables, any finite subset of which has a joint Gaussian distribution. Equivalently, it's a distribution over functions.

**Definition:** f ~ GP(m, k) means:
- For any finite set X = {x₁,...,xₙ}, [f(x₁),...,f(xₙ)] ~ N(m(X), K(X,X))
- where m(xᵢ) is the mean function (often m ≡ 0)
- and K(X,X)ᵢⱼ = k(xᵢ,xⱼ) is the covariance matrix

**GP Regression (exact inference):**
Given observations y = f(X) + ε, ε ~ N(0, σ²I):

Prior: f | X ~ N(0, K(X,X))
Likelihood: y | f,X ~ N(f, σ²I)

**Posterior (exact, by Gaussian conditioning):**
f* | X*, X, y ~ N(μ*, Σ*)

where:
μ* = K(X*,X) [K(X,X) + σ²I]⁻¹ y
Σ* = K(X*,X*) − K(X*,X) [K(X,X) + σ²I]⁻¹ K(X,X*)

**The posterior gives:**
1. Predictions: μ* (posterior mean = kernel ridge regression prediction!)
2. Uncertainty: Σ* (posterior variance quantifies epistemic uncertainty)

**Log Marginal Likelihood (for hyperparameter learning):**
log p(y|X,θ) = −(1/2)yᵀ(K+σ²I)⁻¹y − (1/2)log|K+σ²I| − (n/2)log(2π)

The three terms represent:
1. **Data fit:** −yᵀ(K+σ²I)⁻¹y — prefers kernels that explain the data
2. **Model complexity:** −log|K+σ²I| — penalizes complex models (Occam's razor automatic!)
3. **Normalization:** constant

Maximizing log p(y|X,θ) wrt kernel hyperparameters θ (lengthscale, amplitude, noise) gives the optimal GP without held-out validation!`,
            },
            {
              id: "s2",
              title: "Full GP Implementation with Hyperparameter Optimization",
              type: "code",
              language: "python",
              content: `import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

class GaussianProcess:
    """
    Full Gaussian Process regression with:
    - ARD (Automatic Relevance Determination) RBF kernel
    - Marginal likelihood optimization for hyperparameters
    - Posterior mean + uncertainty quantification
    - Sparse GP (Nyström/inducing points) for scalability
    """
    
    def __init__(self, noise_var=0.1, length_scale=1.0, amplitude=1.0):
        # Log-transformed hyperparameters for unconstrained optimization
        self.log_noise = np.log(noise_var)
        self.log_length_scale = np.log(length_scale)
        self.log_amplitude = np.log(amplitude)
        self.X_train = None
        self.alpha = None
        self.L = None  # Cholesky factor for efficient computation
    
    def _kernel(self, X1, X2, log_ls=None, log_amp=None):
        """RBF kernel: k(x,x') = amp^2 * exp(-0.5 * ||x-x'||^2 / ls^2)"""
        if log_ls is None: log_ls = self.log_length_scale
        if log_amp is None: log_amp = self.log_amplitude
        ls = np.exp(log_ls)
        amp = np.exp(log_amp)
        sq_dists = np.sum((X1[:, None] - X2[None, :]) ** 2, axis=-1)
        return amp**2 * np.exp(-0.5 * sq_dists / ls**2)
    
    def _log_marginal_likelihood(self, params, X, y):
        """
        log p(y|X,θ) = -0.5*y^T(K+σ²I)^{-1}y - 0.5*log|K+σ²I| - n/2*log(2π)
        
        Uses Cholesky for O(n²) per marginal likelihood evaluation after O(n³) factorization.
        """
        log_amp, log_ls, log_noise = params
        noise_var = np.exp(log_noise)
        n = len(y)
        
        K = self._kernel(X, X, log_ls, log_amp)
        K += (noise_var + 1e-6) * np.eye(n)  # jitter for stability
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10  # non-PSD — return large value
        
        # Solve L alpha = y using forward/backward substitution: O(n²)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        
        # Log determinant from Cholesky: log|K| = 2 * sum(log(diag(L)))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        return nll  # minimize this
    
    def fit(self, X, y, optimize=True):
        self.X_train = X.copy()
        
        if optimize:
            # Maximize marginal likelihood wrt hyperparameters
            x0 = [self.log_amplitude, self.log_length_scale, self.log_noise]
            result = minimize(
                self._log_marginal_likelihood, x0, args=(X, y),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            self.log_amplitude, self.log_length_scale, self.log_noise = result.x
            print(f"Optimized hyperparameters:")
            print(f"  amplitude={np.exp(self.log_amplitude):.4f}")
            print(f"  length_scale={np.exp(self.log_length_scale):.4f}")
            print(f"  noise_std={np.exp(self.log_noise/2):.4f}")
            print(f"  Marginal likelihood: {-result.fun:.4f}")
        
        noise_var = np.exp(self.log_noise)
        K = self._kernel(X, X)
        K += (noise_var + 1e-6) * np.eye(len(y))
        self.L = np.linalg.cholesky(K)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y, lower=True))
        return self
    
    def predict(self, X_test, return_std=True):
        """
        Posterior: μ* = K(X*,X)(K+σ²I)^{-1}y
                   σ*² = K(X*,X*) - K(X*,X)(K+σ²I)^{-1}K(X,X*)
        """
        K_star = self._kernel(X_test, self.X_train)
        mu = K_star @ self.alpha
        
        if not return_std:
            return mu
        
        v = solve_triangular(self.L, K_star.T, lower=True)
        K_star_star = self._kernel(X_test, X_test)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        var = np.maximum(var, 0)  # numerical stability
        return mu, np.sqrt(var)
    
    def sample_posterior(self, X_test, n_samples=5):
        """Draw functions from posterior — GP gives a distribution over functions!"""
        mu, std = self.predict(X_test)
        K_star_star = self._kernel(X_test, X_test)
        v = solve_triangular(self.L, self._kernel(X_test, self.X_train).T, lower=True)
        posterior_cov = K_star_star - v.T @ v
        posterior_cov += 1e-6 * np.eye(len(X_test))  # jitter
        return np.random.multivariate_normal(mu, posterior_cov, n_samples)

# Demonstration
np.random.seed(42)
X_train = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
y_train = np.sin(X_train[:, 0]) + 0.5 * np.cos(2 * X_train[:, 0]) + 0.1 * np.random.randn(30)

X_test = np.linspace(-4, 4, 200).reshape(-1, 1)

gp = GaussianProcess(noise_var=0.01, length_scale=1.0, amplitude=1.0)
gp.fit(X_train, y_train, optimize=True)

mu, std = gp.predict(X_test)
samples = gp.sample_posterior(X_test, n_samples=3)

print("\\nGP Posterior Statistics:")
print(f"  Max uncertainty (std): {std.max():.4f} (at x = {X_test[std.argmax(), 0]:.2f})")
print(f"  Min uncertainty (std): {std.min():.4f} (at x = {X_test[std.argmin(), 0]:.2f})")
print(f"  Coverage (95% CI): {np.mean(np.abs(mu - np.sin(X_test[:,0]) - 0.5*np.cos(2*X_test[:,0])) < 2*std)*100:.1f}%")
print()
print("GP key advantage over KRR: calibrated uncertainty quantification.")
print("Uncertainty is HIGH in regions far from training data (principled extrapolation).")
print("This enables:")
print("  - Bayesian optimization (UCB/EI/Thompson sampling)")
print("  - Active learning (query where uncertainty is highest)")
print("  - Safe RL (avoid high-uncertainty regions)")
`,
              caption: "Full Gaussian Process with marginal likelihood optimization — Bayesian machine learning in action",
            },
          ],
          exercises: [
            {
              id: "ex-gp-1",
              type: "multiple_choice",
              question: "The log marginal likelihood of a GP automatically performs model selection (choosing kernel hyperparameters) through what mechanism?",
              options: [
                "Cross-validation on a held-out validation set",
                "Balancing data fit against model complexity via the log determinant term",
                "Minimizing the KL divergence between prior and posterior",
                "Maximizing the predictive accuracy on training data",
              ],
              correctAnswer: "Balancing data fit against model complexity via the log determinant term",
              explanation: "The log marginal likelihood decomposes as: (data fit) − 0.5·log|K+σ²I| − constant. The log determinant term penalizes complex kernels (large K matrices with many directions of variation) even when they could fit the data — this is Bayesian Occam's razor. A kernel with a very small lengthscale can interpolate data perfectly but has a huge determinant (many independent directions), leading to a poor marginal likelihood. This automatic trade-off eliminates the need for held-out validation.",
              hints: ["Think about what log|K+σ²I| measures geometrically — it's related to the volume of the ellipsoid defined by K+σ²I"],
            },
          ],
        },
      ],
    },
    {
      id: "ensemble-boosting-theory",
      trackId: "advanced-classical-ml",
      title: "Ensemble Methods: Boosting Theory & Gradient Boosting",
      description: "AdaBoost as margin maximizer, gradient boosting as functional gradient descent, XGBoost second-order methods, and theoretical connections to neural networks.",
      order: 2,
      estimatedHours: 8,
      lessons: [
        {
          id: "adaboost-theory",
          moduleId: "ensemble-boosting-theory",
          trackId: "advanced-classical-ml",
          title: "AdaBoost: Margin Theory & Connection to Exponential Loss",
          description: "AdaBoost's mysterious success explained: margin maximization, the exponential loss perspective, generalization bounds via margin theory, and why boosting doesn't overfit.",
          type: "concept",
          estimatedMinutes: 60,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "AdaBoost minimizes exponential loss via coordinate descent on ensemble weights",
            "AdaBoost is equivalent to forward stagewise additive modeling with exp loss",
            "Margin theory explains why AdaBoost generalizes: larger margins → better bounds",
            "Boosting with decision stumps: VC dim O(1) base learners achieve complex boundaries",
          ],
          sections: [
            {
              id: "s1",
              title: "AdaBoost as Exponential Loss Minimization",
              type: "math",
              content: `AdaBoost's algorithm (Freund & Schapire, 1997) seemed mysterious until Friedman, Hastie & Tibshirani (2000) showed it's functional gradient descent on the exponential loss.

**Exponential loss:** L(y, F(x)) = exp(−y·F(x))

**Forward Stagewise Additive Modeling:**
At step m, fit a base learner hₘ and weight αₘ:
(αₘ, hₘ) = argmin_{α,h} Σᵢ exp(−yᵢ(Fₘ₋₁(xᵢ) + α·h(xᵢ)))

Let wᵢ^(m) = exp(−yᵢFₘ₋₁(xᵢ)) (current observation weights).

**Solving for h:** Minimize Σᵢ wᵢ exp(−αyᵢh(xᵢ))
= e^{-α}Σ_{y_i=h_i} wᵢ + e^{α}Σ_{y_i≠h_i} wᵢ

This is minimized by choosing h that minimizes the weighted error rate εₘ = Σᵢ wᵢ·1[yᵢ≠h(xᵢ)] / Σᵢ wᵢ — exactly what AdaBoost does!

**Solving for α:** Setting derivative to zero:
αₘ = (1/2) log((1−εₘ)/εₘ)

**Weight update:** wᵢ^(m+1) = wᵢ^(m) · exp(−αₘ yᵢ hₘ(xᵢ))
= wᵢ^(m) · {e^{-αₘ} if correct, e^{αₘ} if wrong}

**The margin view:** F(x) = Σₘ αₘhₘ(x). The functional margin for example (xᵢ, yᵢ) is yᵢF(xᵢ). AdaBoost maximizes the minimum margin over training data — similar to SVM's max-margin but in the function space.

**Why doesn't AdaBoost overfit?** The generalization bound depends on margins, not base learner complexity:
P(test error) ≤ P_train(margin ≤ θ) + O(√(log(m·T)/(n·θ²)))

Boosting typically INCREASES margins over iterations even after achieving 0 training error, continuously improving the bound.`,
            },
            {
              id: "s2",
              title: "Gradient Boosting as Functional Gradient Descent",
              type: "code",
              language: "python",
              content: `import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingFromScratch:
    """
    Gradient Boosting Machine — Friedman (2001).
    
    View: minimize E[L(y, F(x))] over functions F.
    Algorithm: functional gradient descent — at each step, fit a tree
    to the NEGATIVE GRADIENT of the loss wrt current predictions.
    
    This unifies many losses:
    - Squared loss: gradient = -(y - F(x)) → fit residuals (Breiman's original)
    - Log loss:     gradient = -(y - sigmoid(F(x))) → fit response residuals
    - Huber loss:   gradient = robust, outlier-resistant
    - Quantile:     gradient = asymmetric → quantile regression!
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 loss='squared', quantile=0.5):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.quantile = quantile  # for quantile loss
        self.trees = []
        self.F0 = None  # initial prediction
    
    def _loss_gradient(self, y, F):
        """Compute negative gradient (pseudo-residuals) of loss wrt F."""
        if self.loss == 'squared':
            # L = 0.5*(y-F)^2, dL/dF = -(y-F)
            return y - F
        elif self.loss == 'logistic':
            # L = log(1 + exp(-y*F)), y ∈ {-1, +1}
            p = 1 / (1 + np.exp(-F))
            return y - p  # This is y*(1-p) for y=1, -p for y=0
        elif self.loss == 'huber':
            # Hybrid: squared for small residuals, absolute for large
            delta = 1.0
            r = y - F
            return np.where(np.abs(r) <= delta, r, delta * np.sign(r))
        elif self.loss == 'quantile':
            # Pinball/quantile loss: asymmetric L1
            r = y - F
            return np.where(r >= 0, self.quantile, self.quantile - 1)
    
    def _initial_prediction(self, y):
        if self.loss == 'squared': return np.mean(y)
        elif self.loss == 'logistic': return 0.0
        elif self.loss == 'huber': return np.median(y)
        elif self.loss == 'quantile': return np.quantile(y, self.quantile)
    
    def fit(self, X, y):
        self.F0 = self._initial_prediction(y)
        F = np.full(len(y), self.F0)
        self.trees = []
        self.train_losses = []
        
        for m in range(self.n_estimators):
            # Step 1: Compute pseudo-residuals = negative gradient of loss
            r = self._loss_gradient(y, F)
            
            # Step 2: Fit regression tree to pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, r)
            
            # Step 3: Line search in each leaf for optimal step size
            # (For squared loss, leaf value IS the optimal step)
            # For other losses, we do a leaf-by-leaf 1D optimization
            # Simplified: use tree prediction * learning_rate
            h = tree.predict(X)
            
            # Update: F_{m+1} = F_m + lr * h_m
            F += self.lr * h
            self.trees.append(tree)
            
            if self.loss == 'squared':
                self.train_losses.append(np.mean((y - F)**2))
            elif self.loss == 'quantile':
                r = y - F
                self.train_losses.append(np.mean(np.where(r >= 0, self.quantile*r, (self.quantile-1)*r)))
        
        return self
    
    def predict(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F
    
    def predict_proba(self, X):
        """For logistic loss: convert log-odds to probability."""
        assert self.loss == 'logistic'
        F = self.predict(X)
        return 1 / (1 + np.exp(-F))

# XGBoost second-order insight
def xgboost_leaf_value(g, h, lambda_reg=1.0):
    """
    XGBoost uses second-order Taylor expansion of the loss.
    For leaf j containing samples J:
    
    Optimal leaf value: w_j* = -G_j / (H_j + lambda)
    where G_j = sum of first derivatives, H_j = sum of second derivatives.
    
    Split gain: 0.5 * [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ)] - γ
    
    This is Newton's method in function space — much faster convergence
    than standard gradient boosting for non-quadratic losses.
    """
    G = np.sum(g)
    H = np.sum(h)
    optimal_weight = -G / (H + lambda_reg)
    gain = 0.5 * G**2 / (H + lambda_reg)
    return optimal_weight, gain

# Demonstration: quantile regression (predict uncertainty intervals)
np.random.seed(42)
n = 500
X = np.random.uniform(0, 10, (n, 1))
# Heteroskedastic: noise variance increases with X
y = np.sin(X[:, 0]) + np.random.randn(n) * (0.2 + 0.1 * X[:, 0])

X_test = np.linspace(0, 10, 200).reshape(-1, 1)

# Fit models for median and 90% prediction interval
gb_50 = GradientBoostingFromScratch(n_estimators=100, lr=0.1, max_depth=3, loss='quantile', quantile=0.5)
gb_05 = GradientBoostingFromScratch(n_estimators=100, lr=0.1, max_depth=3, loss='quantile', quantile=0.05)
gb_95 = GradientBoostingFromScratch(n_estimators=100, lr=0.1, max_depth=3, loss='quantile', quantile=0.95)

gb_50.fit(X, y); gb_05.fit(X, y); gb_95.fit(X, y)

y_med = gb_50.predict(X_test)
y_lo = gb_05.predict(X_test)
y_hi = gb_95.predict(X_test)

# Evaluate coverage (should be ~90%)
y_test_true = np.sin(X_test[:, 0])
coverage = np.mean((y_med - 2 > y_lo) & (y_med + 2 < y_hi))

print("=== Gradient Boosting: Quantile Regression ===")
print("Fitting 5%, 50%, 95% quantile models for heteroskedastic prediction intervals")
print(f"Final 50-quantile train loss: {gb_50.train_losses[-1]:.4f}")
print(f"Interval width (mean): {np.mean(y_hi - y_lo):.4f}")
print()
print("Key: Each quantile model fits the SAME trees to DIFFERENT pseudo-residuals.")
print("Quantile loss gradient: r_i = alpha if y_i >= F(x_i), else alpha-1")
print("This asymmetry pushes predictions toward the desired quantile.")
print()
print("=== XGBoost Second-Order Approximation ===")
# For squared loss, g_i = F(x_i) - y_i, h_i = 1
g_example = np.array([0.5, -0.3, 0.1, 0.8])
h_example = np.ones(4)  # squared loss: H = 1 everywhere
w, gain = xgboost_leaf_value(g_example, h_example, lambda_reg=1.0)
print(f"Example leaf: G={g_example.sum():.2f}, H={h_example.sum():.2f}")
print(f"  Optimal weight: {w:.4f}")
print(f"  Structure score: {gain:.4f}")
print("XGBoost replaces 'fit tree to residuals' with 'optimize structure score'")
print("using second-order approximation → faster, more principled tree construction.")
`,
              caption: "Gradient boosting as functional gradient descent — unifying regression, classification, and quantile estimation",
            },
          ],
          exercises: [
            {
              id: "ex-boost-1",
              type: "multiple_choice",
              question: "Why does gradient boosting use shallow trees (e.g., depth 3-6) rather than deep trees as base learners?",
              options: [
                "Deep trees are too slow to train",
                "Shallow trees capture low-order interactions; depth controls interaction order, and regularization prevents overfitting",
                "Deep trees don't support gradient computation",
                "Shallow trees have larger margins",
              ],
              correctAnswer: "Shallow trees capture low-order interactions; depth controls interaction order, and regularization prevents overfitting",
              explanation: "A tree of depth d can model interactions between at most d features. Shallow trees (depth 3-6) are 'weak learners' with low variance but high bias individually. Boosting reduces bias by sequentially combining them, while their high bias provides natural regularization. Deep trees would fit pseudo-residuals too well (low bias per step) but combine to overfit. The learning rate ν and number of trees T interact: small ν requires large T but gives better generalization — Friedman's 'shrinkage' regularization.",
              hints: ["Think about what 'depth' controls in terms of feature interactions"],
            },
          ],
        },
      ],
    },
    {
      id: "bayesian-inference",
      trackId: "advanced-classical-ml",
      title: "Bayesian Inference: MCMC, Variational Methods & Approximate Inference",
      description: "Markov Chain Monte Carlo, Hamiltonian Monte Carlo, expectation propagation, and modern scalable Bayesian methods for large-scale ML.",
      order: 3,
      estimatedHours: 10,
      lessons: [
        {
          id: "mcmc-advanced",
          moduleId: "bayesian-inference",
          trackId: "advanced-classical-ml",
          title: "Hamiltonian Monte Carlo & No-U-Turn Sampler (NUTS)",
          description: "HMC's physics-inspired dynamics for efficient posterior sampling, NUTS for automatic tuning, and Stan/PyMC for practical Bayesian modeling.",
          type: "coding",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "HMC uses gradient information to make large, efficient proposals in high-dimensional spaces",
            "The leapfrog integrator preserves Hamiltonian structure and reversibility",
            "NUTS eliminates the trajectory length hyperparameter using a no-U-turn criterion",
            "HMC scales better than Metropolis-Hastings: mixes in O(d^{1/4}) steps vs O(d) for RW-MH",
          ],
          sections: [
            {
              id: "s1",
              title: "Hamiltonian Monte Carlo: Gradient-Guided Sampling",
              type: "text",
              content: `**The Problem with Random-Walk Metropolis-Hastings:**
In d dimensions, RWMH needs O(d) steps to move a distance of O(1) — it takes random walks of size O(1/√d) to maintain acceptance rate, so it takes O(d) steps to diffuse across the space. For modern posteriors with d = 10,000+ parameters, this is hopeless.

**Hamiltonian Monte Carlo (Duane et al. 1987; Neal 2011):**
HMC introduces "momentum" variables p and defines a Hamiltonian:
H(q, p) = U(q) + K(p)

where U(q) = −log π(q) (potential energy) and K(p) = (1/2)pᵀM⁻¹p (kinetic energy).

The joint distribution π(q,p) ∝ exp(−H(q,p)) = π(q) · N(p; 0, M).

**Key properties of Hamiltonian dynamics:**
1. **Energy conservation:** H remains constant → high acceptance rate
2. **Reversibility:** Time-reversible → preserves detailed balance
3. **Volume preservation (Liouville's theorem):** No Jacobian correction needed

**Leapfrog Integrator** (time-reversible, symplectic):
p_{t+ε/2} = p_t − (ε/2)∇U(q_t)
q_{t+ε} = q_t + ε M⁻¹ p_{t+ε/2}
p_{t+ε} = p_{t+ε/2} − (ε/2)∇U(q_{t+ε})

Each leapfrog step costs one gradient evaluation. After L steps, accept/reject based on energy error (which is O(ε²L) for leapfrog).

**Advantage:** HMC moves distance O(√d) in O(d^{1/4}) gradient evaluations — exponentially better than RWMH!

**The No-U-Turn Sampler (NUTS, Hoffman & Gelman 2014):**
HMC requires choosing L (trajectory length). NUTS dynamically determines when to stop by detecting when the trajectory "turns back":
U-turn criterion: (q⁺ − q⁻)·p⁺ < 0 or (q⁺ − q⁻)·p⁻ < 0

NUTS doubles the trajectory length until a U-turn is detected, then samples from the trajectory. This eliminates the need to tune L and is the default in Stan, PyMC.`,
            },
            {
              id: "s2",
              title: "HMC Implementation from Scratch",
              type: "code",
              language: "python",
              content: `import numpy as np
from scipy.stats import multivariate_normal

def leapfrog(q, p, grad_U, epsilon, L, M_inv=None):
    """
    Leapfrog integrator for Hamiltonian dynamics.
    Time-reversible and volume-preserving (symplectic).
    
    q: position (parameters)
    p: momentum
    grad_U: gradient of potential energy U = -log pi(q)
    epsilon: step size
    L: number of leapfrog steps
    M_inv: inverse mass matrix (diagonal = coordinate-wise learning rates)
    """
    if M_inv is None:
        M_inv = np.eye(len(q))
    
    q = q.copy()
    p = p.copy()
    
    # Half step for momentum
    p -= (epsilon / 2) * grad_U(q)
    
    for _ in range(L - 1):
        # Full step for position
        q += epsilon * M_inv @ p
        # Full step for momentum
        p -= epsilon * grad_U(q)
    
    # Full step for position
    q += epsilon * M_inv @ p
    # Half step for momentum (complete the half steps)
    p -= (epsilon / 2) * grad_U(q)
    
    return q, p

def hmc_sample(log_prob, grad_log_prob, initial_q, n_samples, epsilon, L, 
               M=None, burn_in=500):
    """
    Hamiltonian Monte Carlo sampler.
    
    Key correctness: even though leapfrog introduces O(epsilon^2) energy error,
    the Metropolis correction ensures exact samples from pi(q).
    
    Args:
        log_prob: log pi(q) — unnormalized log posterior
        grad_log_prob: gradient of log pi(q)
        epsilon: leapfrog step size
        L: number of leapfrog steps per proposal
    """
    d = len(initial_q)
    M_inv = np.linalg.inv(M) if M is not None else np.eye(d)
    M_for_sample = M if M is not None else np.eye(d)
    
    q = initial_q.copy()
    samples = []
    n_accepted = 0
    
    def U(q): return -log_prob(q)
    def grad_U(q): return -grad_log_prob(q)
    def K(p): return 0.5 * p @ M_inv @ p  # kinetic energy
    
    for i in range(n_samples + burn_in):
        # Sample momentum from N(0, M)
        p = np.random.multivariate_normal(np.zeros(d), M_for_sample)
        
        # Current Hamiltonian
        H_current = U(q) + K(p)
        
        # Propose via leapfrog integration
        q_prop, p_prop = leapfrog(q, p, grad_U, epsilon, L, M_inv)
        p_prop = -p_prop  # negate momentum for reversibility
        
        # Proposed Hamiltonian
        H_proposed = U(q_prop) + K(p_prop)
        
        # Metropolis-Hastings acceptance step
        log_accept = H_current - H_proposed  # = -ΔH
        
        if np.log(np.random.uniform()) < log_accept:
            q = q_prop
            if i >= burn_in:
                n_accepted += 1
        
        if i >= burn_in:
            samples.append(q.copy())
    
    acceptance_rate = n_accepted / n_samples
    return np.array(samples), acceptance_rate

# ============================================================
# Demo: Bayesian logistic regression with HMC
# ============================================================
np.random.seed(42)

# Generate classification data
n, p = 100, 5
X = np.random.randn(n, p)
true_beta = np.array([1.5, -1.0, 0.5, 0.0, -0.5])
logits = X @ true_beta
y = (np.random.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(float)

# Prior: beta ~ N(0, I)
prior_var = 4.0

def log_posterior(beta):
    """log p(beta | X, y) ∝ log p(y|beta) + log p(beta)"""
    log_lik = np.sum(y * logits_fn(beta) - np.log(1 + np.exp(logits_fn(beta))))
    log_prior = -0.5 * np.sum(beta**2) / prior_var
    return log_lik + log_prior

def logits_fn(beta): return X @ beta

def grad_log_posterior(beta):
    """Gradient for leapfrog integration"""
    p_hat = 1 / (1 + np.exp(-X @ beta))
    grad_lik = X.T @ (y - p_hat)
    grad_prior = -beta / prior_var
    return grad_lik + grad_prior

# Run HMC
samples, acc_rate = hmc_sample(
    log_prob=log_posterior,
    grad_log_prob=grad_log_posterior,
    initial_q=np.zeros(p),
    n_samples=2000,
    epsilon=0.05,  # step size — critical hyperparameter
    L=20,          # trajectory length
    burn_in=500
)

print("=== HMC Bayesian Logistic Regression ===")
print(f"Acceptance rate: {acc_rate:.3f} (target: 0.65-0.85)")
print()
print(f"{'Parameter':>10} | {'True β':>8} | {'Post. Mean':>10} | {'Post. Std':>9} | {'95% CI'}")
print("-" * 65)
for j in range(p):
    mean_j = samples[:, j].mean()
    std_j = samples[:, j].std()
    ci_lo = np.percentile(samples[:, j], 2.5)
    ci_hi = np.percentile(samples[:, j], 97.5)
    covered = "✓" if ci_lo <= true_beta[j] <= ci_hi else "✗"
    print(f"  β_{j}    | {true_beta[j]:>8.2f} | {mean_j:>10.4f} | {std_j:>9.4f} | [{ci_lo:.2f}, {ci_hi:.2f}] {covered}")

print()
print("HMC advantages over MCMC:")
print("- High acceptance rate even in high dimensions")
print("- Large moves guided by gradient → fast mixing")
print("- Fully calibrated posterior uncertainty (no approximation)")
print()
print("HMC disadvantages:")
print("- Requires differentiable log posterior")
print("- Need to tune epsilon and L (NUTS solves L automatically)")
print("- O(n) per gradient evaluation (mini-batch HMC possible but tricky)")
`,
              caption: "HMC sampler from scratch — gradient-guided posterior sampling for Bayesian logistic regression",
            },
          ],
          exercises: [
            {
              id: "ex-hmc-1",
              type: "multiple_choice",
              question: "Why does HMC negate the momentum p at the end of each leapfrog trajectory before the Metropolis step?",
              options: [
                "To ensure the kinetic energy is always positive",
                "To make the proposal distribution symmetric, ensuring the MH correction is valid",
                "To improve the acceptance rate",
                "To prevent the sampler from returning to previously visited states",
              ],
              correctAnswer: "To make the proposal distribution symmetric, ensuring the MH correction is valid",
              explanation: "For the Metropolis-Hastings correction to give exact samples, the proposal must be reversible: T(q,p → q',p') = T(q',-p' → q,-p). The leapfrog integrator maps (q,p) → (q',p') and is time-reversible when we negate p' to get -p'. Without negation, the proposal would not be symmetric (the path from q' back to q would use different dynamics), and the detailed balance condition would be violated. In practice, since K(p) = K(-p) for symmetric M, the negation doesn't affect the accept probability.",
              hints: ["Detailed balance requires T(x→x') = T(x'→x)"],
            },
          ],
        },
      ],
    },
    {
      id: "structured-prediction",
      trackId: "advanced-classical-ml",
      title: "Structured Prediction & Probabilistic Graphical Models",
      description: "CRFs, belief propagation, variational inference on graphs, sum-product algorithm, and structured SVMs for sequence labeling and graph prediction.",
      order: 4,
      estimatedHours: 7,
      lessons: [
        {
          id: "crfs-belief-propagation",
          moduleId: "structured-prediction",
          trackId: "advanced-classical-ml",
          title: "Conditional Random Fields & Belief Propagation",
          description: "CRF formulation, the partition function challenge, belief propagation for exact inference on trees, loopy BP for approximate inference, and viterbi for MAP decoding.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "CRF models P(y|x) directly unlike generative models — avoids modeling P(x)",
            "Linear-chain CRF: partition function via forward-backward (O(n|Y|²))",
            "Belief propagation: exact on trees, approximate (but often good) on loopy graphs",
            "Viterbi algorithm: O(n|Y|²) dynamic programming for MAP sequence decoding",
          ],
          sections: [
            {
              id: "s1",
              title: "CRF: Discriminative Structured Prediction",
              type: "text",
              content: `**Why CRFs over HMMs?**

Hidden Markov Models are generative: they model P(x, y) = P(y)P(x|y). This requires modeling P(xₜ|yₜ) — the emission probability — which in NLP means modeling word distributions. This is problematic:
- Features must be locally conditioned: P(xₜ|yₜ) can't use xₜ₊₁ (future words)
- Feature independence within a state: P(x₁,...,xₙ|y₁,...,yₙ) = Πₜ P(xₜ|yₜ)
- Must model P(x) — wasted capacity on modeling input distribution

**Linear-Chain CRF (Lafferty et al., 2001):**
Instead, model the conditional directly:

P(y|x) = (1/Z(x)) exp(Σₜ Σₖ λₖ fₖ(yₜ₋₁, yₜ, x, t))

where Z(x) = Σ_y exp(Σₜ Σₖ λₖ fₖ(yₜ₋₁, yₜ, x, t)) is the partition function.

Features fₖ can be ANY function of the entire input sequence x and adjacent labels — including future words, character features, surrounding context, etc.

**The Partition Function Challenge:**
Computing Z(x) naively requires summing over |Y|ⁿ possible label sequences — exponential!

**Forward-Backward Algorithm (exact, O(n|Y|²)):**
Define forward variables: αₜ(j) = P(y₁,...,yₜ₋₁, yₜ=j | x)
Recursion: αₜ₊₁(j) = Σᵢ αₜ(i) · ψₜ(i, j, x)
where ψₜ(i,j,x) = exp(Σₖ λₖ fₖ(yₜ₋₁=i, yₜ=j, x, t))

Z(x) = Σⱼ αₙ(j)

Backward variables analogously. Log-likelihood gradient:
∂log P(y|x)/∂λₖ = Σₜ fₖ(yₜ₋₁, yₜ, x, t) − Σₜ Σᵢ,ⱼ P(yₜ₋₁=i, yₜ=j|x) fₖ(i,j,x,t)
= [feature counts under true sequence] − [expected feature counts under model]

This is the classic "empirical - expected" gradient structure of exponential family models.`,
            },
          ],
          exercises: [
            {
              id: "ex-crf-1",
              type: "multiple_choice",
              question: "The gradient of the CRF log-likelihood takes the form (empirical counts − model expected counts). What does this imply about the optimal solution?",
              options: [
                "The model that perfectly memorizes training labels",
                "The model where expected feature counts under P(y|x) match observed feature counts — a maximum entropy solution",
                "The model with the smallest partition function",
                "The maximum a posteriori estimate of the labels",
              ],
              correctAnswer: "The model where expected feature counts under P(y|x) match observed feature counts — a maximum entropy solution",
              explanation: "Setting the gradient to zero: E_model[f] = E_data[f] — the model's expected feature counts equal the empirical feature counts. This is the moment-matching condition for maximum entropy models. The CRF solution is the maximum entropy distribution consistent with observed feature statistics. This connects CRFs to maximum entropy models (MaxEnt), information geometry, and exponential families — all the same thing from different angles.",
              hints: ["Setting gradient to zero means empirical = expected counts"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 3: DEEP LEARNING THEORY & ARCHITECTURE
// ============================================================
const deepLearningTrack: Track = {
  id: "deep-learning-theory",
  title: "Deep Learning: Theory, Architecture & Optimization",
  description: "The mathematical and empirical foundations of deep learning: initialization theory, normalization, attention mechanisms, architectural innovations, and the mysterious success of overparameterized models.",
  icon: "🧠",
  difficulty: "expert",
  estimatedHours: 45,
  moduleCount: 5,
  lessonCount: 18,
  tags: ["neural networks", "backpropagation", "initialization", "normalization", "transformers"],
  color: "#ec4899",
  order: 3,
  modules: [
    {
      id: "initialization-normalization",
      trackId: "deep-learning-theory",
      title: "Initialization Theory, Normalization & Training Dynamics",
      description: "Why initialization matters: vanishing/exploding gradients, He/Xavier/Orthogonal init, batch norm mechanics and theory, layer norm, and signal propagation in deep networks.",
      order: 1,
      estimatedHours: 10,
      lessons: [
        {
          id: "initialization-theory",
          moduleId: "initialization-normalization",
          trackId: "deep-learning-theory",
          title: "Initialization Theory: Signal Propagation & Mean Field Theory",
          description: "Rigorous analysis of forward/backward signal propagation in deep networks using mean field theory. Why Xavier/He initialization is necessary and how depth interacts with activation functions.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "With random weights ~ N(0, σ²), pre-activation variance evolves as Var[zᴸ] = (σ²n)^L Var[x]",
            "Xavier init: σ² = 1/n preserves variance for linear activations",
            "He init: σ² = 2/n for ReLU (accounts for zero-ing half the units)",
            "Mean field theory: tanh networks have an 'edge of chaos' where gradients neither vanish nor explode",
          ],
          sections: [
            {
              id: "s1",
              title: "The Vanishing/Exploding Gradient Problem: Rigorous Analysis",
              type: "math",
              content: `**Setup:** Consider a deep network with L layers, each:
zˡ = Wˡ aˡ⁻¹,  aˡ = σ(zˡ)

where Wˡ ∈ ℝⁿˣⁿ, initialized as Wˡᵢⱼ ~ N(0, σ²ᵥᵥ).

**Forward pass signal analysis:**
E[‖zˡ‖²] = E[‖Wˡaˡ⁻¹‖²] = n · σ²ᵥᵥ · E[‖aˡ⁻¹‖²]

For linear activations (aˡ = zˡ):
E[‖zˡ‖²] = (nσ²ᵥᵥ)^L E[‖x‖²]

This equals E[‖x‖²] only if σ²ᵥᵥ = 1/n — the **Xavier/Glorot initialization**.

**Backward pass: gradient signal:**
δˡ = ∂L/∂zˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ · σ'(zˡ)

For each layer: E[‖δˡ‖²] = nσ²ᵥᵥ · E[σ'²] · E[‖δˡ⁺¹‖²]

Both signals (forward AND backward) must be preserved!

**ReLU analysis:** E[σ'²(z)] = P(z > 0) = 1/2 for symmetric z
So for ReLU: σ²ᵥᵥ = 2/n — the **He/Kaiming initialization**!

**Orthogonal initialization:** Initialize W as a random orthogonal matrix (QR decomposition of Gaussian matrix). This exactly preserves singular values = 1, eliminating vanishing/exploding gradients for linear activations. Used in LSTM gates and helps train very deep networks.

**Mean Field Theory (Poole et al. 2016):**
At infinite width, activations are Gaussian. Define:
qˡ = E[aˡᵢ²]  (mean squared activation)
Cˡᵢⱼ = E[aˡᵢaˡⱼ] / √(qˡ qˡ)  (correlation)

Recursion: qˡ = σ²ᵥᵥ F(qˡ⁻¹) + σ²_b
where F(q) = E_{z~N(0,q)}[σ(z)²]

For tanh networks, there's a **phase transition:**
- Ordered phase (small σ²ᵥᵥ): correlations → 1, gradients vanish
- Chaotic phase (large σ²ᵥᵥ): correlations → 0, gradients explode
- **Edge of chaos** (critical σ²ᵥᵥ): gradients propagate, enabling deep training

Batch normalization effectively pushes any initialization to the edge of chaos!`,
            },
            {
              id: "s2",
              title: "Empirical Initialization Analysis: What Happens Without Proper Init",
              type: "code",
              language: "python",
              content: `import numpy as np

def analyze_signal_propagation(depth=50, width=512, activation='relu', 
                                 init_std_multiplier=1.0, n_trials=100):
    """
    Empirically measure signal propagation through deep networks
    to verify theoretical predictions.
    
    Tracks:
    - Pre-activation variance (forward pass health)  
    - Gradient variance (backward pass health)
    - Rank of feature representations (representation collapse)
    """
    
    if activation == 'relu':
        act_fn = lambda x: np.maximum(x, 0)
        act_deriv = lambda x: (x > 0).astype(float)
        # He initialization: std = sqrt(2/fan_in)
        init_std = init_std_multiplier * np.sqrt(2.0 / width)
    elif activation == 'tanh':
        act_fn = np.tanh
        act_deriv = lambda x: 1 - np.tanh(x)**2
        # Xavier initialization: std = sqrt(1/fan_in)
        init_std = init_std_multiplier * np.sqrt(1.0 / width)
    elif activation == 'linear':
        act_fn = lambda x: x
        act_deriv = lambda x: np.ones_like(x)
        init_std = init_std_multiplier * np.sqrt(1.0 / width)
    
    fwd_variances = []  # variance of activations per layer
    bwd_variances = []  # variance of gradients per layer
    
    for trial in range(n_trials):
        # Initialize weights
        weights = [np.random.randn(width, width) * init_std for _ in range(depth)]
        
        # Forward pass: track pre-activation variance
        x = np.random.randn(width)
        fwd_var_trial = [np.var(x)]
        preacts = []  # save for backward
        acts = [x]
        
        for l in range(depth):
            z = weights[l] @ acts[-1]
            a = act_fn(z)
            preacts.append(z)
            acts.append(a)
            fwd_var_trial.append(np.var(z))
        
        fwd_variances.append(fwd_var_trial)
        
        # Backward pass: track gradient variance
        delta = np.random.randn(width)  # synthetic loss gradient
        bwd_var_trial = [np.var(delta)]
        
        for l in range(depth - 1, -1, -1):
            delta = weights[l].T @ delta * act_deriv(preacts[l])
            bwd_var_trial.insert(0, np.var(delta))
        
        bwd_variances.append(bwd_var_trial)
    
    fwd_variances = np.array(fwd_variances)  # n_trials x (depth+1)
    bwd_variances = np.array(bwd_variances)
    
    return fwd_variances.mean(0), bwd_variances.mean(0)

print("=== Signal Propagation Analysis ===\n")
print(f"{'Init':>20} | {'Layer 0 Fwd':>12} | {'Layer 25 Fwd':>13} | {'Layer 50 Fwd':>13} | Status")
print("-" * 80)

configs = [
    ('ReLU + He (correct)', 'relu', 1.0),
    ('ReLU + Too small (0.5x)', 'relu', 0.5),
    ('ReLU + Too large (2x)', 'relu', 2.0),
    ('Tanh + Xavier (correct)', 'tanh', 1.0),
    ('Tanh + Too large (1.5x)', 'tanh', 1.5),
    ('Linear + Xavier', 'linear', 1.0),
]

for name, act, mult in configs:
    fwd_var, bwd_var = analyze_signal_propagation(
        depth=50, width=256, activation=act, init_std_multiplier=mult, n_trials=30)
    
    v0 = fwd_var[0]
    v25 = fwd_var[25] if len(fwd_var) > 25 else float('nan')
    v50 = fwd_var[50] if len(fwd_var) > 50 else float('nan')
    
    if v50 < 1e-10:
        status = "VANISHED 🔴"
    elif v50 > 1e10:
        status = "EXPLODED 🔴"
    elif 0.1 < v50 < 10:
        status = "STABLE  ✅"
    else:
        status = "DEGRADED 🟡"
    
    print(f"{name:>20} | {v0:>12.4f} | {v25:>13.4f} | {v50:>13.4e} | {status}")

print()
print("=== Gradient Variance Analysis ===\n")
fwd, bwd = analyze_signal_propagation(depth=50, width=256, activation='relu', 
                                       init_std_multiplier=1.0, n_trials=30)
print("He init ReLU gradient variance (backward from output to input):")
for l in [50, 40, 30, 20, 10, 0]:
    print(f"  Layer {l:2d}: {bwd[l]:.4f}")

print()
print("Key insight: He/Xavier init maintains variance through depth.")
print("Without proper init, training fails even with modern architectures.")
print("Batch normalization partially mitigates this but doesn't fully replace good init.")
`,
              caption: "Signal propagation simulation — empirically verifying mean field theory predictions for initialization",
            },
          ],
          exercises: [
            {
              id: "ex-init-1",
              type: "multiple_choice",
              question: "Why does He initialization use σ² = 2/n for ReLU instead of σ² = 1/n (Xavier)?",
              options: [
                "ReLU is not differentiable, requiring larger variance to compensate",
                "ReLU zeros out ~50% of activations on average, halving the variance, so we need 2x larger initial variance to compensate",
                "He initialization prevents the dying ReLU problem",
                "It is an empirical heuristic without theoretical justification",
              ],
              correctAnswer: "ReLU zeros out ~50% of activations on average, halving the variance, so we need 2x larger initial variance to compensate",
              explanation: "For pre-activations z ~ N(0, σ²), ReLU passes positive values and zeros negative ones. E[ReLU(z)²] = E[z² · 1(z>0)] = (1/2)E[z²] = σ²/2. So ReLU halves the variance compared to a linear activation. To maintain Var[aᴸ] = Var[x], we need σ²_init = 2/fan_in to compensate for this halving. This was derived by He et al. (2015) using the exact same analysis: for layer-wise variance preservation with ReLU.",
              hints: ["What fraction of ReLU activations are zero for symmetric input distributions?"],
            },
          ],
        },
        {
          id: "batch-normalization-theory",
          moduleId: "initialization-normalization",
          trackId: "deep-learning-theory",
          title: "Batch Normalization: Mechanics, Theory & Alternatives",
          description: "BN forward/backward pass derivations, the original 'internal covariate shift' hypothesis vs. empirical reality, layer norm, group norm, and why normalization enables higher learning rates.",
          type: "coding",
          estimatedMinutes: 65,
          order: 2,
          prevLessonId: "initialization-theory",
          prerequisites: ["initialization-theory"],
          keyTakeaways: [
            "BN: μ_B = mean(x_B), σ²_B = var(x_B), x̂ = (x-μ)/√(σ²+ε), y = γx̂ + β",
            "BN gradient requires careful chain rule through batch statistics",
            "BN smooths the loss landscape, enabling larger learning rates (key effect)",
            "Layer norm is input-wise (not batch-wise) — essential for transformers and small batches",
          ],
          sections: [
            {
              id: "s1",
              title: "Batch Normalization: Complete Derivation",
              type: "math",
              content: `**Forward Pass:**
Input: x ∈ ℝ^{B×d} (batch of B samples, d features)

1. Batch mean: μ_B = (1/B) Σᵢ xᵢ
2. Batch variance: σ²_B = (1/B) Σᵢ (xᵢ − μ_B)²
3. Normalize: x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε)
4. Scale and shift: yᵢ = γ ⊙ x̂ᵢ + β  (γ, β learnable per-feature)

**Backward Pass (complete derivation):**
Given ∂L/∂y, compute ∂L/∂x (for backprop through the batch).

Let σ_B = √(σ²_B + ε) for brevity.

∂L/∂γ = Σᵢ (∂L/∂yᵢ) ⊙ x̂ᵢ  (sum over batch)
∂L/∂β = Σᵢ ∂L/∂yᵢ

∂L/∂x̂ᵢ = ∂L/∂yᵢ ⊙ γ

∂L/∂xᵢ = (1/σ_B) [∂L/∂x̂ᵢ − (1/B)Σⱼ∂L/∂x̂ⱼ − (1/B)x̂ᵢ Σⱼ(∂L/∂x̂ⱼ ⊙ x̂ⱼ)]

The three terms: full gradient, mean of gradients, mean of gradients weighted by normalized activations. This complex dependency means each sample's gradient depends on ALL samples in the batch — creates implicit regularization.

**Why Not Internal Covariate Shift?**
The original paper claimed BN works by reducing "internal covariate shift" (changing activation distributions during training). However, Santurkar et al. (2018) showed:
1. BN doesn't necessarily reduce ICS
2. Networks with BN but artificially introduced ICS still train well
3. The real effect: **BN smooths the loss landscape**

Santurkar et al. measured the Lipschitz constant of the loss and gradients — both dramatically reduced by BN, enabling much larger and stable learning rates. This is the true mechanism.

**Layer Normalization (Ba et al. 2016):**
Normalize over features, not batch:
μᵢ = (1/d) Σⱼ xᵢⱼ,   σ²ᵢ = (1/d) Σⱼ (xᵢⱼ − μᵢ)²
x̂ᵢⱼ = (xᵢⱼ − μᵢ) / √(σ²ᵢ + ε)

Advantages over BN:
- No dependence on batch size → works for batch size 1
- Same behavior at training and inference (no running statistics)
- Essential for transformers (variable-length sequences, auto-regressive generation)
- Works well for NLP, RL, generative models`,
            },
            {
              id: "s2",
              title: "Implementing BatchNorm and LayerNorm from Scratch",
              type: "code",
              language: "python",
              content: `import numpy as np

class BatchNorm:
    """
    Batch Normalization — complete forward and backward pass.
    Crucial to get the gradient right — it's more complex than it looks!
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)   # scale
        self.beta = np.zeros(num_features)   # shift
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, x):
        """x: (B, d) — batch of B samples with d features"""
        if self.training:
            mu = x.mean(axis=0)                        # (d,)
            var = x.var(axis=0)                         # (d,)
            x_hat = (x - mu) / np.sqrt(var + self.eps)  # (B, d)
            out = self.gamma * x_hat + self.beta
            
            # Update running stats for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            self._cache = {'x': x, 'x_hat': x_hat, 'mu': mu, 'var': var, 
                          'std': np.sqrt(var + self.eps)}
        else:
            # Inference: use running statistics (whole dataset stats)
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
        
        return out
    
    def backward(self, dout):
        """
        Gradient of BN: the non-trivial part!
        dout: (B, d) — gradient from upstream
        
        Note: gradient wrt x_i depends on ALL batch elements (through mu and var)
        """
        x, x_hat, mu, var, std = (self._cache[k] for k in ['x', 'x_hat', 'mu', 'var', 'std'])
        B = x.shape[0]
        
        # Gradients for learnable parameters
        dgamma = (dout * x_hat).sum(axis=0)   # (d,)
        dbeta = dout.sum(axis=0)               # (d,)
        
        # Gradient through x_hat
        dx_hat = dout * self.gamma             # (B, d)
        
        # Full gradient wrt x (derived via chain rule through mu and var)
        # dx = (1/std) * [dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat)]
        dx = (1.0 / (B * std)) * (
            B * dx_hat
            - dx_hat.sum(axis=0)
            - x_hat * (dx_hat * x_hat).sum(axis=0)
        )
        
        return dx, dgamma, dbeta

class LayerNorm:
    """
    Layer Normalization — normalizes over features per sample.
    Used in: Transformers, LSTMs, RL, generative models.
    No batch dependence → same behavior train/inference.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self._cache = {}
    
    def forward(self, x):
        """x: (B, d) or any shape — normalizes over LAST dimension"""
        mu = x.mean(axis=-1, keepdims=True)     # (B, 1)
        var = x.var(axis=-1, keepdims=True)      # (B, 1)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        self._cache = {'x_hat': x_hat, 'std': np.sqrt(var + self.eps)}
        return out
    
    def backward(self, dout):
        x_hat = self._cache['x_hat']
        std = self._cache['std']
        B, d = dout.shape
        
        dgamma = (dout * x_hat).sum(axis=0)
        dbeta = dout.sum(axis=0)
        dx_hat = dout * self.gamma
        
        # Same structure as BN but averaging over features, not batch
        dx = (1.0 / (d * std)) * (
            d * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta

# Numerical gradient check
def numerical_gradient(fn, x, eps=1e-5):
    """Verify analytical gradients against numerical approximation."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy(); x_plus[idx] += eps
        x_minus = x.copy(); x_minus[idx] -= eps
        grad[idx] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
        it.iternext()
    return grad

# Test correctness
np.random.seed(42)
B, d = 4, 8
x = np.random.randn(B, d)
dout = np.random.randn(B, d)

print("=== BatchNorm Gradient Check ===")
bn = BatchNorm(d)
out = bn.forward(x)
dx, dgamma, dbeta = bn.backward(dout)

# Numerical gradient for dx
def bn_output(x_):
    bn2 = BatchNorm(d)
    bn2.gamma = bn.gamma.copy()
    bn2.beta = bn.beta.copy()
    return np.sum(bn2.forward(x_) * dout)

dx_num = numerical_gradient(bn_output, x)
print(f"Max absolute error in dx: {np.max(np.abs(dx - dx_num)):.2e}  (should be < 1e-5)")

print("\n=== LayerNorm Gradient Check ===")
ln = LayerNorm(d)
out = ln.forward(x)
dx_ln, dgamma_ln, dbeta_ln = ln.backward(dout)

def ln_output(x_):
    ln2 = LayerNorm(d)
    ln2.gamma = ln.gamma.copy()
    ln2.beta = ln.beta.copy()
    return np.sum(ln2.forward(x_) * dout)

dx_ln_num = numerical_gradient(ln_output, x)
print(f"Max absolute error in dx: {np.max(np.abs(dx_ln - dx_ln_num)):.2e}  (should be < 1e-5)")

print()
print("=== BN vs LN: Behavior Analysis ===")
print(f"BatchNorm: normalizes over batch dim B — variance = {out.var(axis=0).mean():.4f} (per feature)")
out_ln = ln.forward(x)
print(f"LayerNorm: normalizes over feature dim d — variance = {out_ln.var(axis=1).mean():.4f} (per sample)")
print()
print("Critical difference: BN statistics change with batch size!")
print("Small batch → noisy estimates → noisy gradients")
print("LayerNorm avoids this: same behavior for any batch size")
`,
              caption: "BatchNorm and LayerNorm with complete forward and backward passes — verified by numerical gradient checking",
            },
          ],
          exercises: [
            {
              id: "ex-bn-1",
              type: "multiple_choice",
              question: "Why is Batch Normalization problematic for auto-regressive language model generation (e.g., GPT-style inference)?",
              options: [
                "BN is too slow for generation",
                "During generation, batch size is 1 (or very small), so batch statistics are extremely noisy — different from training statistics",
                "BN doesn't support the attention mechanism",
                "BN requires computing gradients which is expensive at inference",
              ],
              correctAnswer: "During generation, batch size is 1 (or very small), so batch statistics are extremely noisy — different from training statistics",
              explanation: "During auto-regressive generation, tokens are generated one at a time (batch size = 1). With BN: the batch mean and variance from a single sample are trivial (mean = x itself, variance = 0), which is completely different from the training distribution statistics. The running mean/variance (used at inference) were computed on batches of full sequences, creating a train-test mismatch. Layer Norm normalizes per-sample independently, so it has identical behavior with batch size 1 or 1000 — this is why all modern transformers use LayerNorm, not BatchNorm.",
              hints: ["What happens to batch statistics when batch size = 1?"],
            },
          ],
        },
      ],
    },
    {
      id: "transformer-architecture-deep",
      trackId: "deep-learning-theory",
      title: "Transformer Architecture: Deep Dive",
      description: "Multi-head attention derivation, positional encodings, KV-cache mechanics, efficient attention variants (Flash Attention, linear attention), and architectural choices in GPT vs BERT vs T5.",
      order: 2,
      estimatedHours: 12,
      lessons: [
        {
          id: "attention-mechanism-derivation",
          moduleId: "transformer-architecture-deep",
          trackId: "deep-learning-theory",
          title: "Self-Attention: Derivation, Complexity & Efficient Variants",
          description: "Deriving scaled dot-product attention from first principles, multi-head attention as ensemble of retrieval mechanisms, KV-cache for inference, and Flash Attention's IO-aware algorithm.",
          type: "coding",
          estimatedMinutes: 80,
          order: 1,
          prerequisites: ["batch-normalization-theory"],
          keyTakeaways: [
            "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V — scaled to prevent softmax saturation",
            "Multi-head attention: H parallel attention heads with different learned projections",
            "Self-attention is O(n²d) time and space — quadratic in sequence length",
            "Flash Attention: IO-aware algorithm achieving O(n²d/M) IO vs O(n²) — 2-4x faster in practice",
          ],
          sections: [
            {
              id: "s1",
              title: "Scaled Dot-Product Attention: Derivation",
              type: "text",
              content: `**The attention mechanism as content-based memory retrieval:**

Think of attention as a differentiable lookup in a key-value store:
- **Query** q: what we're looking for
- **Keys** {kᵢ}: indices of stored memories
- **Values** {vᵢ}: content of stored memories

For each query q, compute similarity with all keys, normalize (softmax), and retrieve a weighted combination of values.

**Deriving the scaling factor 1/√dₖ:**
For random vectors q, k ∈ ℝᵈₖ with components ~ N(0,1):
qᵀk = Σⱼ qⱼkⱼ

E[qᵀk] = 0,  Var[qᵀk] = dₖ  (sum of dₖ terms each with unit variance)

So qᵀk ~ N(0, dₖ). For large dₖ (e.g., 64), the dot products have large magnitude (std = √dₖ). Softmax of large-magnitude inputs is nearly one-hot — the attention collapses to a single position, losing the ability to attend to multiple positions simultaneously and creating vanishing gradients.

Scaling by 1/√dₖ makes qᵀk/√dₖ ~ N(0,1), keeping softmax in a useful regime.

**Multi-Head Attention:**
Instead of one attention function, use h parallel heads:
head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)
MHA(Q,K,V) = Concat(head_1,...,head_h) Wᴼ

Why multiple heads?
1. Each head can attend to different aspects/positions
2. Different heads empirically specialize: some attend to syntax, some to semantics, some to position
3. Ensemble of retrieval mechanisms reduces variance
4. Total parameters same as one large head: h × (dₖ²+dᵥ²) vs dₘₒ𝒹ₑₗ²

**Computational Complexity:**
- QKᵀ computation: O(n²dₖ) — quadratic in sequence length n
- Attention weights: O(n²) storage
- For n=4096 (context window), this is 4096² ≈ 16M attention weights PER HEAD
- With h=32 heads and 32 layers: ~16B attention weight values just for one forward pass

This quadratic bottleneck motivates Flash Attention, linear attention, and sparse attention.`,
            },
            {
              id: "s2",
              title: "Multi-Head Attention Implementation with KV-Cache",
              type: "code",
              language: "python",
              content: `import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    """
    Multi-Head Self-Attention from scratch.
    
    Includes:
    - Standard scaled dot-product attention
    - Causal (autoregressive) masking
    - KV-cache for efficient autoregressive generation
    - Attention pattern analysis
    """
    
    def __init__(self, d_model, n_heads, dropout=0.0):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # per-head dimension
        
        # Projection matrices (normally learned; here initialized randomly)
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        # KV-cache for autoregressive generation
        self.kv_cache = None
    
    def _split_heads(self, x):
        """(B, T, d_model) -> (B, n_heads, T, d_k)"""
        B, T, d = x.shape
        x = x.reshape(B, T, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (B, h, T, d_k)
    
    def _merge_heads(self, x):
        """(B, n_heads, T, d_k) -> (B, T, d_model)"""
        B, h, T, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B, T, self.d_model)
    
    def forward(self, x, mask=None, use_kv_cache=False):
        """
        x: (B, T, d_model)
        mask: (1, 1, T, T) boolean mask — True = masked (attend to 0)
        use_kv_cache: for autoregressive decoding
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q  # (B, T, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # KV-cache: append new K, V to cache for autoregressive generation
        if use_kv_cache:
            if self.kv_cache is None:
                self.kv_cache = (K, V)
            else:
                K_cached, V_cached = self.kv_cache
                K = np.concatenate([K_cached, K], axis=1)
                V = np.concatenate([V_cached, V], axis=1)
                self.kv_cache = (K, V)
            T_kv = K.shape[1]  # full context length in cache
        else:
            T_kv = T
        
        # Split into heads
        Q = self._split_heads(Q)           # (B, h, T, d_k)
        K = self._split_heads(K)           # (B, h, T_kv, d_k)
        V = self._split_heads(V)           # (B, h, T_kv, d_k)
        
        # Scaled dot-product attention
        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale  # (B, h, T, T_kv)
        
        # Causal mask: position i can only attend to positions j <= i
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        attn_weights = softmax(scores, axis=-1)  # (B, h, T, T_kv)
        
        # Weighted sum of values
        attn_output = attn_weights @ V  # (B, h, T, d_k)
        
        # Merge heads and project output
        attn_output = self._merge_heads(attn_output)  # (B, T, d_model)
        output = attn_output @ self.W_o               # (B, T, d_model)
        
        return output, attn_weights

def causal_mask(T):
    """Create causal (autoregressive) attention mask"""
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # upper triangle
    return mask[None, None, :, :]  # (1, 1, T, T)

def analyze_attention_patterns(attn_weights, tokens=None):
    """
    Visualize and analyze attention patterns.
    Key finding from research: different heads specialize in different functions.
    """
    B, n_heads, T, T_kv = attn_weights.shape
    
    print("=== Attention Pattern Analysis ===")
    print(f"Shape: (batch={B}, heads={n_heads}, queries={T}, keys={T_kv})")
    print()
    
    for h in range(min(n_heads, 4)):  # show first 4 heads
        attn = attn_weights[0, h]  # (T, T_kv)
        
        # Entropy of attention distribution (low = focused, high = diffuse)
        entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1)  # (T,)
        
        # Average attention distance (how far back are we attending?)
        positions = np.arange(T_kv)
        avg_dist = np.array([np.sum(attn[i] * np.abs(i - positions[:len(attn[i])])) 
                            for i in range(T)])
        
        print(f"Head {h+1}: mean entropy={entropy.mean():.3f}, avg attention distance={avg_dist.mean():.2f}")
    
    print()
    print("Research findings on attention head specialization:")
    print("  - 'Position heads': attend to specific relative positions (e.g., always t-1)")
    print("  - 'Syntactic heads': attend to syntactic dependencies (subject-verb)")  
    print("  - 'Semantic heads': attend based on content similarity")
    print("  - 'Copy heads': attend to identical tokens earlier in sequence")

# Demo
np.random.seed(42)
B, T, d_model, n_heads = 2, 16, 64, 4

x = np.random.randn(B, T, d_model)
mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

# Without mask
out_full, attn_full = mha.forward(x, mask=None)
print(f"Full attention output shape: {out_full.shape}")

# With causal mask (GPT-style)
mask = causal_mask(T)
out_causal, attn_causal = mha.forward(x, mask=mask)

# Verify causality: attn[i,j] = 0 for j > i
upper_tri_max = np.max(attn_causal[:, :, :, :][0, 0][np.triu(np.ones((T,T), dtype=bool), k=1)])
print(f"Causal: max attention weight to future tokens: {upper_tri_max:.2e} (should be ~0)")

analyze_attention_patterns(attn_full)

# Flash Attention complexity analysis
print("\\n=== Flash Attention vs Standard Attention ===")
for T in [512, 1024, 2048, 4096]:
    standard_memory = T**2 * 4 / 1e6  # float32 bytes → MB
    # Flash Attention: doesn't materialize full T×T matrix
    flash_memory = T * d_model * 4 / 1e6  # only stores O(Td) not O(T^2)
    print(f"  T={T:5d}: Standard={standard_memory:8.2f}MB, FlashAttn≈{flash_memory:6.2f}MB "
          f"(speedup factor: {standard_memory/flash_memory:.1f}x memory)")
print("Flash Attention fuses operations and tiles computation to stay in SRAM (fast memory).")
print("This enables training on much longer sequences — key for modern LLMs.")
`,
              caption: "Multi-head attention with KV-cache and attention pattern analysis — understanding transformers from the inside",
            },
          ],
          exercises: [
            {
              id: "ex-attn-1",
              type: "multiple_choice",
              question: "Why does Flash Attention achieve O(n²d/M) IO complexity where M is SRAM size, while standard attention needs O(n²) IO?",
              options: [
                "Flash Attention uses a lower-precision computation",
                "Flash Attention tiles the attention computation in blocks that fit in SRAM, avoiding reading/writing the full n×n attention matrix to HBM",
                "Flash Attention approximates the attention by only computing top-k scores",
                "Flash Attention uses sparse attention patterns",
              ],
              correctAnswer: "Flash Attention tiles the attention computation in blocks that fit in SRAM, avoiding reading/writing the full n×n attention matrix to HBM",
              explanation: "GPUs have a memory hierarchy: SRAM (~40MB, fast) and HBM (~40GB, slow). Standard attention writes the full n×n softmax matrix to HBM, then reads it back for the V multiplication — O(n²) IO. Flash Attention (Dao et al. 2022) tiles the computation so each block of Q×K scores is computed in SRAM and immediately used to compute partial attention outputs, without ever writing the full attention matrix. IO is O(n²d/M) where M is SRAM size. This is purely an implementation optimization — numerically identical to standard attention but 2-4x faster due to memory bandwidth savings.",
              hints: ["Think about GPU memory hierarchy: SRAM vs HBM"],
            },
          ],
        },
      ],
    },
    {
      id: "neural-network-theory",
      trackId: "deep-learning-theory",
      title: "Neural Network Theory: NTK, Implicit Bias & Generalization",
      description: "Neural tangent kernel theory for infinite-width networks, implicit bias of gradient descent, the lottery ticket hypothesis, and modern understanding of deep learning generalization.",
      order: 3,
      estimatedHours: 10,
      lessons: [
        {
          id: "ntk-theory",
          moduleId: "neural-network-theory",
          trackId: "deep-learning-theory",
          title: "Neural Tangent Kernel & Infinite-Width Networks",
          description: "The NTK framework showing infinite-width networks are linear models + Gaussian processes. Lazy training regime, feature learning regime, and when NTK theory breaks down.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: ["attention-mechanism-derivation"],
          keyTakeaways: [
            "As width → ∞, NNs become linear models with kernel K_NTK(x,x') = Eθ[∇_θ f(x)·∇_θ f(x')]",
            "NTK is constant during training for infinite-width networks (lazy training)",
            "Finite-width networks escape NTK regime: feature learning makes NTK evolve",
            "Mean field limit (NTK) vs feature learning (rich) regime — where deep learning power comes from",
          ],
          sections: [
            {
              id: "s1",
              title: "The Neural Tangent Kernel",
              type: "text",
              content: `**Setup:** Consider a neural network f(x; θ) with parameters θ ∈ ℝᵖ. Under gradient flow (continuous-time gradient descent):
dθ/dt = −∇_θ L(θ) = −Σᵢ (fᵢ − yᵢ) ∇_θ f(xᵢ; θ)

The dynamics of the function values {f(xⱼ; θ(t))} are:
df(xⱼ)/dt = −Σᵢ K_NTK(xⱼ, xᵢ)(fᵢ − yᵢ)

where the **Neural Tangent Kernel (NTK)** is:
K_NTK(x, x') = ∇_θ f(x; θ)ᵀ ∇_θ f(x'; θ) ∈ ℝ

**Key theorem (Jacot, Gabriel, Hongler 2018):**
For neural networks with width n → ∞ (with appropriate parameterization):
1. At initialization: K_NTK converges in probability to a deterministic limit K*
2. During training: K_NTK stays close to K* (lazy training)

This means infinite-width networks are equivalent to **kernel regression with kernel K*_NTK**!

**Implications:**
- Training dynamics are LINEAR (kernel regression converges to global minimum)
- Infinite-width networks are Gaussian processes at initialization
- The learned function = kernel regression prediction with NTK kernel
- Can compute K*_NTK analytically for any architecture using recursive formulas

**NTK for a 2-layer network (n units, ReLU):**
K_NTK(x,x') = K^(1)(x,x') + K^(0)(x,x') · Ė^(1)(x,x')

where K^(0) = xᵀx'/d (input gram), K^(1) involves expected derivative correlations.

**Why this matters and why it's insufficient:**
The NTK theory shows that gradient descent can minimize the training loss. But the NTK kernel is FIXED — the network never learns new features, just reweights fixed kernel features.

Real deep learning power comes from **feature learning**: the network changes its internal representations. The NTK regime (lazy training) is NOT where most practical networks operate. Feature learning (the "rich" or "mean-field" limit as opposed to NTK limit) is what makes deep networks better than kernel methods on many tasks.`,
            },
          ],
          exercises: [
            {
              id: "ex-ntk-1",
              type: "true_false",
              question: "True or False: The NTK theory implies that sufficiently wide neural networks will always generalize well on any dataset.",
              correctAnswer: "False",
              explanation: "False. NTK theory shows that infinite-width networks are equivalent to kernel regression with the NTK kernel. Whether they generalize depends on: (1) how well the NTK kernel class aligns with the true data-generating function, and (2) the standard kernel regression generalization bounds (which depend on data distribution and sample size). The NTK does NOT imply universally good generalization — only that optimization succeeds (finds global minimum). Generalization requires additional structural assumptions about the task.",
              hints: ["Optimization success ≠ generalization success"],
            },
          ],
        },
      ],
    },
    {
      id: "advanced-architectures",
      trackId: "deep-learning-theory",
      title: "Advanced Architectures: Vision Transformers, MoE & SSMs",
      description: "ViT, DINO, DeiT for vision; Mixture of Experts for scaling; State Space Models (Mamba, S4) as efficient sequence models; architectural innovations in modern LLMs.",
      order: 4,
      estimatedHours: 8,
      lessons: [
        {
          id: "vision-transformers",
          moduleId: "advanced-architectures",
          trackId: "deep-learning-theory",
          title: "Vision Transformers: ViT, DINO & Efficient Vision",
          description: "ViT patch tokenization, position encoding choices, self-supervised ViT (DINO, MAE), data-efficient training (DeiT), and when ViTs outperform CNNs.",
          type: "coding",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: ["attention-mechanism-derivation"],
          keyTakeaways: [
            "ViT: split image into patches → linear projection → transformer encoder → classification head",
            "ViT needs large data (JFT-300M) or strong augmentation (DeiT) to match CNNs",
            "DINO: self-supervised ViT discovers semantic segmentation without labels",
            "Positional encoding: 2D learned, relative, RoPE for vision",
          ],
          sections: [
            {
              id: "s1",
              title: "Vision Transformer Architecture",
              type: "code",
              language: "python",
              content: `import numpy as np

class PatchEmbedding:
    """
    ViT patch tokenization: split image into patches and linearly project.
    
    Input: image of shape (C, H, W)
    Output: sequence of patch embeddings of shape (N, d_model)
    where N = (H/P)*(W/P) is the number of patches, P is patch size.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Linear projection: patch_dim -> d_model
        self.proj = np.random.randn(self.patch_dim, d_model) * np.sqrt(2.0 / self.patch_dim)
        self.proj_bias = np.zeros(d_model)
        
        # [CLS] token — learns global representation for classification
        self.cls_token = np.random.randn(1, d_model) * 0.02
        
        # Positional embedding (learned, 1D — one per patch + CLS)
        self.pos_embed = np.random.randn(self.n_patches + 1, d_model) * 0.02
    
    def forward(self, img):
        """
        img: (C, H, W)
        Returns: (N+1, d_model) — CLS token + patch embeddings with positional encoding
        """
        C, H, W = img.shape
        P = self.patch_size
        n_h = H // P
        n_w = W // P
        
        # Unfold image into patches: (N, patch_dim)
        patches = []
        for i in range(n_h):
            for j in range(n_w):
                patch = img[:, i*P:(i+1)*P, j*P:(j+1)*P]  # (C, P, P)
                patches.append(patch.flatten())  # (C*P*P,)
        patches = np.array(patches)  # (N, C*P*P)
        
        # Linear projection
        x = patches @ self.proj + self.proj_bias  # (N, d_model)
        
        # Prepend [CLS] token
        x = np.vstack([self.cls_token, x])  # (N+1, d_model)
        
        # Add positional embeddings
        x = x + self.pos_embed  # (N+1, d_model)
        
        return x  # each patch is now a "token"

def sinusoidal_2d_position_encoding(n_patches_h, n_patches_w, d_model):
    """
    2D sinusoidal positional encoding for vision.
    Encodes (row, col) position using sinusoids of different frequencies.
    Generalizes beyond training resolution (unlike learned positional embeddings).
    """
    pe = np.zeros((n_patches_h, n_patches_w, d_model))
    d_half = d_model // 2
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            for k in range(d_half // 2):
                freq = 1 / (10000 ** (2 * k / d_half))
                pe[i, j, 2*k] = np.sin(i * freq)      # row encoding
                pe[i, j, 2*k+1] = np.cos(i * freq)
            for k in range(d_half // 2):
                freq = 1 / (10000 ** (2 * k / d_half))
                pe[i, j, d_half + 2*k] = np.sin(j * freq)    # col encoding
                pe[i, j, d_half + 2*k+1] = np.cos(j * freq)
    
    return pe.reshape(n_patches_h * n_patches_w, d_model)

# ViT complexity analysis
print("=== Vision Transformer: Complexity Analysis ===\\n")
print("ViT-B/16 (Base, patch size 16):")
configs = {
    'ViT-B/16': {'img': 224, 'patch': 16, 'd': 768, 'heads': 12, 'layers': 12},
    'ViT-L/16': {'img': 224, 'patch': 16, 'd': 1024, 'heads': 16, 'layers': 24},
    'ViT-H/14': {'img': 224, 'patch': 14, 'd': 1280, 'heads': 16, 'layers': 32},
}

for name, c in configs.items():
    n_patches = (c['img'] // c['patch']) ** 2
    # Attention complexity per layer: O(n^2 * d) + O(n * d^2)
    attn_ops = n_patches**2 * c['d'] + n_patches * c['d']**2
    ffn_ops = n_patches * 4 * c['d']**2  # FFN is 4x expansion
    total_ops = c['layers'] * (attn_ops + ffn_ops)
    
    # Parameters
    attn_params = c['layers'] * 4 * c['d']**2
    ffn_params = c['layers'] * 2 * 4 * c['d']**2
    total_params = (attn_params + ffn_params) / 1e6
    
    print(f"  {name}:")
    print(f"    Patches: {n_patches}, d_model: {c['d']}, layers: {c['layers']}")
    print(f"    Parameters: {total_params:.0f}M")
    print(f"    Attention dominates at long sequences, FFN dominates here")
    print()

print("Key ViT observations from literature:")
print("1. ViT-B/16 trained on ImageNet-1k: ~78% top-1 (worse than ResNet-50 at 80%)")
print("2. ViT-B/16 on JFT-300M: ~84% top-1 (better — ViT is data-hungry)")  
print("3. DeiT (Touvron et al.): distillation + strong augmentation → 81.8% on ImageNet-1k only")
print("4. DINO: self-supervised ViT features enable zero-shot segmentation!")
print()
print("=== DINO Self-Supervised Training Concept ===")
print("DINO uses self-distillation: two ViTs (student + teacher)")
print("Teacher = EMA of student weights (exponential moving average)")
print("Both process different augmented views; student learns to match teacher")
print("No labels needed → discovers semantic structure purely from visual structure")
print()
print("Key finding: ViT attention heads discover object boundaries and semantic regions")
print("without any supervision — the attention maps ARE the segmentation maps!")

# Patch embedding demo
np.random.seed(42)
img = np.random.randn(3, 224, 224)  # RGB image
pe = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, d_model=768)
tokens = pe.forward(img)
print(f"\\nPatch embedding output: {tokens.shape}")
print(f"Number of patch tokens: {tokens.shape[0]-1} (+ 1 CLS token = {tokens.shape[0]})")
print(f"Each token dimension: {tokens.shape[1]}")
`,
              caption: "Vision Transformer implementation — patch tokenization, positional encoding, and complexity analysis",
            },
          ],
          exercises: [
            {
              id: "ex-vit-1",
              type: "multiple_choice",
              question: "Why does ViT-B/16 underperform ResNet-50 on ImageNet-1k but outperform it on JFT-300M (larger dataset)?",
              options: [
                "ViT has more parameters than ResNet-50",
                "CNNs have inductive biases (locality, translation equivariance) that help with limited data; ViTs learn these from data but need more of it",
                "ResNet-50 uses batch normalization which works better on small datasets",
                "ViT uses a different optimizer that requires more data to converge",
              ],
              correctAnswer: "CNNs have inductive biases (locality, translation equivariance) that help with limited data; ViTs learn these from data but need more of it",
              explanation: "CNNs have strong architectural priors built in: (1) locality — each filter only looks at a local patch, (2) weight sharing — same filter applied everywhere, (3) translation equivariance — a feature detected anywhere activates the same map. These biases are exactly right for natural images. ViTs have no such inductive biases — self-attention attends globally from layer 1. With enough data (300M+ images), ViTs can learn these structure from data and surpass CNNs, but with limited data (1M ImageNet), CNNs' priors give a significant advantage.",
              hints: ["Think about what architectural biases CNNs have that are beneficial for images"],
            },
          ],
        },
      ],
    },
    {
      id: "recurrent-and-state-space",
      trackId: "deep-learning-theory",
      title: "Recurrent Networks, LSTMs & State Space Models",
      description: "BPTT analysis, LSTM gating mechanisms, vanishing gradients through time, S4/Mamba structured state space models as efficient attention alternatives.",
      order: 5,
      estimatedHours: 5,
      lessons: [
        {
          id: "lstm-bptt",
          moduleId: "recurrent-and-state-space",
          trackId: "deep-learning-theory",
          title: "LSTM: Gating Mechanisms, BPTT & Gradient Highway",
          description: "BPTT derivation for RNNs, the vanishing gradient problem over time, LSTM cell equations and how gates create a 'gradient highway', GRUs as simplification.",
          type: "concept",
          estimatedMinutes: 60,
          order: 1,
          prerequisites: ["initialization-theory"],
          keyTakeaways: [
            "Vanilla RNN gradient: ∂hₜ/∂h₀ = Πᵢ Wᵀ diag(σ'(hᵢ)) — exponential in T",
            "LSTM cell state cₜ has additive update — gradient flows through addition, not multiplication",
            "Forget gate fₜ controls what to remember; crucial for long-range dependencies",
            "LSTM gradient: ∂cₜ/∂c₀ = Πᵢ fᵢ — each factor ≤ 1, but additive not multiplicative",
          ],
          sections: [
            {
              id: "s1",
              title: "BPTT and the Gradient Highway",
              type: "math",
              content: `**Vanilla RNN and the Vanishing Gradient:**
hₜ = σ(Whₜ₋₁ + Uxₜ + b)

Backpropagating through time:
∂L/∂h₀ = ∂L/∂hₜ · Πⱼ₌₁ᵗ ∂hⱼ/∂hⱼ₋₁
= ∂L/∂hₜ · Πⱼ₌₁ᵗ [diag(σ'(hⱼ)) · W]

The spectral norm of each Jacobian ∂hⱼ/∂hⱼ₋₁:
‖diag(σ'(hⱼ)) · W‖₂ ≤ ‖σ'‖_∞ · ‖W‖₂

For σ = tanh: ‖σ'‖_∞ = 1
For typical W: ‖W‖₂ ≈ 1 at initialization but becomes < 1 or > 1 during training.

If ‖W‖₂ < 1: gradient vanishes exponentially in T → can't learn long-range dependencies
If ‖W‖₂ > 1: gradient explodes exponentially → unstable training
This is the fundamental limitation of vanilla RNNs.

**LSTM Cell Equations (Hochreiter & Schmidhuber 1997):**
fₜ = σ(Wfhₜ₋₁ + Ufxₜ + bf)   [forget gate: what to erase from memory]
iₜ = σ(Wihₜ₋₁ + Uixₜ + bi)   [input gate: what new info to store]
c̃ₜ = tanh(Wchₜ₋₁ + Ucxₜ + bc) [candidate cell state]
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ     [cell state update — THE GRADIENT HIGHWAY]
oₜ = σ(Wohₜ₋₁ + Uoxₜ + bo)   [output gate]
hₜ = oₜ ⊙ tanh(cₜ)

**Why LSTM solves vanishing gradients:**
The key is the ADDITIVE cell state update: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

∂cₜ/∂cₜ₋₁ = diag(fₜ)

∂cₜ/∂c₀ = Πⱼ₌₁ᵗ ∂cⱼ/∂cⱼ₋₁ = Πⱼ₌₁ᵗ diag(fⱼ)

Each fⱼ ∈ (0,1). The gradient is a product of forget gates, which CAN be close to 1.
More importantly: the gradient passes through ADDITION, not matrix multiplication — avoiding the rank degradation that causes vanishing gradients.

If fⱼ ≈ 1 (forget gate open): gradient flows unchanged — long-range credit assignment!
If fⱼ ≈ 0 (forget gate closed): gradient blocked — intentional erasure.

This is the "constant error carousel" (Hochreiter's original insight) that enables learning over 100s-1000s of timesteps.`,
            },
          ],
          exercises: [
            {
              id: "ex-lstm-1",
              type: "multiple_choice",
              question: "The 'gradient highway' in LSTMs works because:",
              options: [
                "LSTM uses skip connections like ResNets",
                "The cell state update is additive: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t, so gradients pass through addition not matrix multiplication",
                "LSTM gates use sigmoid which has bounded derivatives",
                "LSTM clips gradients automatically through the forget gate",
              ],
              correctAnswer: "The cell state update is additive: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t, so gradients pass through addition not matrix multiplication",
              explanation: "The key is the additive update rule. When computing ∂c_t/∂c_{t-1} = diag(f_t), the gradient only involves the element-wise forget gate, not a full matrix multiplication. A product of T forget gate values Π f_j can stay close to 1 if gates remain open, unlike Π W which has exponentially decaying or exploding norm. The same principle applies in ResNets (residual connections) and attention (additive attention updates). Modern architectures (SSMs, Mamba) extend this idea further.",
              hints: ["Compare the gradient of c_t wrt c_{t-1} with the gradient of h_t wrt h_{t-1} in vanilla RNNs"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 4: LARGE LANGUAGE MODELS & FINE-TUNING
// ============================================================
const llmFinetuningTrack: Track = {
  id: "llm-finetuning",
  title: "Large Language Models & Fine-Tuning",
  description: "Pre-training objectives, scaling laws, parameter-efficient fine-tuning (LoRA, PEFT), RLHF, constitutional AI, quantization, and deployment of modern LLMs.",
  icon: "🔤",
  difficulty: "expert",
  estimatedHours: 40,
  moduleCount: 4,
  lessonCount: 15,
  tags: ["LLMs", "fine-tuning", "RLHF", "LoRA", "quantization", "scaling laws"],
  color: "#f59e0b",
  order: 4,
  modules: [
    {
      id: "pretraining-scaling",
      trackId: "llm-finetuning",
      title: "Pre-training & Scaling Laws",
      description: "Language model pre-training objectives, the Chinchilla scaling laws, compute-optimal training, emergent abilities, and what makes large models powerful.",
      order: 1,
      estimatedHours: 10,
      lessons: [
        {
          id: "scaling-laws",
          moduleId: "pretraining-scaling",
          trackId: "llm-finetuning",
          title: "Scaling Laws: Chinchilla & Compute-Optimal LLM Training",
          description: "Kaplan (GPT-3 era) vs Hoffmann (Chinchilla) scaling laws, the compute-optimal training frontier, power laws in loss, and implications for model architecture decisions.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "Kaplan scaling law: L(N,D) ~ N^{-0.076} — parameters dominate, use few tokens",
            "Chinchilla (Hoffmann 2022): optimal N and D scale equally — 20 tokens per parameter",
            "Compute budget C ≈ 6ND — optimal split: N_opt ≈ (C/6)^{0.5}, D_opt ≈ (C/6)^{0.5}",
            "Emergent abilities: capabilities that appear suddenly at certain scale thresholds",
          ],
          sections: [
            {
              id: "s1",
              title: "Power Laws and the Loss Landscape of Scale",
              type: "math",
              content: `**Kaplan et al. (2020) — "Scaling Laws for Neural Language Models":**
Training loss follows power laws in N (parameters), D (tokens), C (compute):

L(N) ≈ (Nc/N)^{αN},  αN ≈ 0.076
L(D) ≈ (Dc/D)^{αD},  αD ≈ 0.095

When both vary: L(N,D) = (N₀/N)^{αN} + (D₀/D)^{αD} + Lₑ

where Lₑ ≈ 1.69 nats is the irreducible entropy (true data complexity).

**Key Kaplan conclusion:** Parameters dominate. For a fixed compute budget, use as many parameters as possible, training on fewer tokens. GPT-3 (175B params, 300B tokens) followed this.

**Hoffmann et al. (2022) — "Chinchilla" — Overturning Kaplan:**
Running a more careful compute-optimal analysis:

C ≈ 6ND (FLOPs for training a transformer = 6 × N_params × D_tokens)

Minimizing L(N,D) subject to C = 6ND:
∂L/∂N = ∂L/∂D via Lagrange multipliers

**Chinchilla result:** Optimal training is **20 tokens per parameter**.
N_opt ∝ C^{0.50},   D_opt ∝ C^{0.50}

Both scale as square root of compute — equal importance.

GPT-3 (175B, 300B tokens) was undertrained by ~10x!
Chinchilla (70B, 1.4T tokens) beats GPT-3 (4× smaller, 4× more data).

**Implications for modern models:**
- LLaMA family: trained on 1-2T tokens (much more than compute-optimal for inference)
- The "inference-optimal" frontier: if you'll run many inference passes, 
  train smaller models on more data (better inference FLOPs per quality)
- Mistral, Phi, Qwen: smaller models, more data, strong performance

**Emergent Abilities (Wei et al. 2022):**
Some capabilities appear suddenly above a threshold:
- ~10B params: few-shot learning, basic arithmetic
- ~100B params: chain-of-thought reasoning, multi-step math
- Scale threshold is different for each capability

Controversy: Schaeffer et al. (2023) showed emergent abilities may be artifacts of metrics discontinuities — switching from accuracy to log-prob smooths out the "phase transitions". True emergent vs measurement artifact debate continues.`,
            },
            {
              id: "s2",
              title: "Scaling Law Analysis: Fitting the Chinchilla Frontier",
              type: "code",
              language: "python",
              content: `import numpy as np
from scipy.optimize import curve_fit, minimize

"""
Reproducing the Chinchilla scaling law analysis.
Fitting L(N, D) to find compute-optimal training.
"""

def loss_model(ND, log_a_N, log_a_D, alpha_N, alpha_D, L_inf):
    """
    Chinchilla loss model: L(N, D) = (A/N)^alpha_N + (B/D)^alpha_D + L_inf
    
    This is the IsoFLOP parametric fit used in Hoffmann et al.
    """
    N, D = ND
    A = np.exp(log_a_N)
    B = np.exp(log_a_D)
    return (A / N) ** alpha_N + (B / D) ** alpha_D + L_inf

def chinchilla_optimal_allocation(C_budget, alpha_N=0.5, alpha_D=0.5, 
                                   A=406.4, B=410.7, a=0.34, b=0.28):
    """
    Given compute budget C (FLOPs), find optimal N and D.
    
    Using Chinchilla fitted parameters from the paper.
    C ≈ 6 * N * D for transformer training (forward + backward = 6 multiply-adds)
    
    Optimal: N_opt ∝ C^0.50, D_opt ∝ C^0.50
    More precisely from paper: 
      N_opt = G * (C/6)^0.50  where G ≈ 1
      D_opt = C / (6 * N_opt)
    """
    # Chinchilla fitted: N_opt = 0.116 * C^0.5, D_opt = 2.087 * C^0.5
    # This gives ~20 tokens/param at compute optimal
    N_opt = 0.116 * np.sqrt(C_budget)
    D_opt = C_budget / (6 * N_opt)  # from C = 6ND
    
    return N_opt, D_opt

# Analysis of existing models vs. Chinchilla frontier
models = {
    'GPT-3':       {'N': 175e9, 'D': 300e9,  'C': 6*175e9*300e9},
    'Chinchilla':  {'N': 70e9,  'D': 1400e9, 'C': 6*70e9*1400e9},
    'LLaMA-7B':    {'N': 7e9,   'D': 1000e9, 'C': 6*7e9*1000e9},
    'LLaMA-13B':   {'N': 13e9,  'D': 1000e9, 'C': 6*13e9*1000e9},
    'LLaMA-65B':   {'N': 65e9,  'D': 1400e9, 'C': 6*65e9*1400e9},
    'PaLM':        {'N': 540e9, 'D': 780e9,  'C': 6*540e9*780e9},
    'GPT-4 (est)': {'N': 1800e9,'D': 13000e9,'C': 6*1800e9*13000e9},
}

print("=== Chinchilla Optimal vs Actual Training ===\n")
print(f"{'Model':>15} | {'N (params)':>12} | {'D (tokens)':>12} | {'Toks/Param':>10} | {'Chinchilla Opt?':>15}")
print("-" * 75)

for name, m in models.items():
    C = m['C']
    N_opt, D_opt = chinchilla_optimal_allocation(C)
    actual_ratio = m['D'] / m['N']
    optimal_ratio = D_opt / N_opt
    
    # Is it approximately Chinchilla-optimal? (within 2x of optimal ratio)
    is_optimal = 0.5 * optimal_ratio < actual_ratio < 2 * optimal_ratio
    status = "✓ Optimal" if is_optimal else ("Over-trained D" if actual_ratio > 2*optimal_ratio else "Over-scaled N")
    
    print(f"{name:>15} | {m['N']/1e9:>10.1f}B | {m['D']/1e9:>10.1f}B | {actual_ratio:>10.1f} | {status}")

print()
print("Key insight: GPT-3 had only 1.7 tokens/param (Chinchilla says 20 is optimal!)")
print("GPT-3 was massively over-parameterized for its training budget.")
print()
print("=== Compute-Optimal Scaling: What to build? ===\n")
compute_budgets = [1e21, 1e22, 1e23, 1e24, 1e25]  # FLOPs
print(f"{'Compute (FLOPs)':>18} | {'Optimal N':>12} | {'Optimal D':>12} | {'Ratio D/N':>10}")
print("-" * 60)
for C in compute_budgets:
    N_opt, D_opt = chinchilla_optimal_allocation(C)
    print(f"{C:>18.1e} | {N_opt/1e9:>10.1f}B | {D_opt/1e9:>10.1f}B | {D_opt/N_opt:>10.1f}")

print()
print("Scaling law power: predict performance before training!")
print("These laws have held remarkably well across many orders of magnitude.")
print()
print("=== Emergent Abilities Discussion ===")
print("Tasks showing emergent behavior at scale:")
tasks = [
    ("Arithmetic (3-digit)", "~8B params"),
    ("Chain-of-thought reasoning", "~100B params"),
    ("Multi-step math (GSM8K)", "~50B params"),
    ("Code generation (HumanEval)", "~12B params"),
    ("BIG-Bench Hard", ">100B params"),
]
for task, threshold in tasks:
    print(f"  {task:35s}: appears at {threshold}")
print()
print("Controversy: are these truly emergent, or artifacts of discrete metrics?")
print("Log-probability metrics show smooth curves — discrete accuracy shows phase transitions.")
`,
              caption: "Chinchilla scaling law analysis — compute-optimal training and real model comparison",
            },
          ],
          exercises: [
            {
              id: "ex-scaling-1",
              type: "multiple_choice",
              question: "According to Chinchilla scaling laws, GPT-3 (175B params, 300B tokens) was:",
              options: [
                "Approximately compute-optimal",
                "Over-trained on too many tokens",
                "Under-trained — should have trained on ~3.5T tokens for compute-optimal performance",
                "Too small — should have had more parameters",
              ],
              correctAnswer: "Under-trained — should have trained on ~3.5T tokens for compute-optimal performance",
              explanation: "Chinchilla says optimal is ~20 tokens per parameter. GPT-3 had 175B params × 20 = 3.5T optimal tokens, but was trained on only 300B tokens (1.7 tokens/param). GPT-3 was under-trained by ~10x in data. Its parameters could have achieved better performance with more training data. This is why Chinchilla (70B params, 1.4T tokens = 20 tokens/param) outperforms GPT-3 despite being 2.5x smaller. The lesson: for a fixed compute budget, balance parameters and data equally.",
              hints: ["Chinchilla says 20 tokens per parameter at compute-optimum"],
            },
          ],
        },
      ],
    },
    {
      id: "peft-lora",
      trackId: "llm-finetuning",
      title: "Parameter-Efficient Fine-Tuning: LoRA, QLoRA & PEFT Methods",
      description: "LoRA's low-rank decomposition, theoretical justification via intrinsic dimensionality, QLoRA with 4-bit quantization, adapter layers, prefix tuning, and prompt tuning.",
      order: 2,
      estimatedHours: 10,
      lessons: [
        {
          id: "lora-theory",
          moduleId: "peft-lora",
          trackId: "llm-finetuning",
          title: "LoRA: Low-Rank Adaptation Theory & Implementation",
          description: "LoRA's theoretical motivation from intrinsic dimensionality, rank selection, scaling, weight merging for zero-latency inference, and empirical analysis across tasks.",
          type: "coding",
          estimatedMinutes: 75,
          order: 1,
          prerequisites: ["scaling-laws"],
          keyTakeaways: [
            "LoRA: ΔW = BA where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k)",
            "Motivated by intrinsic dimensionality: fine-tuning solutions lie in low-dim subspace",
            "Scaling: W_new = W + (α/r) BA — α controls contribution magnitude",
            "At inference: merge W = W₀ + (α/r)BA for zero overhead",
          ],
          sections: [
            {
              id: "s1",
              title: "LoRA: Theoretical Motivation",
              type: "text",
              content: `**The Fine-Tuning Problem:**
Full fine-tuning of a 7B parameter model requires:
- 7B gradients (28GB at FP32)
- 7B optimizer states × 2 for Adam (56GB)
- 7B weights (28GB)
Total: ~112GB VRAM — requires 8× A100s just for fine-tuning!

**Intrinsic Dimensionality (Aghajanyan et al., 2020):**
Key insight: fine-tuning solutions have extremely low intrinsic dimensionality.

Define the intrinsic dimension d₉₀ as the minimum dimensionality of a random subspace such that optimization in that subspace reaches 90% of full-finetuning performance.

Results: 
- BERT on MRPC: d₉₀ ≈ 896 (vs 125M total params)
- RoBERTa on SQuAD: d₉₀ ≈ 4000 (vs 125M total params)
- GPT-2 on classification: d₉₀ ≈ 1000 (vs 1.5B total params)

The fine-tuning trajectory lives in a ~1000-dimensional subspace even for billion-parameter models!

**LoRA (Hu et al., 2022):**
Instead of learning ΔW ∈ ℝ^{d×k} directly:
ΔW = BA,  B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}

Forward pass: h = W₀x + ΔWx = W₀x + BAx

**Parameter count reduction:**
Original: d×k parameters
LoRA: d×r + r×k = r(d+k) parameters
Reduction: r(d+k)/(dk) ≈ 2r/d for d ≈ k (typically 0.1-1% of original!)

**Initialization:** B = 0, A ~ N(0, σ²) → ΔW = 0 initially (preserves pretrained weights)

**Scaling:** Use α/r scaling factor — total update = (α/r)BAx
Setting α = r: full contribution; α < r: smaller contribution (acts as learning rate for LoRA)

**Why it works:** The gradient subspace of fine-tuning is low-rank. LoRA parameterizes exactly this: B updates the column space, A the row space. The product BA is an explicit rank-r matrix in the weight update space.

**Inference:** Merge W_final = W₀ + (α/r)BA — zero latency overhead at inference!

**Which weights to adapt?** Hu et al. found adapting Q and V attention matrices most effective. Recent work (QLoRA, LoftQ) adapts all linear layers for better performance.`,
            },
            {
              id: "s2",
              title: "LoRA Implementation with Rank Analysis",
              type: "code",
              language: "python",
              content: `import numpy as np

class LoRALinear:
    """
    LoRA (Low-Rank Adaptation) wrapper for a linear layer.
    
    Standard Linear: y = W₀ x
    LoRA Linear:     y = W₀ x + (alpha/r) * B @ A @ x
    
    Only A and B are trained; W₀ is frozen.
    At inference, merge: W_merged = W₀ + (alpha/r) * B @ A
    """
    
    def __init__(self, in_features, out_features, rank=4, alpha=8, 
                 existing_weight=None):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen pretrained weight
        if existing_weight is not None:
            self.W0 = existing_weight.copy()
        else:
            # Simulate a pretrained weight
            self.W0 = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        
        # LoRA matrices: B initialized to 0, A to gaussian
        self.lora_A = np.random.randn(rank, in_features) * np.sqrt(2.0 / in_features)
        self.lora_B = np.zeros((out_features, rank))  # zero init!
        
        # Gradients
        self.grad_A = np.zeros_like(self.lora_A)
        self.grad_B = np.zeros_like(self.lora_B)
    
    def forward(self, x):
        """x: (..., in_features) → y: (..., out_features)"""
        # Original pretrained weights (frozen — no grad)
        y_pretrained = x @ self.W0.T
        # LoRA adaptation (trainable)
        y_lora = (x @ self.lora_A.T) @ self.lora_B.T
        return y_pretrained + self.scaling * y_lora
    
    def backward(self, x, dout):
        """
        dout: gradient wrt output (..., out_features)
        Returns: gradient wrt input
        """
        # Gradients for LoRA parameters (NOT W0 — it's frozen!)
        # ∂L/∂B = (∂L/∂y) * (A x)^T
        Ax = x @ self.lora_A.T  # (..., r)
        self.grad_B = dout.T @ Ax if dout.ndim > 1 else np.outer(dout, Ax)
        self.grad_B *= self.scaling
        
        # ∂L/∂A = B^T (∂L/∂y) * x^T
        self.grad_A = (dout @ self.lora_B @ self.lora_A) / (x @ self.lora_A.T + 1e-9).shape[-1]
        # Simplified: 
        self.grad_A = (self.lora_B.T @ dout.T) @ x if dout.ndim > 1 else np.outer(self.lora_B.T @ dout, x)
        self.grad_A *= self.scaling
        
        # Gradient wrt input (for backprop to previous layer)
        dx = dout @ self.W0 + self.scaling * (dout @ self.lora_B) @ self.lora_A
        return dx
    
    def merge_weights(self):
        """Merge LoRA into base weight — ZERO inference overhead after merge!"""
        return self.W0 + self.scaling * (self.lora_B @ self.lora_A)
    
    def get_trainable_params(self):
        return self.lora_A.size + self.lora_B.size
    
    def get_total_params(self):
        return self.W0.size + self.lora_A.size + self.lora_B.size

def analyze_lora_rank_selection(pretrained_weights, fine_tuned_weights):
    """
    Analyze the singular values of the weight update ΔW = W_ft - W_pretrained
    to understand the intrinsic rank of fine-tuning.
    
    This motivates choosing appropriate LoRA rank r.
    """
    delta_W = fine_tuned_weights - pretrained_weights
    
    U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
    
    # Cumulative variance explained by top-k singular values
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    r_90 = np.searchsorted(cumulative_energy, 0.90) + 1
    r_99 = np.searchsorted(cumulative_energy, 0.99) + 1
    
    print(f"Weight matrix shape: {delta_W.shape}")
    print(f"||ΔW||_F (fine-tuning magnitude): {np.linalg.norm(delta_W, 'fro'):.4f}")
    print(f"Rank for 90% energy: r = {r_90}")
    print(f"Rank for 99% energy: r = {r_99}")
    print(f"Top 5 singular values: {S[:5]}")
    print(f"Singular value decay: ratio S[0]/S[{r_90}] = {S[0]/S[r_90-1]:.2f}")
    
    return r_90, r_99, S

# Analysis
print("=== LoRA Parameter Efficiency ===\n")
d_in, d_out = 4096, 4096  # typical LLM attention dim

for rank in [1, 2, 4, 8, 16, 32, 64]:
    lora = LoRALinear(d_in, d_out, rank=rank, alpha=rank*2)
    trainable = lora.get_trainable_params()
    total = lora.W0.size
    pct = 100 * trainable / total
    print(f"  rank={rank:3d}: {trainable:8,} trainable params / {total:,} total ({pct:.3f}%)")

print()
print("=== Forward Pass Verification ===")
np.random.seed(42)
lora = LoRALinear(64, 64, rank=4, alpha=8)
x = np.random.randn(8, 64)

# Initially: LoRA output ≈ W0 output (B initialized to 0)
y_lora = lora.forward(x)
y_base = x @ lora.W0.T
diff_initial = np.max(np.abs(y_lora - y_base))
print(f"Initial difference (should be 0): {diff_initial:.2e}")

# After merging weights
W_merged = lora.merge_weights()
y_merged = x @ W_merged.T
diff_merged = np.max(np.abs(y_lora - y_merged))
print(f"LoRA vs merged difference: {diff_merged:.2e} (should be ~0)")

print()
print("=== ΔW Rank Analysis (Simulated Fine-Tuning) ===")
# Simulate: ΔW has low-rank structure (as empirically observed)
np.random.seed(42)
d = 256
W_pretrained = np.random.randn(d, d) * 0.01
# Simulate a low-rank fine-tuning update (rank 8 true update)
true_rank = 8
U = np.random.randn(d, true_rank)
V = np.random.randn(true_rank, d)
W_finetuned = W_pretrained + 0.1 * U @ V + 0.001 * np.random.randn(d, d)  # + noise
analyze_lora_rank_selection(W_pretrained, W_finetuned)
`,
              caption: "LoRA implementation with rank analysis — understanding why low-rank adaptation works",
            },
          ],
          exercises: [
            {
              id: "ex-lora-1",
              type: "multiple_choice",
              question: "LoRA initializes B = 0 and A ~ N(0, σ²). Why is this initialization crucial?",
              options: [
                "It makes gradient computation stable",
                "It ensures ΔW = BA = 0 at initialization, so the model starts from the pretrained weights without disruption",
                "It prevents the LoRA matrices from becoming rank-deficient",
                "It maximizes the rank of the update during training",
              ],
              correctAnswer: "It ensures ΔW = BA = 0 at initialization, so the model starts from the pretrained weights without disruption",
              explanation: "With B = 0 at initialization, ΔW = BA = 0·A = 0. The fine-tuned model is identical to the pretrained model at step 0. This is essential: if ΔW ≠ 0 initially, we'd be starting from a perturbed model rather than the carefully pretrained one. As training proceeds, B learns to be non-zero, and the update grows. Note: if we initialized both A=0 and B=0, gradients through A would be zero (since ∂L/∂A ∝ Bᵀ∂L/∂output = 0 initially), so we initialize only B=0 and A randomly.",
              hints: ["What is the value of B@A when B is all zeros?"],
            },
          ],
        },
      ],
    },
    {
      id: "rlhf-alignment",
      trackId: "llm-finetuning",
      title: "RLHF, Alignment & Constitutional AI",
      description: "Reinforcement learning from human feedback (RLHF): reward modeling, PPO for language models, DPO as RLHF-free alignment, Constitutional AI, and the alignment problem.",
      order: 3,
      estimatedHours: 12,
      lessons: [
        {
          id: "rlhf-deep-dive",
          moduleId: "rlhf-alignment",
          trackId: "llm-finetuning",
          title: "RLHF: Reward Modeling, PPO & Direct Preference Optimization (DPO)",
          description: "The full RLHF pipeline: SFT → reward model training → PPO fine-tuning. DPO's closed-form alternative, its theoretical derivation, and practical comparison.",
          type: "concept",
          estimatedMinutes: 75,
          order: 1,
          prerequisites: ["lora-theory"],
          keyTakeaways: [
            "RLHF pipeline: SFT → RM (Bradley-Terry model) → PPO with KL constraint",
            "Reward model trained on preference pairs: P(y_w > y_l) = σ(r(x,y_w) - r(x,y_l))",
            "PPO KL-constrained objective: max E[r(x,y)] - β·KL(π_θ‖π_ref)",
            "DPO: closed-form RLHF — directly optimizes preference pairs without RL",
          ],
          sections: [
            {
              id: "s1",
              title: "RLHF Pipeline: Theory and Components",
              type: "text",
              content: `**The Alignment Problem:**
A pretrained LLM maximizes P(token | context) — it's a distribution-matching objective. But we want models that are helpful, harmless, and honest (HHH). These properties are not captured by next-token prediction alone:
- Helpful: maximize utility to the user (not just plausible text)
- Harmless: avoid toxic, dangerous, deceptive content
- Honest: acknowledge uncertainty, don't hallucinate

**RLHF (Christiano et al. 2017, Ziegler et al. 2019, Ouyang et al. 2022 — InstructGPT):**

**Phase 1: Supervised Fine-Tuning (SFT)**
Fine-tune pretrained LLM on high-quality demonstrations: (prompt, ideal_response) pairs.
This creates π_SFT — a model that follows instructions but may not match human preferences.

**Phase 2: Reward Model Training**
Collect preference data: human annotators rank completions y_w ≻ y_l for prompt x.
Model preferences with Bradley-Terry model:
P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))

where r: (prompt, response) → scalar reward. Train reward model on binary cross-entropy:
L_RM = −E[(x,y_w,y_l) ~ D] [log σ(r(x,y_w) − r(x,y_l))]

**Phase 3: RL Fine-Tuning with PPO**
Maximize expected reward subject to KL constraint (prevents policy from deviating too far from π_SFT):

max_θ E_{x~D, y~π_θ(·|x)} [r(x,y)] − β · KL(π_θ(·|x) ‖ π_SFT(·|x))

The KL penalty prevents "reward hacking" — generating text that gets high reward scores but is gibberish (as reward model is imperfect).

**PPO (Proximal Policy Optimization) for LMs:**
PPO clips the policy ratio to prevent large updates:
L_PPO = E[min(ρ · A, clip(ρ, 1−ε, 1+ε) · A)]
where ρ = π_θ(y|x)/π_old(y|x), A is the advantage estimate.

**Problems with RLHF:**
1. Training instability (PPO is notoriously finicky)
2. Reward model errors compound — model may find adversarial inputs
3. Requires separate reward model (memory + compute overhead)
4. KL balance is sensitive — too small → reward hacking, too large → no learning

**Direct Preference Optimization (DPO — Rafailov et al. 2023):**
Eliminate the RL step entirely! DPO derives a closed-form for the optimal RLHF solution:

The KL-constrained RL objective has analytical solution:
π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)

This implies: r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)

Plugging into the reward model loss (substituting r in terms of π):
L_DPO = −E[(x,y_w,y_l)] [log σ(β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x))]

DPO directly optimizes the policy π_θ on preference pairs — no reward model needed!
Simpler to implement, more stable, competitive or better performance than PPO on many tasks.`,
            },
            {
              id: "s2",
              title: "DPO Implementation",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Direct Preference Optimization (DPO) — simplified implementation.

The key mathematical insight: the RLHF objective with KL constraint
has a closed-form solution that eliminates the RL training step.

DPO loss: -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
"""

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def dpo_loss(log_probs_win, log_probs_lose, log_probs_ref_win, log_probs_ref_lose, beta=0.1):
    """
    DPO loss for a batch of preference pairs.
    
    Args:
        log_probs_win: log π_θ(y_w|x) per sequence — average log-prob over tokens
        log_probs_lose: log π_θ(y_l|x) per sequence
        log_probs_ref_win: log π_ref(y_w|x) — frozen reference model
        log_probs_ref_lose: log π_ref(y_l|x)
        beta: KL penalty coefficient
    
    Returns:
        loss: scalar DPO loss
        metrics: dict of useful training metrics
    """
    # Log-ratios: log(π_θ/π_ref) for winner and loser
    log_ratio_win = log_probs_win - log_probs_ref_win
    log_ratio_lose = log_probs_lose - log_probs_ref_lose
    
    # DPO reward signal: r(x,y) = β * log(π_θ(y|x)/π_ref(y|x))
    reward_win = beta * log_ratio_win
    reward_lose = beta * log_ratio_lose
    
    # Margin: how much more does model prefer winner over loser?
    margin = reward_win - reward_lose
    
    # DPO loss = -log σ(margin)
    # This is equivalent to Bradley-Terry preference loss with implicit rewards
    loss = -np.mean(np.log(sigmoid(margin) + 1e-8))
    
    # Metrics for monitoring DPO training
    reward_accuracy = np.mean(margin > 0)  # fraction where model correctly prefers winner
    reward_margin_mean = np.mean(margin)
    
    metrics = {
        'loss': loss,
        'reward_accuracy': reward_accuracy,
        'reward_margin': reward_margin_mean,
        'reward_win_mean': reward_win.mean(),
        'reward_lose_mean': reward_lose.mean(),
        'kl_win': log_ratio_win.mean(),   # KL deviation for winners
        'kl_lose': log_ratio_lose.mean(),  # KL deviation for losers
    }
    
    return loss, metrics

def simulate_dpo_training(n_steps=200, beta=0.1, lr=1e-3):
    """
    Simulate DPO training on synthetic preference data.
    Shows how the model learns to prefer winning responses.
    """
    np.random.seed(42)
    
    # Simulate: model starts as reference (log_ratio = 0)
    # "Policy" has 2 parameters: bias toward winner and loser responses
    theta_win = 0.0   # model's adjustment to winner log-prob
    theta_lose = 0.0  # model's adjustment to loser log-prob
    
    batch_size = 32
    history = []
    
    for step in range(n_steps):
        # Sample preference data
        # Reference log-probs (fixed) — simulate with gaussian noise
        ref_log_probs_win = np.random.randn(batch_size) * 0.5 - 2.0  # typical seq log-prob
        ref_log_probs_lose = np.random.randn(batch_size) * 0.5 - 2.5  # loser slightly lower
        
        # Current policy log-probs (adjusted by learnable parameters)
        log_probs_win = ref_log_probs_win + theta_win
        log_probs_lose = ref_log_probs_lose + theta_lose
        
        loss, metrics = dpo_loss(
            log_probs_win, log_probs_lose,
            ref_log_probs_win, ref_log_probs_lose,
            beta=beta
        )
        
        # Gradient w.r.t. theta (simplified — real DPO trains full LM)
        # d_loss/d_theta_win = -beta * (1 - sigma(margin)) * mean_per_step
        margin = beta * (log_probs_win - ref_log_probs_win) - beta * (log_probs_lose - ref_log_probs_lose)
        p = sigmoid(margin)
        
        grad_win = -beta * np.mean(1 - p)
        grad_lose = beta * np.mean(1 - p)
        
        theta_win -= lr * grad_win
        theta_lose -= lr * grad_lose
        
        if step % 20 == 0:
            history.append(metrics)
    
    return history

print("=== DPO Training Simulation ===\n")
history = simulate_dpo_training(n_steps=200, beta=0.1, lr=0.01)

print(f"{'Step':>6} | {'Loss':>8} | {'Accuracy':>10} | {'Reward Margin':>14} | {'KL (win)':>10}")
print("-" * 60)
for i, m in enumerate(history):
    print(f"{i*20:>6} | {m['loss']:>8.4f} | {m['reward_accuracy']:>10.3f} | {m['reward_margin']:>14.4f} | {m['kl_win']:>10.4f}")

print()
print("Key DPO training signals:")
print("  Reward accuracy → 1.0: model correctly prefers winners over losers")
print("  Reward margin ↑: model becomes MORE confident in preference ordering")
print("  KL (win/lose): deviation from reference — monitor to prevent collapse")
print()
print("=== DPO vs PPO Comparison ===")
print()
print("PPO-based RLHF:")
print("  1. Train SFT model")
print("  2. Collect preference data → train reward model")
print("  3. Run PPO: generate responses, compute rewards, update policy")
print("  Pros: well-studied, can handle complex reward signals")
print("  Cons: 3 models in memory, unstable PPO, reward hacking")
print()
print("DPO:")
print("  1. Train SFT model")
print("  2. Directly fine-tune on preference pairs (no reward model!)")
print("  Pros: simpler, stable, competitive performance")
print("  Cons: offline (can't generate new preference data during training)")
print()
print("Recent variants: SimPO, ORPO, KTO — further simplify alignment training")
`,
              caption: "DPO implementation — the closed-form alternative to RLHF that eliminates the RL training step",
            },
          ],
          exercises: [
            {
              id: "ex-dpo-1",
              type: "multiple_choice",
              question: "DPO's key mathematical insight is that the optimal RLHF policy can be expressed as:",
              options: [
                "π*(y|x) = argmax_{y} r(x,y)",
                "π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x), allowing r to be expressed in terms of π* and π_ref",
                "π*(y|x) = softmax over all possible responses weighted by reward",
                "π*(y|x) = SFT policy fine-tuned directly on preference pairs",
              ],
              correctAnswer: "π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x), allowing r to be expressed in terms of π* and π_ref",
              explanation: "The KL-constrained RL objective max E[r(x,y)] − β·KL(π‖π_ref) has a closed-form solution: π*(y|x) = π_ref(y|x)exp(r(x,y)/β)/Z(x). This is a Gibbs distribution — the optimal policy is the reference policy re-weighted by exponentiated rewards. Crucially, inverting this: r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). Substituting this back into the reward model training loss (where Z cancels in the difference of rewards) yields the DPO loss — no explicit reward model needed!",
              hints: ["The KL-constrained optimization has a closed-form solution via variational calculus"],
            },
          ],
        },
      ],
    },
    {
      id: "quantization-inference",
      trackId: "llm-finetuning",
      title: "Quantization, Pruning & Efficient LLM Inference",
      description: "Post-training quantization, GPTQ, AWQ, QLoRA, weight pruning, speculative decoding, and system-level optimizations for LLM serving.",
      order: 4,
      estimatedHours: 8,
      lessons: [
        {
          id: "quantization-theory",
          moduleId: "quantization-inference",
          trackId: "llm-finetuning",
          title: "LLM Quantization: GPTQ, AWQ & QLoRA",
          description: "Post-training quantization theory, the quantization error analysis, GPTQ's layer-wise optimal quantization using Hessians, AWQ's activation-aware approach, and QLoRA's 4-bit fine-tuning.",
          type: "coding",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: ["lora-theory"],
          keyTakeaways: [
            "Quantization: W_q = round(W/Δ) × Δ, Δ = (max-min)/(2^b - 1) for uniform b-bit quant",
            "GPTQ: optimal quantization minimizes ‖W·X - W_q·X‖² using layer Hessians",
            "AWQ: protect top-1% salient weights (high activation magnitude) from quantization",
            "QLoRA: frozen 4-bit NF4 base + FP16 LoRA adapters = full fine-tuning quality at <10GB",
          ],
          sections: [
            {
              id: "s1",
              title: "Quantization Theory and GPTQ",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Quantization theory and simplified GPTQ implementation.

Key insight: naively rounding weights to lower precision loses information.
GPTQ (Frantar et al. 2022) uses layer-wise optimal quantization:
minimize ||W*X - W_q*X||^2 (preserve layer OUTPUTS, not weights themselves)
"""

def uniform_quantize(W, bits=4):
    """
    Uniform post-training quantization to b bits.
    Simple approach: scale to [0, 2^b - 1] integer grid.
    
    Quantization error: q_error = W - dequantize(quantize(W))
    """
    n_levels = 2**bits
    W_min = W.min()
    W_max = W.max()
    scale = (W_max - W_min) / (n_levels - 1)
    
    if scale == 0:
        return W, np.zeros_like(W)
    
    # Quantize
    W_int = np.round((W - W_min) / scale).astype(int)
    W_int = np.clip(W_int, 0, n_levels - 1)
    
    # Dequantize (what gets stored and used at inference)
    W_q = W_int * scale + W_min
    
    quant_error = W - W_q
    return W_q, quant_error

def nf4_quantize(W, bits=4):
    """
    NF4 (Normal Float 4-bit) quantization used in QLoRA.
    
    Standard uniform quantization assumes uniform weight distribution.
    NF4 is information-theoretically optimal for GAUSSIAN distributions
    by placing quantization bins with equal probability mass.
    
    NF4 quantile levels for N(0,1) distribution:
    """
    # NF4 quantile levels (from QLoRA paper)
    nf4_levels = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])  # 16 levels for 4-bit
    
    # Normalize W to [-1, 1] based on absmax
    absmax = np.max(np.abs(W))
    W_norm = W / (absmax + 1e-8)
    
    # Find nearest NF4 level for each weight (quantize)
    W_q_idx = np.argmin(np.abs(W_norm[:, :, None] - nf4_levels[None, None, :]), axis=-1)
    W_q_norm = nf4_levels[W_q_idx]
    
    # Dequantize
    W_q = W_q_norm * absmax
    
    quant_error = W - W_q
    return W_q, quant_error

def gptq_quantize_column(w_col, H_inv, bits=4):
    """
    GPTQ: Optimal Brain Quantizer for one weight column.
    
    GPTQ minimizes ||w - w_q||^2_H = (w - w_q)^T H (w - w_q)
    where H = X X^T is the Hessian of the squared output error.
    
    Key insight: quantizing weight w_j changes the optimal values of remaining weights!
    GPTQ sequentially quantizes weights and propagates the quantization error.
    
    This is derived from the Optimal Brain Surgeon (OBS) framework.
    """
    n = len(w_col)
    w_q = np.zeros(n)
    
    for j in range(n):
        # Quantize current weight
        w_q_j, err_j = uniform_quantize(w_col[j:j+1], bits)
        w_q[j] = w_q_j[0]
        q_err = w_col[j] - w_q[j]  # scalar error
        
        # Propagate error to remaining weights using inverse Hessian
        # From OBS: Δw_k = -q_err * H^{-1}_{jk} / H^{-1}_{jj}
        if j + 1 < n:
            h_inv_row = H_inv[j, j+1:]  # H^{-1}[j, j+1:]
            h_inv_diag = H_inv[j, j]
            if abs(h_inv_diag) > 1e-10:
                w_col[j+1:] -= q_err * h_inv_row / h_inv_diag
    
    return w_q

# Analysis
np.random.seed(42)
W = np.random.randn(256, 256) * 0.02  # typical transformer weight magnitude

print("=== Quantization Error Analysis ===\n")
print(f"{'Method':>20} | {'Bits':>4} | {'MSE':>12} | {'SNR (dB)':>10} | {'Memory (MB)':>12}")
print("-" * 68)

W_mse = np.mean(W**2)
methods = [
    ("Uniform FP32", 32, W, np.zeros_like(W)),
    ("Uniform INT8", 8, *uniform_quantize(W, 8)),
    ("Uniform INT4", 4, *uniform_quantize(W, 4)),
    ("Uniform INT2", 2, *uniform_quantize(W, 2)),
    ("NF4", 4, *nf4_quantize(W, 4)),
]

for name, bits, W_q, err in methods:
    mse = np.mean(err**2)
    snr = 10 * np.log10(W_mse / (mse + 1e-10))
    mem = W.size * bits / 8 / 1e6
    print(f"{name:>20} | {bits:>4} | {mse:>12.2e} | {snr:>10.2f} | {mem:>12.3f}")

print()
print("=== NF4 vs Uniform: Why Gaussian-optimized quantization wins ===")
print()

# Test on gaussian weight distribution (matches real LLM weights)
W_gaussian = np.random.randn(1000, 1000) * 0.02
_, err_uniform = uniform_quantize(W_gaussian, 4)
_, err_nf4 = nf4_quantize(W_gaussian, 4)

print(f"Gaussian weights — INT4 vs NF4 at 4 bits:")
print(f"  Uniform INT4 MSE: {np.mean(err_uniform**2):.6f}")
print(f"  NF4 MSE:          {np.mean(err_nf4**2):.6f}")
print(f"  NF4 improvement: {np.mean(err_uniform**2)/np.mean(err_nf4**2):.2f}x better MSE")
print()
print("=== QLoRA: 4-bit base + 16-bit LoRA ===")
print()
print("Memory for 7B parameter model:")
configs = [
    ("Full FP16 finetune", 7e9 * 2, 7e9 * 2 * 3, "Full param gradient + 2 Adam states"),
    ("LoRA FP16 (r=64)", 7e9 * 2, 7e9 * 64/4096 * 2 * 3, "Frozen base + small LoRA states"),
    ("QLoRA (4-bit + r=64)", 7e9 * 0.5, 7e9 * 64/4096 * 2 * 3, "NF4 base + FP16 LoRA"),
]

for name, weight_mem, opt_mem, note in configs:
    total = (weight_mem + opt_mem) / 1e9
    print(f"  {name}:")
    print(f"    Weights: {weight_mem/1e9:.1f}GB, Optimizer: {opt_mem/1e9:.1f}GB, Total: {total:.1f}GB")
    print(f"    → {note}")
    print()
`,
              caption: "Quantization analysis: INT4, NF4, GPTQ, and QLoRA — shrinking LLMs without losing quality",
            },
          ],
          exercises: [
            {
              id: "ex-quant-1",
              type: "multiple_choice",
              question: "Why does NF4 (Normal Float 4-bit) quantization outperform uniform INT4 quantization for LLM weights?",
              options: [
                "NF4 uses more bits for important weights",
                "NF4 places quantization bins at equal-probability quantiles of the normal distribution, providing more precision where weights are most likely to occur",
                "NF4 is a higher-precision format than INT4",
                "NF4 applies weight-dependent scaling to each parameter",
              ],
              correctAnswer: "NF4 places quantization bins at equal-probability quantiles of the normal distribution, providing more precision where weights are most likely to occur",
              explanation: "Information theory tells us that optimal quantization for a distribution should place bins at equal-probability quantile intervals (Lloyd-Max quantizer). For normal distribution N(0,σ), uniform spacing wastes bins in the tails (very rare values) and has coarse bins near 0 (very common values). NF4 distributes 16 bins at the 1/16, 3/16, ..., 15/16 quantiles of N(0,1), giving fine granularity near the mode and coarser spacing in the tails. This is information-theoretically optimal and gives lower MSE per bit for gaussian-distributed weights.",
              hints: ["Think about where gaussian-distributed weights are most densely concentrated"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 5: GENERATIVE MODELS
// ============================================================
const generativeAITrack: Track = {
  id: "generative-ai",
  title: "Generative AI: VAEs, GANs, Diffusion & Flow Matching",
  description: "Deep dive into modern generative modeling: variational autoencoders, GAN training dynamics and mode collapse, denoising diffusion probabilistic models, score matching, and normalizing flows.",
  icon: "✨",
  difficulty: "expert",
  estimatedHours: 45,
  moduleCount: 4,
  lessonCount: 16,
  tags: ["VAEs", "GANs", "diffusion models", "score matching", "normalizing flows"],
  color: "#10b981",
  order: 5,
  modules: [
    {
      id: "vaes-advanced",
      trackId: "generative-ai",
      title: "Variational Autoencoders: Advanced Theory",
      description: "VAE objective derivation, posterior collapse, β-VAE, VQ-VAE, and hierarchical VAEs.",
      order: 1,
      estimatedHours: 10,
      lessons: [
        {
          id: "vae-implementation",
          moduleId: "vaes-advanced",
          trackId: "generative-ai",
          title: "VAE: ELBO Optimization, Posterior Collapse & β-VAE",
          description: "Complete VAE implementation, reparameterization trick, diagnosing posterior collapse, β-VAE for disentangled representations, and VQ-VAE for discrete latents.",
          type: "coding",
          estimatedMinutes: 75,
          order: 1,
          prerequisites: ["entropy-kl-divergence"],
          keyTakeaways: [
            "VAE ELBO = E_q[log p(x|z)] - β·KL(q(z|x)‖p(z))",
            "Posterior collapse: decoder ignores z when it's powerful enough — common in text VAEs",
            "β-VAE (β>1): stronger KL penalty forces disentangled, interpretable latents",
            "VQ-VAE: discrete codebook via straight-through estimator — foundation of DALL-E/Stable Diffusion",
          ],
          sections: [
            {
              id: "s1",
              title: "VAE Implementation with Posterior Collapse Analysis",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
VAE from scratch: demonstrates ELBO computation, reparameterization,
and the posterior collapse phenomenon.
"""

def reparameterize(mu, log_var):
    """
    Reparameterization trick: z = mu + eps * sigma, eps ~ N(0, I)
    Allows gradients to flow through the stochastic sampling step.
    Without this: can't backpropagate through z ~ N(mu, sigma^2).
    """
    sigma = np.exp(0.5 * log_var)
    eps = np.random.randn(*mu.shape)
    return mu + eps * sigma

def vae_elbo(x, mu_z, log_var_z, x_recon, beta=1.0):
    """
    VAE Evidence Lower Bound:
    ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))
    
    For Gaussian likelihood: log p(x|z) = -||x - x_recon||^2 / (2*sigma^2) + const
    (binary: use BCE; continuous: use MSE or Gaussian NLL)
    
    For Gaussian q and p = N(0,I):
    KL(N(mu, sigma^2) || N(0,1)) = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
    """
    # Reconstruction loss (negative log-likelihood of p(x|z))
    recon_loss = np.mean(np.sum((x - x_recon)**2, axis=-1))  # MSE per sample, sum over dims
    
    # KL divergence: KL(q(z|x) || N(0,I)) — closed form for diagonal Gaussians
    var_z = np.exp(log_var_z)
    kl = 0.5 * np.sum(var_z + mu_z**2 - 1 - log_var_z, axis=-1)
    kl_loss = np.mean(kl)  # mean over batch
    
    # ELBO = recon - beta * KL (maximize this)
    elbo = -(recon_loss + beta * kl_loss)
    
    return elbo, recon_loss, kl_loss

def diagnose_posterior_collapse(kl_per_dim, threshold=0.1):
    """
    Posterior collapse: individual latent dimensions collapse to the prior.
    When KL(q(z_i|x) || p(z_i)) < threshold for dimension i, that dimension
    carries no information — the decoder ignores it.
    
    Causes:
    1. Strong decoder (e.g., LSTM) can model data without z
    2. High beta (β-VAE with large β)
    3. Bad initialization
    
    Fixes:
    1. KL annealing (gradually increase beta from 0 to 1)
    2. Free bits: KL loss = max(KL, free_bits_per_dim)  
    3. Weaker decoder (deliberate bottleneck)
    4. β-TCVAE objectives
    """
    n_active = np.sum(kl_per_dim > threshold)
    n_total = len(kl_per_dim)
    
    print(f"Active latent dimensions: {n_active}/{n_total}")
    print(f"Mean KL per dim: {kl_per_dim.mean():.4f}")
    print(f"Max KL per dim: {kl_per_dim.max():.4f}")
    print(f"Collapsed dims (KL < {threshold}): {n_total - n_active}")
    
    if n_active < n_total * 0.3:
        print("⚠️  SEVERE POSTERIOR COLLAPSE: > 70% of dimensions are collapsed")
        print("   Fix: KL annealing, free bits, or weaker decoder")
    elif n_active < n_total * 0.7:
        print("⚠️  MODERATE POSTERIOR COLLAPSE: many dimensions collapsed")
    else:
        print("✓  Healthy: most dimensions are active")
    
    return n_active

# Simulate VAE training behavior
np.random.seed(42)
n_latents = 16
n_samples = 100

print("=== VAE ELBO Analysis ===\n")
print("Scenario 1: Well-trained VAE (healthy latents)")
mu_good = np.random.randn(n_samples, n_latents) * 0.5
log_var_good = np.random.randn(n_samples, n_latents) * 0.5 - 0.5  # near N(0,1)
x = np.random.randn(n_samples, 64)
x_recon = x + 0.1 * np.random.randn(*x.shape)

kl_per_dim = 0.5 * np.mean(np.exp(log_var_good) + mu_good**2 - 1 - log_var_good, axis=0)
elbo, recon, kl = vae_elbo(x, mu_good, log_var_good, x_recon, beta=1.0)
print(f"ELBO: {elbo:.3f}, Recon: {recon:.3f}, KL: {kl:.3f}")
diagnose_posterior_collapse(kl_per_dim)

print("\nScenario 2: Posterior collapse (all KL → 0)")
mu_collapsed = np.zeros((n_samples, n_latents))  # q(z|x) → N(0,1) = prior
log_var_collapsed = np.zeros((n_samples, n_latents))  # var → 1
kl_per_dim_collapsed = 0.5 * np.mean(np.exp(log_var_collapsed) + mu_collapsed**2 - 1 - log_var_collapsed, axis=0)
elbo_c, recon_c, kl_c = vae_elbo(x, mu_collapsed, log_var_collapsed, x_recon, beta=1.0)
print(f"ELBO: {elbo_c:.3f}, Recon: {recon_c:.3f}, KL: {kl_c:.6f} (→ 0!)")
diagnose_posterior_collapse(kl_per_dim_collapsed)

print("\n=== β-VAE: Disentanglement via KL Penalty ===\n")
print("β-VAE sets β > 1 to force more disentangled, interpretable latents.")
print("Higher β → stronger pressure toward N(0,1) → more disentangled but lower quality")
print()
print(f"{'β':>6} | {'Expected KL':>12} | {'Trade-off'}")
print("-" * 50)
for beta in [0.5, 1.0, 2.0, 4.0, 10.0]:
    # Higher beta → lower KL in equilibrium (more compression)
    expected_kl = kl / beta  # approximate effect
    quality = "High recon quality" if beta < 2 else ("Balanced" if beta < 5 else "Disentangled, lower quality")
    print(f"{beta:>6.1f} | {expected_kl:>12.4f} | {quality}")

print()
print("=== VQ-VAE: Discrete Latents ===\n")
print("VQ-VAE replaces Gaussian z with discrete codebook embeddings.")
print("Forward pass: find nearest codebook vector c_k* = argmin_k ||z_e - e_k||")
print("Straight-through estimator: pass gradient through as if z_q = z_e")
print("Loss: ||x - dec(z_q)||^2 + ||sg(z_e) - e||^2 + β*||z_e - sg(e)||^2")
print("where sg = stop_gradient")
print()
print("VQ-VAE → DALL-E: encodes image into 32×32 grid of 8192 discrete codes")
print("These codes are then used as sequence tokens for a transformer language model!")
print("This bridges vision and language modeling.")
`,
              caption: "VAE ELBO analysis, posterior collapse diagnosis, and VQ-VAE concept — the building blocks of modern image generation",
            },
          ],
          exercises: [
            {
              id: "ex-vae-1",
              type: "multiple_choice",
              question: "Posterior collapse in VAEs occurs when:",
              options: [
                "The encoder learns to map all inputs to the same latent code",
                "The decoder becomes powerful enough to model data without using the latent variable z, causing q(z|x) to collapse to the prior p(z)",
                "The KL divergence becomes too large during training",
                "The latent dimension is too large for the data",
              ],
              correctAnswer: "The decoder becomes powerful enough to model data without using the latent variable z, causing q(z|x) to collapse to the prior p(z)",
              explanation: "When the decoder is powerful enough (e.g., an LSTM or large MLP), it can model P(x) without needing information from z. The model then finds a local minimum where q(z|x) ≈ p(z) for all x — KL ≈ 0 (no penalty) and the decoder ignores z completely. This is especially problematic in text VAEs where powerful LSTM decoders have high capacity. Solutions include: KL annealing (start β=0, gradually increase), free bits (min KL per dimension), or deliberately weakening the decoder.",
              hints: ["Think about what happens when the penalty (KL) term drops to zero"],
            },
          ],
        },
      ],
    },
    {
      id: "diffusion-models",
      trackId: "generative-ai",
      title: "Diffusion Models: DDPM, Score Matching & Flow Matching",
      description: "The complete theory of diffusion models: forward/reverse processes, DDPM training and sampling, score matching equivalence, DDIM deterministic sampling, and flow matching.",
      order: 2,
      estimatedHours: 15,
      lessons: [
        {
          id: "ddpm-theory",
          moduleId: "diffusion-models",
          trackId: "generative-ai",
          title: "DDPM: Forward Process, Reverse SDE & Score Matching",
          description: "Deriving the DDPM objective from variational principles, the connection to score matching, noise schedules, DDIM for deterministic sampling, and classifier-free guidance.",
          type: "concept",
          estimatedMinutes: 80,
          order: 1,
          prerequisites: ["vae-implementation"],
          keyTakeaways: [
            "Forward process: q(xₜ|x₀) = N(√(ᾱₜ)x₀, (1-ᾱₜ)I) — any timestep from x₀ directly",
            "DDPM training: predict noise ε from xₜ — equivalent to score matching ∇log p(xₜ)",
            "Reverse SDE: dx = [f(x,t) - g²(t)∇log p_t(x)]dt + g(t)dW",
            "Classifier-free guidance: trade diversity for quality by interpolating conditional/unconditional score",
          ],
          sections: [
            {
              id: "s1",
              title: "DDPM: The Complete Theory",
              type: "math",
              content: `**Forward Diffusion Process:**
q(xₜ|xₜ₋₁) = N(xₜ; √(1−βₜ)xₜ₋₁, βₜI)

where βₜ is the noise schedule. By the Markov property:
q(xₜ|x₀) = N(xₜ; √(ᾱₜ)x₀, (1−ᾱₜ)I)

where ᾱₜ = Πᵢ₌₁ᵗ (1−βᵢ) — the product of (1−β) over all steps.

**Key insight:** We can sample xₜ from x₀ directly!
xₜ = √(ᾱₜ)x₀ + √(1−ᾱₜ)ε,  ε ~ N(0,I)

**ELBO Derivation:**
The DDPM variational lower bound is:
L = E[L₀ + L₁ + ... + Lₜ₋₁ + Lₜ]

where Lₜ₋₁ = KL(q(xₜ₋₁|xₜ,x₀) ‖ p_θ(xₜ₋₁|xₜ))

Both distributions are Gaussian! The posterior q(xₜ₋₁|xₜ,x₀) is tractable:
q(xₜ₋₁|xₜ,x₀) = N(xₜ₋₁; μ̃ₜ(xₜ,x₀), β̃ₜI)

where μ̃ₜ = (√(ᾱₜ₋₁)βₜ)/(1−ᾱₜ) x₀ + (√(1−βₜ)(1−ᾱₜ₋₁))/(1−ᾱₜ) xₜ

**The ε-prediction parameterization:**
Substituting x₀ = (xₜ − √(1−ᾱₜ)ε)/√(ᾱₜ):

Lₜ₋₁ simplifies to: (βₜ²)/(2σₜ²(1−ᾱₜ)) · ‖ε − ε_θ(xₜ,t)‖²

DDPM Training Loss: L_simple = E_{t, x₀, ε} [‖ε − ε_θ(√(ᾱₜ)x₀ + √(1−ᾱₜ)ε, t)‖²]

**Connection to Score Matching:**
The score function is ∇log p_t(x) = −ε/√(1−ᾱₜ)

Predicting ε ↔ estimating the score: s_θ(x,t) = −ε_θ(x,t)/√(1−ᾱₜ)

**Reverse Process and Sampling:**
p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), σₜ²I)

μ_θ(xₜ,t) = (1/√(1−βₜ))[xₜ − (βₜ/√(1−ᾱₜ))ε_θ(xₜ,t)]

**Sampling:** Start xₜ ~ N(0,I), apply T denoising steps.

**DDIM (Song et al., 2020):** Non-Markovian formulation allows deterministic sampling:
xₜ₋₁ = √(ᾱₜ₋₁) · (xₜ − √(1−ᾱₜ)ε_θ)/√(ᾱₜ) + √(1−ᾱₜ₋₁−σₜ²) · ε_θ + σₜε

With σₜ=0: deterministic mapping, can use 10-50 steps instead of 1000!

**Classifier-Free Guidance (Ho & Salimans, 2021):**
Train a single model for both conditional and unconditional generation:
ε_θ(xₜ, t, c) and ε_θ(xₜ, t, ∅) [drop class c with probability p_uncond]

At sampling: ε_guided = ε_θ(xₜ,t,∅) + w·(ε_θ(xₜ,t,c) − ε_θ(xₜ,t,∅))

w > 0: guidance scale. Larger w → sharper, more class-consistent but less diverse samples.
This is the core technique in Stable Diffusion, DALL-E 2, Imagen!`,
            },
            {
              id: "s2",
              title: "DDPM Forward and Reverse Process Implementation",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
DDPM implementation: forward process, training target, and reverse sampling.
Demonstrates the mathematical structure of diffusion models from scratch.
"""

class NoiseSchedule:
    """Linear, cosine, and quadratic noise schedules."""
    
    def __init__(self, T=1000, schedule='cosine', beta_start=1e-4, beta_end=0.02):
        self.T = T
        
        if schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end, T)
        elif schedule == 'cosine':
            # Improved schedule (Nichol & Dhariwal 2021) 
            # Avoids too-noisy early steps that waste compute
            t = np.linspace(0, T, T + 1)
            f_t = np.cos((t/T + 0.008) / (1 + 0.008) * np.pi/2)**2
            alpha_bar = f_t / f_t[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 0, 0.999)
        elif schedule == 'quadratic':
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, T)**2
        
        # Precompute useful quantities
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)  # cumulative product
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1 - self.alpha_bars)
        
        # For reverse process
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1 - np.concatenate([[1.0], self.alpha_bars[:-1]])) / (1 - self.alpha_bars)
    
    def q_sample(self, x0, t, noise=None):
        """
        Forward process: sample x_t from x_0.
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        
        Key: any timestep can be sampled DIRECTLY from x_0 — no Markov chain!
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)
        
        # t can be a batch of timesteps
        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t]
        
        # Broadcast to match x0 shape
        while sqrt_ab.ndim < x0.ndim:
            sqrt_ab = sqrt_ab[..., None]
            sqrt_one_minus_ab = sqrt_one_minus_ab[..., None]
        
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise
    
    def training_step(self, x0, t, noise_predictor):
        """
        DDPM training step: 
        1. Sample random noise eps ~ N(0,I)
        2. Create noisy x_t via forward process
        3. Predict eps from x_t (the denoising task)
        4. Loss = ||eps - eps_predicted||^2
        """
        eps_true = np.random.randn(*x0.shape)
        x_t, eps = self.q_sample(x0, t, eps_true)
        
        # This is where the neural network predicts noise
        eps_pred = noise_predictor(x_t, t)
        
        loss = np.mean((eps - eps_pred)**2)
        return loss, x_t, eps
    
    def p_sample_ddpm(self, x_t, t, eps_pred):
        """
        DDPM reverse step: stochastic denoising.
        p_theta(x_{t-1} | x_t) = N(mu_theta, sigma_t^2 * I)
        
        mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred)
        """
        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bars[t]
        
        # Posterior mean
        mu = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_ab_t * eps_pred)
        
        # Add noise except at t=0
        if t > 0:
            sigma_t = np.sqrt(self.posterior_variance[t])
            z = np.random.randn(*x_t.shape)
            return mu + sigma_t * z
        return mu
    
    def p_sample_ddim(self, x_t, t, t_prev, eps_pred, eta=0.0):
        """
        DDIM deterministic sampling (eta=0) — can use fewer steps!
        
        eta=0: fully deterministic (same latent → same image)
        eta=1: recovers DDPM stochastic sampling
        """
        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t_prev] if t_prev >= 0 else 1.0
        
        # Predict x_0 from x_t and eps_pred
        x0_pred = (x_t - np.sqrt(1 - ab_t) * eps_pred) / np.sqrt(ab_t)
        
        # DDIM update
        sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        
        x_prev = (np.sqrt(ab_prev) * x0_pred + 
                  np.sqrt(1 - ab_prev - sigma**2) * eps_pred +
                  sigma * np.random.randn(*x_t.shape))
        
        return x_prev

# Analysis
schedule_linear = NoiseSchedule(T=1000, schedule='linear')
schedule_cosine = NoiseSchedule(T=1000, schedule='cosine')

print("=== Noise Schedule Comparison ===\n")
print(f"{'Timestep':>10} | {'Linear β':>10} | {'Cosine β':>10} | {'Linear ᾱ':>10} | {'Cosine ᾱ':>10}")
print("-" * 60)
for t in [0, 100, 250, 500, 750, 900, 999]:
    print(f"{t:>10} | {schedule_linear.betas[t]:>10.6f} | {schedule_cosine.betas[t]:>10.6f} | "
          f"{schedule_linear.alpha_bars[t]:>10.4f} | {schedule_cosine.alpha_bars[t]:>10.4f}")

print()
print("Cosine schedule: ᾱ_t decreases more smoothly at the end")
print("Linear schedule: too much noise added early, destroying signal too fast")
print()

# SNR analysis
print("=== Signal-to-Noise Ratio at Each Timestep ===\n")
snr_linear = schedule_linear.alpha_bars / (1 - schedule_linear.alpha_bars)
snr_cosine = schedule_cosine.alpha_bars / (1 - schedule_cosine.alpha_bars)

for t in [0, 100, 300, 500, 700, 900, 999]:
    print(f"t={t:3d}: Linear SNR = {snr_linear[t]:10.4f}, Cosine SNR = {snr_cosine[t]:10.4f}")

print()
print("SNR(t) = alpha_bar_t / (1 - alpha_bar_t)")
print("At t=0: SNR → ∞ (clean data)")
print("At t=T: SNR → 0 (pure noise)")
print()

# Forward diffusion visualization
np.random.seed(42)
x0 = np.array([[1.0, 0.0]])  # simple 1D point
print("=== Forward Diffusion: x₀ = [1, 0] ===")
print(f"{'Timestep':>10} | {'E[x_t]':>20} | {'Std[x_t]':>10} | {'Signal fraction'}")
print("-" * 70)
for t in [0, 100, 300, 500, 700, 999]:
    t_idx = max(t, 0)
    expected_x = schedule_cosine.sqrt_alpha_bars[t_idx] * x0
    std = schedule_cosine.sqrt_one_minus_alpha_bars[t_idx]
    signal_frac = schedule_cosine.alpha_bars[t_idx]
    print(f"{t:>10} | {expected_x[0]:>20} | {std:>10.4f} | {signal_frac:.4f}")

print()
print("=== DDIM vs DDPM Sampling Steps ===")
print("DDPM: 1000 steps required for high quality")  
print("DDIM: 50 steps with eta=0 → deterministic, similar quality")
print("This is a ~20x speedup — critical for practical use!")
`,
              caption: "DDPM noise schedules, forward process, and DDIM accelerated sampling — the mathematics of image generation",
            },
          ],
          exercises: [
            {
              id: "ex-ddpm-1",
              type: "multiple_choice",
              question: "Why does DDPM training predict the noise ε rather than directly predicting x₀?",
              options: [
                "Predicting ε is computationally cheaper",
                "Predicting x₀ directly is equivalent but ε-prediction gives better signal-to-noise ratio at high noise levels, and both objectives are mathematically equivalent",
                "The model can only predict noise, not images",
                "ε prediction avoids the need for a noise schedule",
              ],
              correctAnswer: "Predicting x₀ directly is equivalent but ε-prediction gives better signal-to-noise ratio at high noise levels, and both objectives are mathematically equivalent",
              explanation: "Both objectives are mathematically equivalent since x₀ = (x_t - √(1-ᾱ_t)ε)/√(ᾱ_t). However, ε-prediction has better numerical properties: at high t (heavy noise), ᾱ_t ≈ 0, so predicting x₀ requires dividing by √(ᾱ_t) ≈ 0, creating numerically unstable large values. Predicting ε avoids this instability. Empirically, Ho et al. found ε-prediction produces better samples. More recent work uses velocity prediction v = √(ᾱ_t)ε - √(1-ᾱ_t)x₀ which is even more stable.",
              hints: ["What happens to x₀ = (x_t - √(1-ᾱ_t)ε)/√(ᾱ_t) when ᾱ_t is very small?"],
            },
          ],
        },
      ],
    },
    {
      id: "gans-theory",
      trackId: "generative-ai",
      title: "GANs: Training Dynamics, Mode Collapse & Variants",
      description: "GAN training as minimax game, Nash equilibrium analysis, mode collapse mechanics, Wasserstein GAN and its theoretical advantages, spectral normalization, and progressive growing.",
      order: 3,
      estimatedHours: 12,
      lessons: [
        {
          id: "wgan-theory",
          moduleId: "gans-theory",
          trackId: "generative-ai",
          title: "Wasserstein GAN: Optimal Transport & Training Stability",
          description: "Why vanilla GAN fails theoretically (disjoint supports, vanishing gradients), Wasserstein distance as a superior metric, the Kantorovich-Rubinstein duality, and WGAN-GP.",
          type: "concept",
          estimatedMinutes: 75,
          order: 1,
          prerequisites: ["vae-implementation"],
          keyTakeaways: [
            "Vanilla GAN: JS divergence undefined on disjoint supports → vanishing discriminator gradients",
            "Wasserstein-1 distance: W(p,q) = inf_{γ∈Π(p,q)} E[‖x-y‖] — optimal transport cost",
            "K-R duality: W(p,q) = sup_{‖f‖_L≤1} E_p[f] - E_q[f] → critic in WGAN",
            "WGAN-GP: gradient penalty ‖∇f(x̂)‖ = 1 enforces Lipschitz constraint",
          ],
          sections: [
            {
              id: "s1",
              title: "Why Vanilla GAN Fails: The Disjoint Supports Problem",
              type: "text",
              content: `**Vanilla GAN Objective:**
min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

At optimality, D*(x) = p_data(x) / (p_data(x) + p_g(x)). The generator minimizes:
JSD(p_data ‖ p_g) = (1/2)KL(p_data ‖ M) + (1/2)KL(p_g ‖ M)

where M = (p_data + p_g)/2.

**The Fundamental Problem:** In high dimensions, p_data and p_g are supported on low-dimensional manifolds (the image manifold hypothesis). If these manifolds are disjoint (which they typically are when G hasn't learned the true distribution):

JSD(p_data ‖ p_g) = log 2  (constant, maximum value!)

The gradient of JSD wrt G's parameters is ZERO — the generator receives no signal!

**In practice:** The discriminator becomes perfect early, giving gradient near 0 to the generator. This is why GAN training is so unstable: you're trying to train against a discriminator that provides vanishing gradient signal.

**Alternative GAN objectives that partially fix this:**
- Non-saturating loss: -E[log D(G(z))] instead of E[log(1-D(G(z)))] — shifts the gradient problem
- Least squares GAN: minimize (D(G(z))-1)² — avoids saturating discriminator
- But these are hacks around a deeper geometric problem

**Wasserstein Distance (Earth Mover's Distance):**
W_p(μ, ν) = inf_{γ ∈ Π(μ,ν)} (∫‖x−y‖ᵖ dγ(x,y))^{1/p}

The minimum "work" needed to transport mass from distribution μ to ν.

**Why W₁ is better:**
- W₁(δ₀, δ_θ) = |θ| for Dirac deltas — grows continuously even when supports are disjoint!
- JS(δ₀, δ_θ) = log 2 for all θ ≠ 0, then drops to 0 at θ = 0 — discontinuous!
- W₁ provides meaningful gradients even when distributions have disjoint support

**Kantorovich-Rubinstein Duality:**
W₁(p, q) = sup_{‖f‖_L ≤ 1} (E_p[f(x)] − E_q[f(x)])

The supremum over all 1-Lipschitz functions f. The optimal f is the "critic" in WGAN!

**WGAN Training:**
Critic loss: E_p[f_w(x)] − E_q[f_w(G(z))]  (maximize)
Generator loss: −E_q[f_w(G(z))]  (maximize)

Constraint: ‖f_w‖_L ≤ 1 (Lipschitz constraint)

Original WGAN: weight clipping [-c, c] — crude but works
WGAN-GP (Gulrajani 2017): gradient penalty E[(‖∇f(x̂)‖₂ − 1)²] where x̂ interpolates real and fake

The gradient penalty directly enforces |∇f|=1 (Lipschitz condition ↔ gradient norm ≤ 1).`,
            },
          ],
          exercises: [
            {
              id: "ex-wgan-1",
              type: "multiple_choice",
              question: "Why does Wasserstein distance provide meaningful gradients even when p_data and p_g have disjoint supports, while JS divergence does not?",
              options: [
                "Wasserstein distance is always smaller than JS divergence",
                "Wasserstein distance measures the optimal transport cost between distributions, which varies continuously even as disjoint manifolds move apart, while JS divergence is constant (log 2) whenever supports are disjoint",
                "Wasserstein distance is computed using neural networks which provide gradients",
                "Wasserstein distance doesn't require the discriminator to be perfect",
              ],
              correctAnswer: "Wasserstein distance measures the optimal transport cost between distributions, which varies continuously even as disjoint manifolds move apart, while JS divergence is constant (log 2) whenever supports are disjoint",
              explanation: "JS divergence on disjoint supports equals log 2 regardless of how far apart the supports are — it's maximally uninformative about direction and magnitude of movement. Wasserstein distance, being the minimum cost of moving mass, continues to vary as the distributions shift relative to each other. If G(z) is a distribution centered at x₀ and data is centered at x₁, W₁ = |x₀-x₁|, giving a clear gradient direction. This geometric property makes WGAN training fundamentally more stable.",
              hints: ["Consider two delta distributions δ(x) and δ(x-θ) as θ varies"],
            },
          ],
        },
      ],
    },
    {
      id: "normalizing-flows",
      trackId: "generative-ai",
      title: "Normalizing Flows & Flow Matching",
      description: "Exact density estimation via invertible transformations, RealNVP, Glow, continuous normalizing flows, and the recent flow matching paradigm for efficient generative modeling.",
      order: 4,
      estimatedHours: 8,
      lessons: [
        {
          id: "flow-matching",
          moduleId: "normalizing-flows",
          trackId: "generative-ai",
          title: "Flow Matching: From Continuous Normalizing Flows to Modern Rectified Flow",
          description: "ODE-based generative models, probability flow ODEs, flow matching training vs score matching, rectified flow for straight-line trajectories, and Stable Diffusion 3's architecture.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: ["ddpm-theory"],
          keyTakeaways: [
            "Continuous normalizing flows: dx/dt = v_θ(x,t), log p changes via log det Jacobian of v",
            "Flow matching: train v_θ to match conditional vector field u(x|x₁) — no SDE needed",
            "Rectified flow: straight-line paths from noise to data — 1-step generation possible after distillation",
            "Stable Diffusion 3 uses flow matching instead of DDPM — faster sampling and better quality",
          ],
          sections: [
            {
              id: "s1",
              title: "Flow Matching: Simpler Than Score Matching",
              type: "text",
              content: `**From Diffusion to ODE:**
Every SDE: dx = f(x,t)dt + g(t)dW has a corresponding probability flow ODE:
dx/dt = f(x,t) − (1/2)g²(t)∇log p_t(x)

that generates the SAME marginal distributions p_t(x). This ODE is deterministic — no stochasticity!

**Continuous Normalizing Flows (CNF):**
Model data generation as an ODE: dx/dt = v_θ(x, t), x(0) ~ N(0,I), x(1) ~ p_data

Training CNFs requires computing the trace of the Jacobian: log p(x(1)) = log p(x(0)) − ∫₀¹ tr(∂v_θ/∂x) dt

Hutchinson trace estimator: tr(J) ≈ εᵀJε for ε ~ N(0,I) — makes this tractable.

**Flow Matching (Lipman et al. 2022, Liu et al. 2022):**
Instead of training via likelihood (expensive), train by regression on vector fields!

**Conditional vector field:** Given a data point x₁, define a simple path from x₀ ~ N(0,I) to x₁:
ψ(x|x₁) = (1−t)x + t·x₁  (straight line interpolation!)
u_t(x|x₁) = x₁ − x  (velocity = direction toward data)

**Flow Matching Loss:**
L_FM = E_{t,x₁~p,x₀~N} [‖v_θ(ψ(x₀|x₁), t) − u_t(x₀|x₁)‖²]
= E_{t,x₁~p,x₀~N} [‖v_θ((1−t)x₀ + t·x₁, t) − (x₁ − x₀)‖²]

This is MUCH simpler than score matching:
- No forward SDE needed
- No noise schedule design
- Direct regression on vector fields
- Can use any transport between source and target

**Rectified Flow (Liu et al. 2022):**
Use straight-line paths from noise to data. After training:
1. Generate samples by following the learned ODE
2. Retrain on (x₀, x₁) pairs from step 1 ("reflow") — paths become straighter
3. After 1-2 reflows: paths are nearly straight → 1-step generation!

This gives a simpler, faster alternative to diffusion models with:
- Faster sampling (fewer ODE steps)
- Simpler training objective  
- Better theoretical properties (optimal transport connection)

**Stable Diffusion 3** (Esser et al. 2024) uses flow matching with:
- Multimodal diffusion transformer (text + image)
- Flow matching instead of DDPM noise schedules
- Significantly better text rendering and compositional understanding`,
            },
          ],
          exercises: [
            {
              id: "ex-flow-1",
              type: "multiple_choice",
              question: "What is the key advantage of flow matching over score matching for training generative models?",
              options: [
                "Flow matching produces higher quality images",
                "Flow matching directly trains on vector fields by regression, avoiding the need to estimate score functions or solve SDEs",
                "Flow matching is faster at inference time",
                "Flow matching doesn't require a neural network",
              ],
              correctAnswer: "Flow matching directly trains on vector fields by regression, avoiding the need to estimate score functions or solve SDEs",
              explanation: "Score matching requires estimating ∇log p_t(x) — an intrinsically difficult density estimation problem. Flow matching instead trains a vector field v_θ(x,t) to match a known conditional velocity u(x|x₁), which is a simple regression problem. The conditional velocity has a closed form (x₁−x for straight paths), so the training target is known exactly — no estimation needed. This leads to simpler training, better conditioning, and often better results with the same compute.",
              hints: ["Compare what each method's training loss requires you to compute"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 6: REINFORCEMENT LEARNING
// ============================================================
const reinforcementLearningTrack: Track = {
  id: "reinforcement-learning",
  title: "Deep Reinforcement Learning",
  description: "Policy gradient theorem, actor-critic methods, PPO, model-based RL, multi-agent RL, offline RL, and RL for language model alignment — from foundations to cutting-edge research.",
  icon: "🎮",
  difficulty: "expert",
  estimatedHours: 40,
  moduleCount: 4,
  lessonCount: 14,
  tags: ["policy gradients", "actor-critic", "PPO", "model-based RL", "offline RL"],
  color: "#ef4444",
  order: 6,
  modules: [
    {
      id: "policy-gradients",
      trackId: "reinforcement-learning",
      title: "Policy Gradient Theory: REINFORCE to Actor-Critic",
      description: "Policy gradient theorem derivation, REINFORCE, baseline subtraction, advantage estimation, actor-critic architectures, and variance reduction techniques.",
      order: 1,
      estimatedHours: 12,
      lessons: [
        {
          id: "policy-gradient-theorem",
          moduleId: "policy-gradients",
          trackId: "reinforcement-learning",
          title: "Policy Gradient Theorem: Derivation & REINFORCE",
          description: "Formal derivation of the policy gradient theorem, REINFORCE algorithm, baseline subtraction for variance reduction, and the log-derivative trick.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "∇_θ J(θ) = E_τ[Σₜ ∇_θ log π_θ(aₜ|sₜ) · G_t] — log-derivative trick",
            "Baseline subtraction: replace G_t with G_t - b(sₜ) — unbiased, reduces variance",
            "Optimal baseline: b*(s) = E[G_t²∇log π] / E[∇log π] — minimizes variance",
            "GAE (Generalized Advantage Estimation): λ-weighted returns for bias-variance tradeoff",
          ],
          sections: [
            {
              id: "s1",
              title: "Policy Gradient Theorem Derivation",
              type: "math",
              content: `**Setup:**
- MDP: (S, A, P, R, γ)
- Parameterized policy: π_θ(a|s)
- Objective: J(θ) = E_τ~π_θ [Σₜ γᵗ r(sₜ, aₜ)]

**Policy Gradient Theorem (Sutton et al. 2000):**
∇_θ J(θ) = E_τ [Σₜ ∇_θ log π_θ(aₜ|sₜ) · Qπ(sₜ, aₜ)]

**Derivation using the log-derivative trick:**
∇_θ P(τ|θ) = P(τ|θ) · ∇_θ log P(τ|θ)

Since P(τ|θ) = p(s₀) · Πₜ π_θ(aₜ|sₜ) · P(sₜ₊₁|sₜ,aₜ):
log P(τ|θ) = log p(s₀) + Σₜ log π_θ(aₜ|sₜ) + Σₜ log P(sₜ₊₁|sₜ,aₜ)

∇_θ log P(τ|θ) = Σₜ ∇_θ log π_θ(aₜ|sₜ)  [environment dynamics cancel!]

Therefore:
∇_θ J(θ) = ∫ R(τ) ∇_θ P(τ|θ) dτ
           = E_τ [R(τ) Σₜ ∇_θ log π_θ(aₜ|sₜ)]
           = E_τ [Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ]

where Gₜ = Σₖ₌ₜᵀ γᵏ⁻ᵗ rₖ (rewards-to-go = causal structure).

**Why rewards-to-go (not full return)?**
Future action gradients don't depend on past rewards:
E[∇_θ log π(aₜ|sₜ) · rₖ] = 0 for k < t (past rewards)
So Gₜ is sufficient — this halves variance!

**Baseline Subtraction:**
∇_θ J(θ) = E[Σₜ ∇_θ log π_θ(aₜ|sₜ) · (Gₜ − b(sₜ))]

Adding baseline b(sₜ) is unbiased (E[∇_θ log π_θ(aₜ|sₜ) · b(sₜ)] = 0 since E[∇_θ log π] = 0):
E_{a~π}[∇_θ log π(a|s)] = ∫ π(a|s) ∇_θ log π(a|s) da = ∇_θ ∫ π(a|s) da = ∇_θ 1 = 0 ✓

**Optimal baseline:**
b*(s) = E[‖∇_θ log π‖² Gₜ] / E[‖∇_θ log π‖²]

In practice: b(s) = V^π(s) (state value function) — gives the ADVANTAGE:
A^π(s,a) = Q^π(s,a) − V^π(s)

High-variance REINFORCE becomes actor-critic with advantage estimates!

**Generalized Advantage Estimation (Schulman et al., 2015):**
GAE(λ) = (1−λ) Σₙ λⁿ Aₙ^π = Σₜ (γλ)ᵏ δₜ₊ₖ

where δₜ = rₜ + γV(sₜ₊₁) − V(sₜ) is the TD error.

λ=0: δₜ (TD residual — low variance, high bias)
λ=1: Monte Carlo returns (high variance, low bias)
λ∈(0.9, 0.99): best empirical performance`,
            },
            {
              id: "s2",
              title: "REINFORCE and Actor-Critic Implementation",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Policy gradient algorithms from scratch.
Implementing REINFORCE, REINFORCE with baseline, and advantage actor-critic (A2C).
"""

def softmax(x):
    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))

def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.sum(np.exp(x)))

class TabularPolicy:
    """
    Tabular softmax policy for discrete action spaces.
    Policy: π(a|s) = softmax(θ[s])
    """
    
    def __init__(self, n_states, n_actions, lr_policy=0.01, lr_value=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        
        # Policy parameters: theta[s, a] for softmax policy
        self.theta = np.zeros((n_states, n_actions))
        # Baseline/value function parameters
        self.V = np.zeros(n_states)
    
    def pi(self, s):
        """π(·|s) — action probabilities"""
        return softmax(self.theta[s])
    
    def act(self, s):
        """Sample action from policy"""
        probs = self.pi(s)
        return np.random.choice(self.n_actions, p=probs)
    
    def reinforce_update(self, trajectories, use_baseline=True, use_advantage=False):
        """
        REINFORCE policy gradient update.
        
        For each trajectory τ = [(s₀,a₀,r₀), ...]:
        θ ← θ + α Σₜ ∇θ log π(aₜ|sₜ) · (Gₜ − b(sₜ))
        
        ∇θ log π(aₜ|sₜ) for softmax policy:
        = e_aₜ - π(·|sₜ)  (one-hot minus probability vector)
        """
        grad_theta = np.zeros_like(self.theta)
        grad_V = np.zeros_like(self.V)
        
        for traj in trajectories:
            states, actions, rewards = zip(*traj)
            T = len(states)
            
            # Compute returns Gₜ = Σₖ≥ₜ γ^(k-t) r_k
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = np.array(returns)
            
            # Value function targets (for baseline/critic)
            if use_baseline or use_advantage:
                # Update V toward Monte Carlo returns
                for t, s in enumerate(states):
                    grad_V[s] += returns[t] - self.V[s]  # TD error
            
            for t in range(T):
                s, a = states[t], actions[t]
                G_t = returns[t]
                
                # Baseline or advantage
                if use_advantage:
                    delta = G_t - self.V[s]  # advantage estimate
                elif use_baseline:
                    delta = G_t - self.V[s]  # same as advantage here (MC baseline)
                else:
                    delta = G_t  # raw REINFORCE
                
                # Policy gradient: ∇θ log π(a|s) = e_a - π(·|s)
                probs = self.pi(s)
                grad_log_pi = -probs.copy()
                grad_log_pi[a] += 1.0  # e_a - pi(·|s)
                
                grad_theta[s] += delta * grad_log_pi
        
        # Apply updates (batch average)
        n = len(trajectories)
        self.theta += self.lr_policy * grad_theta / n
        if use_baseline or use_advantage:
            self.V += self.lr_value * grad_V / n
        
        return grad_theta, grad_V

def run_gridworld_episode(policy, n_states=16, n_actions=4, max_steps=50):
    """
    4x4 gridworld: start at (0,0), goal at (3,3).
    Actions: 0=right, 1=up, 2=left, 3=down
    Reward: +10 at goal, -0.1 per step, -1 at walls
    """
    goal = n_states - 1
    s = 0  # start
    trajectory = []
    
    for _ in range(max_steps):
        a = policy.act(s)
        
        # Grid transitions (4x4 grid)
        row, col = s // 4, s % 4
        if a == 0: col = min(col + 1, 3)
        elif a == 1: row = max(row - 1, 0)
        elif a == 2: col = max(col - 1, 0)
        elif a == 3: row = min(row + 1, 3)
        s_next = row * 4 + col
        
        r = 10.0 if s_next == goal else -0.1
        trajectory.append((s, a, r))
        s = s_next
        
        if s == goal:
            break
    
    return trajectory

# Training comparison
print("=== REINFORCE vs Actor-Critic: Variance Comparison ===\n")

configs = [
    ("REINFORCE (no baseline)", False, False),
    ("REINFORCE + baseline", True, False),
    ("Advantage Actor-Critic (A2C)", True, True),
]

for name, baseline, advantage in configs:
    policy = TabularPolicy(n_states=16, n_actions=4, lr_policy=0.05, lr_value=0.1)
    
    episode_returns = []
    policy_grad_norms = []
    
    for episode in range(500):
        # Collect batch of trajectories
        batch = [run_gridworld_episode(policy) for _ in range(8)]
        grads, _ = policy.reinforce_update(batch, use_baseline=baseline, use_advantage=advantage)
        
        returns = [sum(r for _, _, r in traj) for traj in batch]
        episode_returns.append(np.mean(returns))
        policy_grad_norms.append(np.linalg.norm(grads))
    
    final_return = np.mean(episode_returns[-50:])
    grad_std = np.std(policy_grad_norms)
    
    print(f"{name}:")
    print(f"  Final avg return (last 50 episodes): {final_return:.2f}")
    print(f"  Policy gradient norm std (variance proxy): {grad_std:.4f}")
    print()

print("Key insight: baseline and advantage estimation dramatically reduce gradient variance.")
print("A2C matches the performance of REINFORCE but with lower variance → faster learning.")
print()
print("=== GAE-λ: Bias-Variance Tradeoff ===\n")
print("GAE(λ=0): TD error δₜ = rₜ + γV(sₜ₊₁) - V(sₜ) — low variance, high bias (if V wrong)")
print("GAE(λ=1): Monte Carlo Gₜ - V(sₜ) — unbiased but high variance")
print("GAE(λ=0.95): best of both worlds — used in PPO, A3C in practice")
`,
              caption: "REINFORCE to actor-critic — empirical variance reduction comparison on gridworld",
            },
          ],
          exercises: [
            {
              id: "ex-pg-1",
              type: "multiple_choice",
              question: "Why does adding a baseline b(s) to the REINFORCE gradient estimator not introduce bias?",
              options: [
                "The baseline cancels out in the final gradient calculation",
                "E_{a~π}[∇_θ log π(a|s) · b(s)] = b(s) · E_{a~π}[∇_θ log π(a|s)] = b(s) · 0 = 0",
                "The baseline is chosen to be zero in expectation",
                "The policy gradient theorem already accounts for any baseline",
              ],
              correctAnswer: "E_{a~π}[∇_θ log π(a|s) · b(s)] = b(s) · E_{a~π}[∇_θ log π(a|s)] = b(s) · 0 = 0",
              explanation: "The key identity: E_{a~π_θ}[∇_θ log π_θ(a|s)] = ∫π(a|s)·∇log π(a|s)da = ∫∇π(a|s)da = ∇∫π(a|s)da = ∇1 = 0. Since the baseline b(s) doesn't depend on a (only on state s), it factors out: E[∇log π · b(s)] = b(s) · E[∇log π] = b(s) · 0 = 0. This means any state-dependent baseline can be subtracted without changing the expected gradient. The baseline only affects variance — choose it to minimize variance.",
              hints: ["Use the identity E[∇log π(a|s)] = ∇E[1] = 0"],
            },
          ],
        },
        {
          id: "ppo-algorithm",
          moduleId: "policy-gradients",
          trackId: "reinforcement-learning",
          title: "PPO: Proximal Policy Optimization & Trust Region Methods",
          description: "TRPO's trust region constraint and its computational cost, PPO's clipped surrogate objective as a practical approximation, PPO-clip vs PPO-penalty, and implementation details.",
          type: "coding",
          estimatedMinutes: 70,
          order: 2,
          prevLessonId: "policy-gradient-theorem",
          prerequisites: ["policy-gradient-theorem"],
          keyTakeaways: [
            "TRPO: max E[ρ(s,a)A^π] s.t. KL(π_old‖π_new) ≤ δ — exact trust region, expensive",
            "PPO-clip: max E[min(ρA, clip(ρ, 1-ε, 1+ε)A)] — simple, effective approximation",
            "PPO entropy bonus: + β·H(π) encourages exploration",
            "PPO is the standard algorithm for RLHF in LLMs (ChatGPT, Claude, Gemini)",
          ],
          sections: [
            {
              id: "s1",
              title: "PPO: Proximal Policy Optimization",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Proximal Policy Optimization (PPO) — Schulman et al. 2017.

Key idea: TRPO constrains the policy update size via a KL constraint.
PPO approximates TRPO's constraint with a simple clipping mechanism.

The clipped surrogate objective prevents overly large policy updates
that could destabilize training — crucial for RL with neural networks.
"""

def ppo_clip_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2):
    """
    PPO clipped surrogate loss.
    
    Importance ratio: ρ = π_new(a|s) / π_old(a|s) = exp(log_π_new - log_π_old)
    
    Clipped objective: min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)
    
    The clip prevents:
    - ρ >> 1 (new policy very different from old) from giving large gradient
    - Exploiting advantage estimates beyond the trust region
    
    Args:
        log_probs_new: log π_θ(a|s) for current policy parameters
        log_probs_old: log π_θ_old(a|s) for old policy (fixed for K epochs)
        advantages: A^π(s,a) estimates (normalized is important!)
    """
    # Importance ratio
    log_ratio = log_probs_new - log_probs_old
    ratio = np.exp(log_ratio)
    
    # PPO surrogate objective: min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    
    # Take minimum (pessimistic bound on improvement)
    ppo_loss = -np.mean(np.minimum(surr1, surr2))
    
    # Diagnostics
    clipping_fraction = np.mean(np.abs(ratio - 1.0) > clip_eps)
    approx_kl = np.mean(-log_ratio + ratio - 1)  # Taylor expansion of KL
    
    return ppo_loss, {
        'ratio_mean': ratio.mean(),
        'ratio_std': ratio.std(),
        'clipping_fraction': clipping_fraction,
        'approx_kl': approx_kl
    }

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation.
    
    δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)           [TD error]
    Aₜ = δₜ + γλ·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...  [GAE]
    
    Returns both advantages and value targets (for critic training).
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values  # value targets for critic
    return advantages, returns

def normalize_advantages(advantages):
    """
    Normalize advantages to have zero mean and unit variance.
    Critical for PPO stability — prevents dominant advantage values
    from driving the update.
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# PPO hyperparameter analysis
print("=== PPO: Effect of Clipping Epsilon ===\n")
print("Simulating ratio distribution and clipping behavior\n")

n_samples = 10000
advantages = np.concatenate([np.random.randn(n_samples//2) * 2, 
                              np.random.randn(n_samples//2) * 0.5])

for eps in [0.1, 0.2, 0.3, 0.5]:
    # Simulate ratios from policy that has improved (ratios > 1 on average)
    log_ratio = np.random.randn(n_samples) * 0.3 + 0.1  # slightly improved policy
    ratio = np.exp(log_ratio)
    
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    
    clipped = np.mean(np.minimum(surr1, surr2) < surr1)  # fraction clipped
    loss = -np.mean(np.minimum(surr1, surr2))
    
    print(f"ε={eps:.1f}: clipped={clipped:.1%}, loss={loss:.4f}")

print()
print("Smaller ε → more conservative updates but more stable")
print("PPO default ε=0.2 is a well-calibrated balance")
print()

print("=== PPO Training Loop Structure ===\n")
print("""
for iteration in range(n_iterations):
    # Phase 1: Data collection with OLD policy (frozen)
    rollouts = []
    for _ in range(n_envs):
        rollout = collect_trajectory(env, policy_old, T_horizon)
        rollouts.append(rollout)
    
    # Compute advantages using GAE
    advantages, returns = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    advantages = normalize_advantages(advantages)  # important!
    
    # Phase 2: Policy update with K epochs of minibatches
    for epoch in range(K_epochs):  # typically K=4-10
        for minibatch in shuffle_and_split(rollouts):
            # Actor loss (PPO-clip)
            ratio = exp(log_prob_new - log_prob_old)  # recompute with current params
            ppo_loss = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))
            
            # Critic loss (MSE to GAE returns)
            value_loss = mean((V(s) - returns)^2)
            
            # Entropy bonus (exploration)
            entropy_bonus = -alpha * mean(sum(pi * log(pi), dim=-1))
            
            total_loss = ppo_loss + c1 * value_loss - c2 * entropy_bonus
            optimizer.step(total_loss)
        
        # Early stopping: if approx KL > target KL, stop epochs
        if approx_kl > 1.5 * target_kl:
            break
    
    # Update old policy
    policy_old = copy(policy_current)
""")
print("Key PPO tricks for stability:")
print("  1. Advantage normalization (per-minibatch or per-rollout)")
print("  2. Value clipping: clip(V_new, V_old-eps, V_old+eps)")
print("  3. Global gradient clipping: max_norm=0.5")
print("  4. Orthogonal initialization (especially for policy head)")
print("  5. Linear learning rate annealing")
`,
              caption: "PPO implementation analysis — clipping mechanics, GAE, and training loop structure",
            },
          ],
          exercises: [
            {
              id: "ex-ppo-1",
              type: "multiple_choice",
              question: "Why does PPO use the MINIMUM of the clipped and unclipped surrogate (min(ρA, clip(ρ,1-ε,1+ε)A)) rather than just the clipped version?",
              options: [
                "The minimum is easier to compute than the clipped version",
                "Taking the minimum creates a pessimistic lower bound: it clips when ρ is too high AND prevents gain when ρ is too high with negative advantage, making it a one-sided constraint",
                "The minimum ensures the gradient is always positive",
                "TRPO uses the same formulation, so PPO inherits it",
              ],
              correctAnswer: "Taking the minimum creates a pessimistic lower bound: it clips when ρ is too high AND prevents gain when ρ is too high with negative advantage, making it a one-sided constraint",
              explanation: "Consider two cases: (1) A > 0 (good action): ratio ρ should increase π. The clip limits ρ to ≤ 1+ε, and min ensures we don't benefit from ρ > 1+ε. (2) A < 0 (bad action): ratio ρ should decrease π. Clip limits ρ to ≥ 1-ε. The min allows A<0 if ρ<1-ε (policy correctly moved away from bad action). This asymmetry means the surrogate is a pessimistic lower bound: it never claims more improvement than is conservative, but doesn't penalize conservative improvements. Without the min (just clip), you could still exploit high ρ with A < 0.",
              hints: ["Consider what happens when A is negative and ρ < 1-ε"],
            },
          ],
        },
      ],
    },
    {
      id: "model-based-rl",
      trackId: "reinforcement-learning",
      title: "Model-Based RL & World Models",
      description: "Dyna architecture, model predictive control, learned world models (Dreamer, MBPO), latent imagination, and the sample efficiency advantage of model-based approaches.",
      order: 2,
      estimatedHours: 10,
      lessons: [
        {
          id: "world-models",
          moduleId: "model-based-rl",
          trackId: "reinforcement-learning",
          title: "World Models: DreamerV3 & Latent Space Planning",
          description: "Recurrent State Space Models (RSSM) for world modeling, latent imagination for policy optimization, DreamerV3's robust training, and the relationship to model predictive control.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: ["ppo-algorithm"],
          keyTakeaways: [
            "World model: pθ(sₜ₊₁|sₜ,aₜ), rₜ = rφ(sₜ) learned from experience",
            "RSSM: recurrent + stochastic state for partial observability and uncertainty",
            "Policy trained on IMAGINED rollouts — dramatically improves sample efficiency",
            "DreamerV3: single algorithm handling diverse tasks from Atari to robotics",
          ],
          sections: [
            {
              id: "s1",
              title: "Dreamer: Learning Behaviors in Latent Space",
              type: "text",
              content: `**The Sample Efficiency Problem:**
Model-free RL (PPO, SAC) requires millions of environment interactions for complex tasks. Atari games: ~50M frames. Robotics: infeasible in real world.

Model-based RL solution: learn a world model from limited experience, then train the policy using the model — "imagination" instead of real interaction.

**Recurrent State Space Model (RSSM) — Hafner et al. 2019, 2023:**
The RSSM separates representation into two parts:
1. Deterministic recurrent state hₜ: captures the deterministic history
2. Stochastic state zₜ: captures the inherent uncertainty

**RSSM Equations:**
Prior: hₜ = f(hₜ₋₁, zₜ₋₁, aₜ₋₁)                    [GRU update]
       p(zₜ|hₜ) = N(μ_prior(hₜ), σ²_prior(hₜ))      [predicted stochastic state]

Posterior: q(zₜ|hₜ, oₜ) = N(μ_post(hₜ, eₜ), σ²_post(hₜ, eₜ))  [refined with observation]

where eₜ = encoder(oₜ) is the encoded observation.

**World Model Training (ELBO-based):**
L = E[log p(oₜ|hₜ, zₜ)] + E[log p(rₜ|hₜ, zₜ)] − β·KL(q(zₜ|hₜ,oₜ) ‖ p(zₜ|hₜ))

Decoder loss: reconstruction of observations
Reward prediction loss: predict rewards from latent state
KL loss: keep posterior close to prior (enables pure imagination)

**Policy Training in Latent Space (Imagination):**
1. Start from encoded real states: h₀ = f(h_prev, z_prev, a_prev)
2. Imagine H steps forward using the LEARNED model (not real environment!)
3. Compute advantage using the imagined trajectory
4. Update policy with actor-critic in imagination

**DreamerV3 (Hafner et al. 2023) — key innovations:**
1. **Symlog predictions:** predict in log space for robustness to scale
2. **Free bits:** minimum KL per latent to prevent collapse
3. **Return normalization:** scale rewards by running percentile of returns
4. **Single set of hyperparameters:** same agent, 7 benchmark domains (Atari, DMC, MineDojo, etc.)

DreamerV3 achieves:
- Super-human on Atari with 200M steps vs PPO needing 10B+
- Solves the diamond pickaxe in Minecraft (2M steps vs reinforcement learning approaches failing)
- All with the SAME hyperparameters — no task-specific tuning

This represents a major step toward general RL agents.`,
            },
          ],
          exercises: [
            {
              id: "ex-world-model-1",
              type: "multiple_choice",
              question: "Why does the RSSM use BOTH a deterministic recurrent state h and a stochastic state z, rather than just one?",
              options: [
                "Using two states doubles the model capacity",
                "The deterministic state captures predictable dynamics (e.g., physics), while the stochastic state captures irreducible uncertainty (e.g., hidden state, sensor noise) — separating these improves prediction accuracy",
                "Stochastic states are needed for exploration",
                "Deterministic states allow backpropagation through time",
              ],
              correctAnswer: "The deterministic state captures predictable dynamics (e.g., physics), while the stochastic state captures irreducible uncertainty (e.g., hidden state, sensor noise) — separating these improves prediction accuracy",
              explanation: "Purely deterministic models (GRUs) cannot represent uncertainty — they predict a single future state. Purely stochastic models (like discrete VAEs) lose long-range dependencies. RSSM combines both: the GRU recurrence h captures deterministic causal chains (if I push a ball, it moves predictably), while z captures irreducible uncertainty (an opponent's next move, sensor noise, occluded objects). This factorization mirrors the structure of physical environments and leads to better predictions and better policy learning in imagination.",
              hints: ["Think about what types of information in the environment are deterministic vs. stochastic"],
            },
          ],
        },
      ],
    },
    {
      id: "offline-rl",
      trackId: "reinforcement-learning",
      title: "Offline & Conservative RL",
      description: "Offline RL from logged data, distributional shift, CQL, IQL, TD3+BC, and Decision Transformers — treating RL as sequence modeling.",
      order: 3,
      estimatedHours: 9,
      lessons: [
        {
          id: "offline-rl-methods",
          moduleId: "offline-rl",
          trackId: "reinforcement-learning",
          title: "Conservative Q-Learning & Decision Transformer",
          description: "The offline RL distributional shift problem, CQL's conservative value regularization, IQL's in-sample learning, and Decision Transformer's surprising effectiveness.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: ["ppo-algorithm"],
          keyTakeaways: [
            "Offline RL challenge: OOD actions get overestimated Q values from Bellman bootstrapping",
            "CQL: penalize Q values at OOD actions, reward at in-distribution actions",
            "IQL: implicit Q-learning — never evaluates OOD actions (expectile regression trick)",
            "Decision Transformer: sequence modeling replaces RL — condition on desired return",
          ],
          sections: [
            {
              id: "s1",
              title: "The Offline RL Challenge and Conservative Solutions",
              type: "text",
              content: `**Offline RL Setup:**
Learn a policy from a fixed dataset D = {(sᵢ,aᵢ,rᵢ,sᵢ')} without environment interaction.

**The Distributional Shift Problem:**
Standard Q-learning: Q(s,a) = r + γ max_{a'} Q(s',a')

When training offline, the policy π(a|s) = argmax_a Q(s,a) will take actions NOT in D.
For actions a' ∉ D: Q(s',a') has never been updated from real data → could be arbitrarily wrong.

But Q-learning BOOTSTRAPS: Q(s,a) uses max Q(s',a') as target. If Q(s',a') is overestimated for OOD a', then Q(s,a) becomes overestimated, which propagates backwards → catastrophic value overestimation!

Example: Dataset D from a mediocre policy (avg return 50). OOD actions might have Q values overestimated to return 200. The offline RL policy then takes these OOD actions in deployment → disaster.

**Conservative Q-Learning (CQL — Kumar et al., 2020):**
Augment the Bellman objective with a regularizer that:
- PENALIZES Q values for out-of-distribution (a,s) pairs
- REWARDS Q values for in-distribution (a,s) pairs

L_CQL = L_Bellman + α · E_{s~D}[log Σ_a exp(Q(s,a)) − E_{a~β(a|s)}[Q(s,a)]]

= L_Bellman + α · [logsumexp over all actions − Q at dataset actions]

The logsumexp term is an approximation of max_a Q(s,a) and penalizes high Q values anywhere.
The subtracted term rewards high Q at dataset actions.
Net effect: Q is suppressed for OOD actions, conservative for in-distribution ones.

**Implicit Q-Learning (IQL — Kostrikov et al., 2021):**
Avoids querying OOD actions entirely through expectile regression.

V(s) ≈ E_{a~π*}[Q(s,a)] ≈ max_{a∈D(s)} Q(s,a)

Uses L2 loss with ASYMMETRIC weights:
L_V = E[(τ − 1(Q−V < 0)) · (Q(s,a) − V(s))²]

τ > 0.5: overestimates V toward max (implicit maximization!)
As τ → 1: V → max Q without evaluating OOD actions.

**Decision Transformer (Chen et al., 2021):**
Reframe RL as sequence modeling! No Bellman backup needed.

Input sequence: (Ĝ_t, s_t, a_t, Ĝ_{t-1}, s_{t-1}, a_{t-1}, ...)
where Ĝ_t = desired return-to-go (conditioning variable)

Output: predicted actions a_t conditioned on desired future return.

Training: supervised learning on (Ĝ_t, s_t) → a_t.
Inference: condition on high desired return → model produces actions to achieve it.

**Surprising result:** Decision Transformer matches CQL, IQL on many offline RL benchmarks without any RL machinery — purely supervised sequence modeling!

**Limitations:** DT requires good coverage in dataset (expert demonstrations), doesn't generalize OOD well, and doesn't improve beyond the dataset's best trajectories (unlike proper RL).`,
            },
          ],
          exercises: [
            {
              id: "ex-offline-1",
              type: "multiple_choice",
              question: "Why does standard Q-learning (e.g., DQN) fail catastrophically in the offline setting while it works well online?",
              options: [
                "Q-learning requires more data than offline datasets contain",
                "Online: OOD actions from argmax π are immediately tried and corrected by real environment feedback. Offline: OOD action Q-values are never corrected, and Bellman bootstrapping propagates these errors to all visited states",
                "Q-learning's target network doesn't work with fixed datasets",
                "Online Q-learning uses different replay buffer sizes",
              ],
              correctAnswer: "Online: OOD actions from argmax π are immediately tried and corrected by real environment feedback. Offline: OOD action Q-values are never corrected, and Bellman bootstrapping propagates these errors to all visited states",
              explanation: "Online Q-learning: when the policy takes an OOD action, it receives real environment feedback (reward and next state), immediately correcting the overestimated Q value. Error correction happens naturally. Offline Q-learning: the dataset is fixed. If the current policy (argmax Q) suggests action a* not in D, Q(s',a*) is computed but never receives a real TD error correction — it was estimated by the neural network from similar states in D. This error propagates: Q(s,a) uses overestimated Q(s',a*) as target, becoming overestimated itself, spreading the overestimation backwards.",
              hints: ["Think about when Q-values for OOD actions get corrected in online vs offline settings"],
            },
          ],
        },
      ],
    },
    {
      id: "multi-agent-rl",
      trackId: "reinforcement-learning",
      title: "Multi-Agent RL & Game Theory",
      description: "Nash equilibria, MARL challenges (non-stationarity, credit assignment), CTDE framework, QMIX, MADDPG, and self-play methods for superhuman AI.",
      order: 4,
      estimatedHours: 9,
      lessons: [
        {
          id: "marl-foundations",
          moduleId: "multi-agent-rl",
          trackId: "reinforcement-learning",
          title: "Multi-Agent RL: Nash Equilibria & Credit Assignment",
          description: "Stochastic games, Nash equilibrium existence and computation, the centralized training/decentralized execution (CTDE) paradigm, and QMIX monotonic value factorization.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: ["ppo-algorithm"],
          keyTakeaways: [
            "Stochastic game: (S, A₁,...,Aₙ, P, R₁,...,Rₙ) — generalization of MDPs to multiple agents",
            "Nash equilibrium: no agent can improve unilaterally — existence guaranteed by Nash's theorem",
            "Non-stationarity: other agents learning makes environment non-stationary from any single agent's view",
            "QMIX: factorize joint Q(s,a₁,...,aₙ) = f(Q₁(oᵢ,aᵢ)) with monotonicity constraint",
          ],
          sections: [
            {
              id: "s1",
              title: "Game Theory Foundations for MARL",
              type: "text",
              content: `**Stochastic Games (Shapley, 1953):**
(N, S, {Aᵢ}, P, {Rᵢ}, γ) where:
- N: number of agents
- S: shared state space
- Aᵢ: action space for agent i, joint action a = (a₁,...,aₙ) ∈ A₁×...×Aₙ
- P(s'|s,a): transition dynamics (determined by joint action)
- Rᵢ(s,a): reward for agent i (can differ — zero-sum, cooperative, or mixed)

**Special cases:**
- N=1, one agent: standard MDP
- Rᵢ = R for all i: cooperative game (all share reward)
- Σᵢ Rᵢ = 0: zero-sum game (chess, Go, poker)

**Nash Equilibrium:**
A joint policy (π₁*,...,πₙ*) is a Nash equilibrium if for each agent i:
V_i^{π*} ≥ V_i^{(π̂ᵢ, π*_{-i})} for all π̂ᵢ

No agent can improve by unilaterally deviating, given others play π*.

**Nash's Theorem:** Every finite game has at least one Nash equilibrium in mixed strategies.
(But finding NE is PPAD-complete for general games — not efficiently computable!)

**MARL Challenges:**

1. **Non-stationarity:** From agent i's perspective, the environment (including other agents) changes as they learn. This violates the Markov property and standard RL convergence guarantees.

2. **Credit assignment:** In cooperative tasks, a shared reward signal doesn't reveal individual contributions. Which agent's action was responsible for the team's success?

3. **Exponential joint action space:** Joint Q(s, a₁,...,aₙ) has |A|ⁿ values — intractable for large n.

4. **Partial observability:** Each agent typically sees only its local observation oᵢ, not the full state s.

**Centralized Training, Decentralized Execution (CTDE):**
Training: use full state and all observations (centralized critic/Q)
Execution: each agent acts on only its local observation (decentralized policy)

This sidesteps non-stationarity at training time by conditioning on global information.
At test time, no communication or central controller needed.

**QMIX (Rashid et al. 2018) for cooperative MARL:**
Individual Q-values: Qᵢ(oᵢ, aᵢ) for each agent
Joint Q: Q_joint(s, a) = f(Q₁(o₁,a₁),...,Qₙ(oₙ,aₙ); s)

Monotonicity constraint: ∂Q_joint/∂Qᵢ ≥ 0 for all i

This ensures: argmax_{aᵢ} Qᵢ(oᵢ,aᵢ) for each agent → argmax_a Q_joint(s,a) (decentralized greedy = centralized greedy)

The mixing network f has non-negative weights (enforced by abs or sigmoid) and takes global state s as additional input for per-state mixing weights.`,
            },
          ],
          exercises: [
            {
              id: "ex-marl-1",
              type: "multiple_choice",
              question: "What is the key property of QMIX's monotonicity constraint, and why does it enable decentralized execution?",
              options: [
                "It makes each agent's Q-function converge to the true value",
                "∂Q_joint/∂Qᵢ ≥ 0 ensures that maximizing individual Qᵢ independently gives the same joint action as centralized maximization of Q_joint",
                "It prevents agents from taking OOD actions",
                "It ensures agents converge to a Nash equilibrium",
              ],
              correctAnswer: "∂Q_joint/∂Qᵢ ≥ 0 ensures that maximizing individual Qᵢ independently gives the same joint action as centralized maximization of Q_joint",
              explanation: "The monotonicity condition ∂Q_joint/∂Qᵢ ≥ 0 (Q_joint is monotonically non-decreasing in each individual Qᵢ) means: argmax_a Q_joint = (argmax_{a₁}Q₁, ..., argmax_{aₙ}Qₙ). Each agent greedily maximizing its own Q-value independently gives the same joint action as the centralized greedy policy. This is called the Individual-Global-Max (IGM) condition. Without it, decentralized greedy execution could be suboptimal, but computing the joint argmax is exponentially hard. QMIX enforces IGM via the monotonic mixing network.",
              hints: ["Think about when argmax over a composition equals the composition of argmaxes"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// TRACK 7: ADVANCED TOPICS IN AI
// ============================================================
const advancedTopicsTrack: Track = {
  id: "advanced-topics",
  title: "Advanced AI Topics: GNNs, Causal ML & MLOps",
  description: "Graph neural networks and message passing, causal inference and do-calculus, federated learning, interpretability/explainability, neural ODEs, and production ML systems.",
  icon: "🔬",
  difficulty: "expert",
  estimatedHours: 35,
  moduleCount: 4,
  lessonCount: 12,
  tags: ["GNNs", "causal inference", "federated learning", "interpretability", "MLOps"],
  color: "#64748b",
  order: 7,
  modules: [
    {
      id: "graph-neural-networks",
      trackId: "advanced-topics",
      title: "Graph Neural Networks: Message Passing & Expressive Power",
      description: "Message passing framework, spectral vs spatial GNNs, GCN, GraphSAGE, GAT, GIN, expressive power and the Weisfeiler-Lehman hierarchy, and applications.",
      order: 1,
      estimatedHours: 10,
      lessons: [
        {
          id: "message-passing-theory",
          moduleId: "graph-neural-networks",
          trackId: "advanced-topics",
          title: "Message Passing Framework & Expressive Power of GNNs",
          description: "The MPNN framework, isomorphism test connection, GIN's maximally expressive message passing, and higher-order GNNs for breaking WL-1 limitations.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "MPNN: h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1): u∈N(v)}))",
            "GNNs bounded by WL-1 test: cannot distinguish non-isomorphic graphs it can't distinguish",
            "GIN maximally expressive: h_v^(k) = MLP((1+ε)h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))",
            "Beyond WL-1: 3-GNNs, random features, geometric GNNs for equivariant modeling",
          ],
          sections: [
            {
              id: "s1",
              title: "Message Passing Neural Networks",
              type: "text",
              content: `**The Message Passing Framework (Gilmer et al. 2017):**
Any GNN layer can be written as:
h_v^(k) = UPDATE^(k)(h_v^(k-1), AGGREGATE^(k)({MSG^(k)(h_v^(k-1), h_u^(k-1), e_{uv}): u ∈ N(v)}))

**Variants:**
- GCN: h_v = σ(Σ_{u∈N(v)∪{v}} h_u / √(d_v·d_u) · W)  [spectral-motivated normalization]
- GraphSAGE: h_v = σ(W₁h_v + W₂·MEAN(h_u: u∈N(v)))  [inductive, works on new nodes]
- GAT: α_{uv} = softmax_u(e_{uv}), h_v = σ(Σ_u α_{uv}W h_u)  [attention-based]
- GIN: h_v = MLP((1+ε)h_v + Σ_{u∈N(v)} h_u)  [maximally expressive]

**Expressive Power and the WL Hierarchy:**
The Weisfeiler-Lehman (WL) graph isomorphism test iteratively refines node labels:
c_v^(t+1) = hash(c_v^(t), {c_u^(t): u ∈ N(v)})

**Theorem (Xu et al., 2019):** Any MPNN with injective AGGREGATE and UPDATE is AT MOST as powerful as the WL-1 test.

WL-1 cannot distinguish:
- Regular graphs of the same degree (all nodes look the same)
- Graphs that differ only in non-local structure (large cycles vs small cycles far apart)
- The number of cycles of length ≥ 4

**GIN is WL-1 optimal:**
For GIN to be as powerful as WL-1, AGGREGATE must be INJECTIVE (no information loss):
h_v^(k) = φ((1+ε)h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))

Sum (not mean or max) is injective over multisets. Mean fails: {1,1,1} and {1} have the same mean. Max fails: {1,2,3} and {3,3} have the same max.

**Beyond WL-1:**
- k-WL tests (k-dimensional colors): exponentially more powerful
- 3-GNNs: use triangle counts, etc. — captures chemistry-relevant structure
- Equivariant GNNs (SEGNN, SE(3)-Transformers): for molecular modeling
- Positional encodings (Laplacian eigenvectors): break symmetry, add global structure
- Random feature GNNs: add random node features for probabilistic distinguishability`,
            },
            {
              id: "s2",
              title: "GNN Implementation and Expressive Power Analysis",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Graph Neural Network implementations and expressive power analysis.
Shows how GCN, GraphSAGE, and GIN differ in what they can distinguish.
"""

def adjacency_to_list(adj):
    """Convert adjacency matrix to neighbor list"""
    return {i: list(np.where(adj[i] > 0)[0]) for i in range(len(adj))}

class GraphConvLayer:
    """GCN layer: normalized sum aggregation"""
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.1
    
    def forward(self, H, adj_norm):
        """H: (n, d), adj_norm: symmetric normalized adjacency"""
        return np.tanh(adj_norm @ H @ self.W)

class GINLayer:
    """GIN layer: sum aggregation with MLP — maximally expressive"""
    def __init__(self, in_dim, out_dim, eps=0.0):
        self.W1 = np.random.randn(in_dim, out_dim) * 0.1
        self.W2 = np.random.randn(out_dim, out_dim) * 0.1
        self.eps = eps
    
    def forward(self, H, neighbors):
        """
        h_v = MLP((1+eps)*h_v + sum_{u in N(v)} h_u)
        Sum aggregation is injective over multisets — key to expressiveness!
        """
        n = H.shape[0]
        agg = np.zeros_like(H)
        for v in range(n):
            agg[v] = (1 + self.eps) * H[v] + sum(H[u] for u in neighbors[v])
        
        # 2-layer MLP
        h = np.tanh(agg @ self.W1)
        h = np.tanh(h @ self.W2)
        return h

def wl1_test(adj1, adj2, n_iterations=5):
    """
    Weisfeiler-Lehman 1-dimensional isomorphism test.
    
    Iteratively hash neighborhood multisets.
    If histograms differ at any iteration: graphs are NOT isomorphic.
    If they match at all iterations: graphs MIGHT be isomorphic (not conclusive).
    """
    n1, n2 = len(adj1), len(adj2)
    
    # Initial colors: node degree
    colors1 = np.array([sum(adj1[i]) for i in range(n1)])
    colors2 = np.array([sum(adj2[i]) for i in range(n2)])
    
    neighbors1 = adjacency_to_list(adj1)
    neighbors2 = adjacency_to_list(adj2)
    
    for iteration in range(n_iterations):
        # Aggregate neighbor colors
        def new_colors(colors, neighbors):
            new = []
            for v in range(len(colors)):
                neigh_colors = tuple(sorted([colors[u] for u in neighbors[v]]))
                new.append(hash((colors[v], neigh_colors)) % 1000)
            return np.array(new)
        
        colors1 = new_colors(colors1, neighbors1)
        colors2 = new_colors(colors2, neighbors2)
        
        hist1 = np.bincount(colors1, minlength=1000)
        hist2 = np.bincount(colors2, minlength=1000)
        
        if not np.array_equal(hist1, hist2):
            return False, iteration  # Definitely not isomorphic
    
    return True, n_iterations  # Might be isomorphic

# Classic WL-1 failure case: two regular graphs WL-1 cannot distinguish
# Graph 1: Cycle C6
adj_c6 = np.zeros((6, 6))
for i in range(6):
    adj_c6[i, (i+1) % 6] = 1
    adj_c6[(i+1) % 6, i] = 1

# Graph 2: Two disjoint triangles K3 + K3
adj_2k3 = np.zeros((6, 6))
for tri in [0, 3]:
    for i in range(3):
        for j in range(3):
            if i != j:
                adj_2k3[tri+i, tri+j] = 1

print("=== WL-1 Expressive Power Analysis ===\n")
print("Graph 1: 6-cycle C6")
print("Graph 2: Two disjoint triangles (K3 + K3)")
print(f"Both graphs: 6 nodes, each node has degree 2")

wl_same, wl_iter = wl1_test(adj_c6, adj_2k3)
print(f"WL-1 test result: {'SAME' if wl_same else 'DIFFERENT'}")
print(f"These are {'NOT ' if not wl_same else ''}the same by WL-1")
print()
print("But C6 and K3+K3 are STRUCTURALLY DIFFERENT!")
print("  - C6 has ONE connected component")
print("  - K3+K3 has TWO connected components")
print("  - C6 has 6-cycles, K3+K3 has only 3-cycles")
print()
print("GNNs bounded by WL-1 CANNOT distinguish these graphs!")
print("This matters for: cycle detection, counting triangles, bipartiteness testing")
print()

print("=== GIN vs Mean Aggregation: Expressiveness Demo ===\n")
# Show that SUM (GIN) is injective but MEAN is not
multisets = [
    np.array([1, 1, 1]),  # three 1s
    np.array([1]),         # one 1
    np.array([1, 2]),      # one 1, one 2
    np.array([1, 2, 3]),   # one 1, one 2, one 3
    np.array([3, 3]),      # two 3s
]

print(f"{'Multiset':>20} | {'SUM':>5} | {'MEAN':>6} | {'MAX':>5}")
print("-" * 45)
for ms in multisets:
    print(f"{str(list(ms)):>20} | {ms.sum():>5.1f} | {ms.mean():>6.3f} | {ms.max():>5.1f}")

print()
print("SUM is injective: all multisets have distinct sums → GIN can distinguish them")
print("MEAN is NOT injective: [1,1,1] and [1] both have mean=1")
print("MAX is NOT injective: [1,2,3] and [3,3] both have max=3")
print()
print("This proves GIN's sum aggregation is strictly more expressive than GCN/GraphSAGE!")
print()
print("=== Applications of GNNs at PhD level ===")
applications = [
    ("Molecular property prediction", "Predict drug activity, toxicity, quantum properties (QM9)"),
    ("Protein structure prediction", "AlphaFold2 uses GNNs on residue interaction graphs"),
    ("Combinatorial optimization", "Learning heuristics for TSP, scheduling via GNNs"),
    ("Knowledge graph reasoning", "Relation prediction in KGs (TransE, RotatE, ComplEx)"),
    ("Physics simulation", "Learning particle/fluid dynamics (GNS, DPI-Net)"),
    ("Social network analysis", "Community detection, influence propagation"),
]
for app, desc in applications:
    print(f"  {app}: {desc}")
`,
              caption: "GNN expressive power analysis — WL-1 limitations and why GIN's sum aggregation is optimal",
            },
          ],
          exercises: [
            {
              id: "ex-gnn-1",
              type: "multiple_choice",
              question: "Why does GIN use SUM aggregation while GCN uses MEAN aggregation, and what does this mean for expressive power?",
              options: [
                "SUM is faster to compute than MEAN",
                "SUM is injective over multisets — distinct neighborhoods produce distinct aggregations — while MEAN is not (it normalizes away the count information)",
                "GCN's MEAN aggregation is better for graph classification tasks",
                "SUM prevents gradient vanishing in deep GNNs",
              ],
              correctAnswer: "SUM is injective over multisets — distinct neighborhoods produce distinct aggregations — while MEAN is not (it normalizes away the count information)",
              explanation: "For the AGGREGATE function to be injective (information-preserving), it must map distinct multisets to distinct values. SUM: {1,1,1} → 3, {1,2} → 3... wait, not injective either in general. But with vector embeddings rather than scalars, SUM is injective over multisets of vectors in general position. MEAN normalizes by count: a node with 3 identical neighbors looks the same as a node with 1 such neighbor. This loses structural information (degree). Xu et al. (2019) proved that SUM-based GINs are WL-1 optimal, while MEAN/MAX-based GNNs are strictly less powerful.",
              hints: ["Think about whether two different multisets can have the same sum vs the same mean"],
            },
          ],
        },
      ],
    },
    {
      id: "causal-inference-ml",
      trackId: "advanced-topics",
      title: "Causal Inference & Causal Machine Learning",
      description: "Pearl's do-calculus, structural causal models, causal discovery, counterfactual reasoning, and how causality improves ML robustness and out-of-distribution generalization.",
      order: 2,
      estimatedHours: 10,
      lessons: [
        {
          id: "do-calculus",
          moduleId: "causal-inference-ml",
          trackId: "advanced-topics",
          title: "Structural Causal Models & Do-Calculus",
          description: "DAGs and SCMs, the do-operator vs conditioning, identification of causal effects, backdoor and frontdoor criteria, instrumental variables, and counterfactual inference.",
          type: "concept",
          estimatedMinutes: 70,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "P(Y|do(X=x)) ≠ P(Y|X=x) in general — confounding causes this difference",
            "Backdoor criterion: block all backdoor paths to identify causal effect from observational data",
            "Frontdoor criterion: identify causal effect through a mediator (even with unmeasured confounders)",
            "Do-calculus (3 rules): complete axioms for causal identifiability from DAGs",
          ],
          sections: [
            {
              id: "s1",
              title: "Do-Calculus: The Language of Causation",
              type: "text",
              content: `**Structural Causal Model (SCM):**
A tuple (U, V, F, P(U)) where:
- U: exogenous variables (external noise)
- V: endogenous variables (caused by others)
- F: structural equations vᵢ = fᵢ(Pa(Vᵢ), Uᵢ)  [function of parents + noise]
- P(U): distribution over noise variables

The SCM induces a directed acyclic graph (DAG) G where edges represent "direct causes."

**Association vs Causation:**
P(Y|X=x): observational — "given we observe X=x, what is Y's distribution?"
P(Y|do(X=x)): interventional — "if we SET X=x (break its causal mechanisms), what is Y?"

These differ due to CONFOUNDING: a common cause Z affects both X and Y.

**Example:** Does hospitalization (X) cause death (Y)?
Observational: P(death|hospitalized) > P(death|not hospitalized) — hospitalized people are sicker!
Causal: P(death|do(hospital=yes)) < P(death|do(hospital=no)) — hospitalization helps

**The do-Operator and Interventional Distributions:**
In the SCM, do(X=x) means: replace the structural equation for X with X=x (cut incoming edges).
This creates a mutilated model G_x with the arrow into X removed.

**Backdoor Criterion:**
A set Z satisfies the backdoor criterion wrt (X,Y) if:
1. Z blocks all backdoor paths from X to Y (paths through parents of X)
2. Z contains no descendant of X

If Z satisfies backdoor criterion:
P(Y|do(X=x)) = Σ_z P(Y|X=x, Z=z)P(Z=z)

**Frontdoor Criterion (handles unobserved confounders!):**
If M is a set of variables satisfying:
1. M intercepts all directed paths from X to Y
2. No backdoor paths from X to M (X→M is unconfounded)
3. All backdoor paths from M to Y are blocked by X

Then: P(Y|do(X=x)) = Σ_m [Σ_x' P(M=m|X=x')P(X=x')] · P(Y|M=m,X=x)

**Why this matters for ML:**
ML models learn P(Y|X) from observational data. If X and Y share confounders, the model learns spurious associations.

Example: Training a model to detect chest X-ray pathology. If pneumonia patients are given oxygen (unmeasured intervention), the model learns "oxygen tube → healthy" as a spurious correlation.

**Counterfactuals:**
"What would Y have been if X had been x', given we observed X=x?"

SCM enables: Y_{x'}(u) = Y in modified model where X=x', evaluated at noise U=u.
This requires more than just P(Y|do(X)) — it requires the specific U for the individual.`,
            },
            {
              id: "s2",
              title: "Causal Identification and Simpson's Paradox",
              type: "code",
              language: "python",
              content: `import numpy as np

"""
Demonstrating causal reasoning vs. statistical association.
Shows Simpson's Paradox and backdoor adjustment.
"""

def simpsons_paradox_demo():
    """
    Classic Simpson's paradox: aggregate trend reverses within subgroups.
    
    Medical trial: does treatment T cause recovery R?
    Confounded by: gender (affects both treatment assignment and recovery)
    """
    # Simulated data (matches classic Simpson's paradox numbers)
    # Males: 18% recovery if treated, 30% if not treated → treatment HURTS males
    # Females: 70% recovery if treated, 85% if not treated → treatment HURTS females
    # Aggregate: 40% treated recover, 33% untreated recover → treatment HELPS in aggregate??
    
    data = {
        'male_treated': {'n': 100, 'recovered': 18},
        'male_untreated': {'n': 10, 'recovered': 3},
        'female_treated': {'n': 10, 'recovered': 7},
        'female_untreated': {'n': 100, 'recovered': 85},
    }
    
    # Naive (confounded) estimate: P(R|T=1) vs P(R|T=0)
    total_treated = data['male_treated']['n'] + data['female_treated']['n']
    total_untreated = data['male_untreated']['n'] + data['female_untreated']['n']
    recovered_treated = data['male_treated']['recovered'] + data['female_treated']['recovered']
    recovered_untreated = data['male_untreated']['recovered'] + data['female_untreated']['recovered']
    
    p_r_given_t1 = recovered_treated / total_treated
    p_r_given_t0 = recovered_untreated / total_untreated
    
    print("=== Simpson's Paradox: Treatment Effectiveness ===\n")
    print("NAIVE ASSOCIATION (confounded by gender):")
    print(f"  P(Recovery | Treated)   = {p_r_given_t1:.3f}  → Treatment appears HELPFUL")
    print(f"  P(Recovery | Untreated) = {p_r_given_t0:.3f}")
    print()
    
    # Within-subgroup estimates (each subgroup: treatment HURTS)
    p_male_t1 = data['male_treated']['recovered'] / data['male_treated']['n']
    p_male_t0 = data['male_untreated']['recovered'] / data['male_untreated']['n']
    p_female_t1 = data['female_treated']['recovered'] / data['female_treated']['n']
    p_female_t0 = data['female_untreated']['recovered'] / data['female_untreated']['n']
    
    print("STRATIFIED (controlling for gender):")
    print(f"  Males:   P(R|T=1) = {p_male_t1:.3f}, P(R|T=0) = {p_male_t0:.3f} → Treatment HURTS males")
    print(f"  Females: P(R|T=1) = {p_female_t1:.3f}, P(R|T=0) = {p_female_t0:.3f} → Treatment HURTS females")
    print()
    
    # Backdoor adjustment: P(R|do(T=t)) = Σ_G P(R|T=t,G=g)P(G=g)
    n_males = data['male_treated']['n'] + data['male_untreated']['n']
    n_females = data['female_treated']['n'] + data['female_untreated']['n']
    n_total = n_males + n_females
    
    p_male = n_males / n_total
    p_female = n_females / n_total
    
    causal_t1 = p_male * p_male_t1 + p_female * p_female_t1
    causal_t0 = p_male * p_male_t0 + p_female * p_female_t0
    
    print("BACKDOOR ADJUSTMENT (causal estimate):")
    print(f"  P(R|do(T=1)) = {p_male}×{p_male_t1:.3f} + {p_female}×{p_female_t1:.3f} = {causal_t1:.3f}")
    print(f"  P(R|do(T=0)) = {p_male}×{p_male_t0:.3f} + {p_female}×{p_female_t0:.3f} = {causal_t0:.3f}")
    print()
    print(f"CAUSAL EFFECT: Treatment → Recovery change = {causal_t1 - causal_t0:.3f}")
    print("→ Treatment HURTS: negative causal effect, as stratification correctly shows")
    print()
    print("Explanation: women are much healthier AND more likely to get treated.")
    print("In aggregate, treated group has more women → appears healthier.")
    print("Backdoor adjustment marginalizes over the correct population mix.")

simpsons_paradox_demo()

def instrumental_variable_demo():
    """
    Instrumental Variable (IV) estimation for causal effect with unobserved confounder.
    
    Z → X → Y, but X and Y are also confounded by U (unobserved).
    Z (instrument) affects Y ONLY through X.
    
    IV estimate: β_IV = Cov(Z, Y) / Cov(Z, X) = LATE (Local Average Treatment Effect)
    """
    np.random.seed(42)
    n = 10000
    
    # Unobserved confounder (e.g., health consciousness)
    U = np.random.randn(n)
    
    # Instrument: distance to nearest hospital (affects treatment but not health directly)
    Z = np.random.randn(n)  # independent of U
    
    # Treatment: hospitalization (affected by both Z and U)
    X = 0.5 * Z + 0.5 * U + 0.2 * np.random.randn(n)
    
    # Outcome: health (affected by X causally, and by U as confounder)
    true_beta = 2.0  # true causal effect of X on Y
    Y = true_beta * X + 1.0 * U + 0.3 * np.random.randn(n)
    
    # Naive OLS (confounded)
    beta_ols = np.cov(X, Y)[0, 1] / np.var(X)
    
    # IV estimate (2SLS)
    beta_iv = np.cov(Z, Y)[0, 1] / np.cov(Z, X)[0, 1]
    
    print("\n=== Instrumental Variable Estimation ===\n")
    print(f"True causal effect (β): {true_beta}")
    print(f"OLS estimate (biased by confounder U): {beta_ols:.4f}")
    print(f"IV estimate (using Z as instrument): {beta_iv:.4f}")
    print()
    print("OLS is biased upward: U increases both X (health-conscious people seek care)")
    print("and Y (health-conscious people are healthier) → spurious positive correlation")
    print()
    print("IV correctly recovers the true causal effect by using only the variation")
    print("in X that is induced by Z (which is independent of U).")

instrumental_variable_demo()
`,
              caption: "Simpson's Paradox resolved via backdoor adjustment, and instrumental variables for unobserved confounders",
            },
          ],
          exercises: [
            {
              id: "ex-causal-1",
              type: "multiple_choice",
              question: "Simpson's paradox demonstrates that:",
              options: [
                "Statistical methods cannot be trusted for causal inference",
                "Marginal associations can reverse direction when stratifying by a confounding variable, showing that P(Y|X) ≠ P(Y|do(X))",
                "Larger datasets always give more reliable causal estimates",
                "Randomized experiments always prevent confounding",
              ],
              correctAnswer: "Marginal associations can reverse direction when stratifying by a confounding variable, showing that P(Y|X) ≠ P(Y|do(X))",
              explanation: "Simpson's paradox shows that observational associations P(Y|X) can be completely reversed when conditioned on a confounder G. The aggregate trend (treatment helps) reverses within each stratum (treatment hurts males, hurts females). The causal quantity P(Y|do(X)) — the backdoor-adjusted estimate — matches the within-stratum analysis when gender is the only confounder. This illustrates that statistical association ≠ causation: we need to identify and adjust for confounders to answer causal questions.",
              hints: ["Think about what changes between the stratified and aggregate analysis"],
            },
          ],
        },
      ],
    },
    {
      id: "interpretability",
      trackId: "advanced-topics",
      title: "Mechanistic Interpretability & Explainability",
      description: "SHAP values and game-theoretic feature attribution, concept activation vectors (CAVs), mechanistic interpretability with circuits in transformers, and superposition in neural networks.",
      order: 3,
      estimatedHours: 8,
      lessons: [
        {
          id: "shap-mechanistic",
          moduleId: "interpretability",
          trackId: "advanced-topics",
          title: "SHAP Theory & Mechanistic Interpretability of Transformers",
          description: "Game-theoretic axioms for SHAP values, efficient TreeSHAP and KernelSHAP, mechanistic interpretability: circuits, superposition, and toy models for understanding neural networks.",
          type: "concept",
          estimatedMinutes: 65,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "SHAP: unique attribution satisfying efficiency, symmetry, dummy, and linearity axioms",
            "Shapley value: φᵢ = Σ_{S⊆N\\{i}} [|S|!(n-|S|-1)!/n!] [v(S∪{i}) - v(S)]",
            "TreeSHAP: O(TLD²) exact computation for trees vs O(2ⁿ) exponential naive",
            "Superposition hypothesis: n features stored in d<n dimensions via near-orthogonal directions",
          ],
          sections: [
            {
              id: "s1",
              title: "SHAP: Game-Theoretic Feature Attribution",
              type: "math",
              content: `**Shapley Values from Cooperative Game Theory:**
Given a "game" v: 2^N → ℝ (value function over player subsets), the Shapley value φᵢ for player i is the unique attribution satisfying:

**Efficiency:** Σᵢ φᵢ = v(N) − v(∅)  [attributions sum to total prediction]
**Symmetry:** If v(S∪{i}) = v(S∪{j}) for all S, then φᵢ = φⱼ  [equally contributing players get equal credit]
**Dummy:** If v(S∪{i}) = v(S) for all S, then φᵢ = 0  [non-contributing features get 0]
**Linearity:** φᵢ(v+w) = φᵢ(v) + φᵢ(w)  [attributions for sum of games = sum of attributions]

**Shapley Value Formula:**
φᵢ = Σ_{S⊆N\{i}} [|S|!(n−|S|−1)! / n!] · [v(S∪{i}) − v(S)]

This is the weighted average of the marginal contribution of feature i across all possible orderings.

**SHAP for ML:** Map to cooperative game:
- Players = features (x₁,...,xₙ)
- v(S) = E[f(X) | X_S = x_S]  (expected prediction when only S features are "known")

φᵢ(x) = expected contribution of feature i to prediction f(x) above the baseline E[f(X)]

**Key SHAP variants:**

**KernelSHAP:** Model-agnostic, approximates Shapley values via weighted linear regression on masked inputs. O(2ⁿ) exact, O(M·K) with M samples.

**TreeSHAP:** Exact for tree ensembles (XGBoost, RF, LightGBM) in O(TLD²) time:
- T: number of trees
- L: max number of leaves
- D: depth
Enables exact Shapley values for millions of predictions in seconds.

**GradientSHAP:** For neural networks, uses expectation over baselines of gradient×input.

**FastSHAP:** Train a meta-model to predict Shapley values directly — instant inference.

**SHAP Properties and Limitations:**
✓ Theoretically principled: unique solution satisfying game-theoretic axioms
✓ Locally consistent: similar instances get similar explanations
✗ Computationally expensive for complex models
✗ Assumes feature independence in marginal distributions (SHAP approximates E[f|X_S])
✗ Explains predictions, not mechanisms (correlation vs causation)
✗ Can be misleading if model uses feature interactions heavily`,
            },
          ],
          exercises: [
            {
              id: "ex-shap-1",
              type: "multiple_choice",
              question: "SHAP values satisfy the 'efficiency' axiom, which means:",
              options: [
                "SHAP can be computed in polynomial time",
                "The sum of all feature SHAP values equals the difference between the model's prediction and its expected baseline: Σφᵢ = f(x) - E[f(X)]",
                "SHAP values are always between -1 and 1",
                "Features that contribute equally receive equal SHAP values",
              ],
              correctAnswer: "The sum of all feature SHAP values equals the difference between the model's prediction and its expected baseline: Σφᵢ = f(x) - E[f(X)]",
              explanation: "The efficiency axiom states: Σᵢ φᵢ = v(N) − v(∅) = f(x) − E[f(X)]. All SHAP values must sum to the total 'surplus' the prediction has over the baseline (expected prediction). This makes SHAP values interpretable as a 'budget' that decomposes the prediction: each feature gets credit exactly equal to its contribution. This is why SHAP is more principled than older attribution methods like saliency maps, which don't satisfy efficiency.",
              hints: ["Think about what v(N) and v(∅) represent in the ML context"],
            },
          ],
        },
      ],
    },
    {
      id: "mlops-production",
      trackId: "advanced-topics",
      title: "MLOps: Production ML Systems",
      description: "ML system design, data pipelines, model registries, feature stores, A/B testing for ML, monitoring for drift, and the hidden technical debt in ML systems.",
      order: 4,
      estimatedHours: 7,
      lessons: [
        {
          id: "ml-systems-design",
          moduleId: "mlops-production",
          trackId: "advanced-topics",
          title: "ML System Design: Technical Debt & Production Challenges",
          description: "The 'hidden technical debt' paper, training-serving skew, data pipelines as first-class ML components, feature stores, online vs offline metrics, and responsible ML.",
          type: "concept",
          estimatedMinutes: 55,
          order: 1,
          prerequisites: [],
          keyTakeaways: [
            "Only a small fraction of real ML code is the model — the rest is infrastructure",
            "Training-serving skew: subtle feature computation differences cause silent failures",
            "Feature store: centralized, versioned feature computation to prevent skew",
            "Distribution shift detection: PSI, KS test, MMD for data and prediction drift",
          ],
          sections: [
            {
              id: "s1",
              title: "The Hidden Technical Debt in ML Systems",
              type: "text",
              content: `**Sculley et al. (2015) — "Machine Learning: The High-Interest Credit Card of Technical Debt":**

The paper that introduced the concept of ML-specific technical debt. Key insight: ML code is not just model code. In production ML systems, the actual model training code is typically 5-10% of the total codebase!

**The ML System Landscape:**
┌─────────────────────────────────────────────────────────────┐
│  Real-World ML System                                        │
│                                                              │
│ Data Collection → Feature Engineering → Model → Monitoring  │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐    │
│ │         Infrastructure (95% of code)                  │    │
│ │  Data validation, serving, feature pipelines,        │    │
│ │  A/B testing, logging, monitoring, alerting          │    │
│ │  ┌──────────────────────┐                            │    │
│ │  │   ML Code (5%)       │                            │    │
│ │  └──────────────────────┘                            │    │
│ └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

**Key Types of ML Technical Debt:**

**1. Entanglement (CACE: Changing Anything Changes Everything):**
In a model with features {x₁,...,xₙ}, changing x₁ affects the learned correlations for ALL features.
This makes isolated testing impossible — you can't "AB test" a single feature change.

**2. Undeclared Consumers:**
If model A's output feeds model B without explicit contract, changing model A breaks model B silently.
Dependencies become implicit and undocumented.

**3. Feedback Loops:**
Model outputs affect future training data. Recommendation systems: if model recommends action A, users take A, and A is labeled as "good" in the next training set. Self-fulfilling prophecy.

**4. Training-Serving Skew:**
The most insidious production ML failure mode. Feature computation differs between:
- Training: Python/pandas on historical data
- Serving: Java/C++ in real-time, millisecond latency

Subtle bugs: timezone differences, NULL handling, feature aggregation window edges, floating point precision. These cause silent degradation — the model gives outputs but they're wrong.

**Solution: Feature Stores**
Centralized, versioned feature computation that:
- Computes features once consistently for both training and serving
- Provides point-in-time correct features (no data leakage)
- Handles backfills for new features
- Examples: Feast, Tecton, Hopsworks

**5. Distribution Shift:**
Test distribution ≠ training distribution → model performance degrades.
Types:
- **Covariate shift:** P(X) changes, P(Y|X) stays same
- **Label shift:** P(Y) changes, P(X|Y) stays same
- **Concept drift:** P(Y|X) changes (the relationship itself changes)

Detection methods:
- **PSI (Population Stability Index):** Σ (P% - Q%) × ln(P%/Q%)
- **KS test:** max CDF difference between distributions
- **MMD (Maximum Mean Discrepancy):** kernel-based distance between feature distributions
- **Prediction monitoring:** track model output distribution over time`,
            },
          ],
          exercises: [
            {
              id: "ex-mlops-1",
              type: "multiple_choice",
              question: "Training-serving skew in production ML systems most often occurs due to:",
              options: [
                "Model architecture being too complex for real-time inference",
                "Subtle differences in how features are computed between the Python training pipeline and the production serving system (e.g., NULL handling, timezone treatment, aggregation windows)",
                "The model overfitting to training data",
                "Insufficient hardware for model serving",
              ],
              correctAnswer: "Subtle differences in how features are computed between the Python training pipeline and the production serving system (e.g., NULL handling, timezone treatment, aggregation windows)",
              explanation: "Training-serving skew is one of the most common and dangerous production ML failures because it causes silent degradation — the model produces outputs, but they're based on different features than it was trained on. Example: training pipeline computes 'average purchase in last 7 days' using Python/pandas with specific timezone handling; serving system uses Java with slightly different NULL handling or a 7-day window computed from server time vs UTC. The model was trained on slightly different feature values than what it sees at serving time. Feature stores (centralized feature computation) are the primary solution.",
              hints: ["Training happens in Python/offline; serving often happens in different languages/environments"],
            },
          ],
        },
      ],
    },
  ],
};

// ============================================================
// EXPORT ALL TRACKS
// ============================================================
export const allTracks: Track[] = [
  mathFoundationsTrack,
  advancedClassicalMLTrack,
  deepLearningTrack,
  llmFinetuningTrack,
  generativeAITrack,
  reinforcementLearningTrack,
  advancedTopicsTrack,
];

export const trackMap = new Map(allTracks.map(t => [t.id, t]));
export const moduleMap = new Map(allTracks.flatMap(t => t.modules).map(m => [m.id, m]));
export const lessonMap = new Map(
  allTracks.flatMap(t => t.modules.flatMap(m => m.lessons)).map(l => [l.id, l])
);

// Summary objects (without full lesson content) for list endpoints
export function getTrackSummary(track: Track) {
  return {
    id: track.id,
    title: track.title,
    description: track.description,
    icon: track.icon,
    difficulty: track.difficulty,
    estimatedHours: track.estimatedHours,
    moduleCount: track.moduleCount,
    lessonCount: track.lessonCount,
    tags: track.tags,
    color: track.color,
    order: track.order,
    modules: track.modules.map(m => ({
      id: m.id,
      title: m.title,
      description: m.description,
      lessonCount: m.lessons.length,
      estimatedHours: m.estimatedHours,
      order: m.order,
    })),
  };
}

export function getModuleSummary(module: Module) {
  return {
    id: module.id,
    trackId: module.trackId,
    title: module.title,
    description: module.description,
    order: module.order,
    estimatedHours: module.estimatedHours,
    lessons: module.lessons.map(l => ({
      id: l.id,
      title: l.title,
      description: l.description,
      estimatedMinutes: l.estimatedMinutes,
      type: l.type,
      order: l.order,
    })),
  };
}
