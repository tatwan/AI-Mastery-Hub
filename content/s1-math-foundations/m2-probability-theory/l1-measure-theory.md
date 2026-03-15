---
title: "Probability Spaces & Measure Theory"
estimatedMinutes: 30
tags: ["measure-theory", "sigma-algebra", "probability-space", "random-variables", "expectation"]
prerequisites: []
---

# Probability Spaces & Measure Theory

## Why Measure Theory?

If you have only worked with discrete distributions (PMFs) and continuous distributions (PDFs), you might wonder why anyone would bother with the abstraction of measure theory. The answer becomes clear the moment you encounter a distribution that is neither purely discrete nor purely continuous — a mixture model with point masses and a smooth component, for instance, or the output distribution of a ReLU network (which places positive probability on exactly zero and spreads the rest continuously over $(0, \infty)$ for a symmetric input distribution such as a Gaussian, for example).

Classical probability, built on counting arguments for discrete spaces and Riemann integration for continuous ones, cannot handle such objects cleanly. Measure theory provides a single framework that subsumes both cases and every hybrid in between. More practically, the convergence theorems of Lebesgue integration (dominated convergence, monotone convergence) are the formal engine behind differentiating under the integral sign — the operation you invoke every time you compute a gradient of an expected loss. PAC learning bounds, variational inference, and the entire theory of diffusion models all rest on this foundation.

## The Probability Space $(\Omega, \mathcal{F}, P)$

A probability space consists of three objects:

1. **Sample space** $\Omega$ — the set of all possible outcomes. For a coin flip, $\Omega = \{H, T\}$. For a continuous signal, $\Omega$ might be a function space.

2. **σ-algebra** $\mathcal{F}$ — a collection of subsets of $\Omega$ that we agree to call "measurable events." It must satisfy three axioms:
   - $\emptyset \in \mathcal{F}$
   - If $A \in \mathcal{F}$, then $A^c \in \mathcal{F}$ (closed under complement)
   - If $A_1, A_2, \ldots \in \mathcal{F}$, then $\bigcup_{i=1}^\infty A_i \in \mathcal{F}$ (closed under countable union)

> **Intuition:** The σ-algebra is the collection of events we are allowed to assign probability to — not every subset qualifies. Think of it as the "vocabulary" of questions your probability model can answer. On uncountable spaces like $\mathbb{R}$, some pathological subsets cannot be assigned a consistent probability, so the σ-algebra excludes them.

3. **Probability measure** $P: \mathcal{F} \to [0,1]$ — assigns probabilities to events. It must satisfy $P(\Omega) = 1$ and **countable additivity**: for disjoint $A_1, A_2, \ldots \in \mathcal{F}$,

$$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

In words: the probability of a countable disjoint union equals the sum of the individual probabilities. This is strictly stronger than finite additivity and is what makes the theory work for limits and infinite sequences.

> **Key insight:** The σ-algebra $\mathcal{F}$ determines what questions you are allowed to ask. If an event $A \notin \mathcal{F}$, the probability $P(A)$ is simply undefined. This is not a technicality — it is how we model partial information in filtrations and conditional expectation.

## σ-Algebras and the Borel σ-Algebra

Why can we not just declare every subset of $\Omega$ to be measurable? For finite or countable $\Omega$, we can — the power set $2^\Omega$ is a perfectly good σ-algebra. The trouble arises on uncountable spaces like $\mathbb{R}$. Vitali's construction shows that if we assume every subset of $[0,1]$ is measurable and translation-invariant, we reach a contradiction with countable additivity. Some subsets of $\mathbb{R}$ must be excluded.

The standard resolution is the **Borel σ-algebra** $\mathcal{B}(\mathbb{R})$, defined as the smallest σ-algebra containing all open intervals $(a, b)$. It also contains all closed sets, countable intersections and unions of open/closed sets, and essentially every set you will ever encounter in practice. The non-measurable sets are pathological constructions requiring the axiom of choice.

For machine learning, the Borel σ-algebra is almost always the right choice. When we write $X \sim \mathcal{N}(0, 1)$, we implicitly mean that $X$ is a measurable function from $(\Omega, \mathcal{F})$ to $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.

## Random Variables as Measurable Functions

A random variable is not a "variable" in the algebraic sense — it is a **function**. Formally, $X: \Omega \to \mathbb{R}$ is a random variable if it is **measurable**, meaning:

$$\{ω \in \Omega : X(ω) \leq x\} \in \mathcal{F} \quad \text{for all } x \in \mathbb{R}$$

This condition ensures that the cumulative distribution function $F(x) = P(X \leq x)$ is well-defined, since the event $\{X \leq x\}$ belongs to $\mathcal{F}$ and can therefore be assigned a probability.

The measurability requirement may seem pedantic, but it enforces a critical constraint: the random variable must be "compatible" with the information structure encoded by $\mathcal{F}$. This becomes essential when we discuss conditional expectation and filtrations below.

## Expectation as Lebesgue Integration

> **Refresher:** Unlike the Riemann integral, which slices the domain (x-axis) into thin vertical strips, the Lebesgue integral slices the range (y-axis) — grouping together all inputs that produce similar output values. This reorganization is what allows the Lebesgue integral to handle discontinuous and complex functions seamlessly, and it is what makes probability theory work for distributions that are neither purely discrete nor purely continuous.

The expectation of a random variable $X$ is defined as the **Lebesgue integral** with respect to the probability measure $P$:

$$\mathbb{E}[X] = \int_\Omega X(\omega) \, dP(\omega)$$

This single definition unifies discrete and continuous cases. For a discrete random variable taking values $x_1, x_2, \ldots$ with probabilities $p_1, p_2, \ldots$, it reduces to $\sum_i x_i p_i$. For a continuous random variable with density $f$, it reduces to $\int_{-\infty}^{\infty} x f(x) \, dx$. For mixed distributions, the Lebesgue integral handles both components seamlessly.

The construction proceeds by first defining the integral for simple functions (finite linear combinations of indicator functions), then extending to non-negative measurable functions via suprema of simple approximations, and finally to general functions by writing $X = X^+ - X^-$.

> **Key insight:** The Lebesgue integral's power lies not in computing different numbers than the Riemann integral (they agree when both exist), but in its superior convergence properties. You can interchange limits and integrals under much weaker conditions.

## The Dominated Convergence Theorem

The most important convergence theorem for ML practitioners is the **Dominated Convergence Theorem (DCT)**:

> **Intuition:** The DCT says you can swap the limit and the integral — move $\lim$ inside $\mathbb{E}$ — whenever the sequence is dominated by something integrable. This is the formal license for differentiating under the integral sign, which you do every time you compute a policy gradient or ELBO gradient. Without it, those interchanges are unjustified.

**Theorem.** If $X_n \to X$ almost surely, and there exists an integrable random variable $Y$ with $|X_n| \leq Y$ a.s. for all $n$, then:

$$\lim_{n \to \infty} \mathbb{E}[X_n] = \mathbb{E}\left[\lim_{n \to \infty} X_n\right] = \mathbb{E}[X]$$

In plain English: if a sequence of random variables converges pointwise and is uniformly bounded by something integrable, then you can swap the limit and the expectation.

This theorem justifies **differentiating under the integral sign**. When we write:

$$\nabla_\theta \mathbb{E}_{p_\theta}[f(X)] = \mathbb{E}_{p_\theta}[f(X) \nabla_\theta \log p_\theta(X)]$$

(the REINFORCE / score function estimator), the interchange of $\nabla_\theta$ and $\mathbb{E}$ is valid precisely because the DCT conditions hold. Without this, the gradients used in policy gradient methods and variational inference would be formally unjustified.

The closely related **Monotone Convergence Theorem** handles the case where $X_n$ is non-decreasing: if $0 \leq X_1 \leq X_2 \leq \ldots$ and $X_n \to X$ a.s., then $\mathbb{E}[X_n] \to \mathbb{E}[X]$ (possibly $+\infty$). No dominating function is needed, but monotonicity is required.

### Fatou's Lemma

Completing the convergence trilogy, **Fatou's Lemma** provides a lower bound without requiring a dominating function:

> **Remember:** $\mathbb{E}[\liminf_n X_n] \leq \liminf_n \mathbb{E}[X_n]$ for any non-negative sequence. The inequality can be strict — the limit of expectations can exceed the expectation of the limit when probability "leaks to infinity." Use Fatou's Lemma when you have non-negativity but no dominating function and need a lower bound on the limiting expectation.

$$\mathbb{E}\left[\liminf_{n \to \infty} X_n\right] \leq \liminf_{n \to \infty} \mathbb{E}[X_n]$$

for any sequence of non-negative random variables $X_n \geq 0$.

Unlike DCT, Fatou's Lemma requires no integrability condition — only non-negativity. The inequality can be strict: the limit of expectations can exceed the expectation of the limit (probability "leaks to infinity"). Together with MCT and DCT, Fatou's Lemma covers the full landscape of when limits and expectations can be exchanged — a question that arises constantly in ML when analyzing convergence of stochastic algorithms.

## Filtrations and Information

> **Intuition:** A filtration is how information accumulates over time. $\mathcal{F}_t$ captures everything that is knowable up to time $t$ — the history of all events that have occurred. As $t$ increases, the filtration can only grow: you never forget information. A random variable is $\mathcal{F}_t$-measurable exactly when its value is fully determined by the information available at time $t$, with no peeking into the future.

A **filtration** $\{\mathcal{F}_t\}_{t \geq 0}$ is an increasing sequence of σ-algebras:

$$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \mathcal{F}_2 \subseteq \ldots \subseteq \mathcal{F}$$

Think of $\mathcal{F}_t$ as the "information available at time $t$." As $t$ increases, more events become distinguishable. A random variable $X$ is **$\mathcal{F}_t$-measurable** if its value is entirely determined by the information in $\mathcal{F}_t$ — you can compute $X$ without looking into the future.

This formalism is the backbone of sequential decision-making. In reinforcement learning, the filtration $\mathcal{F}_t = \sigma(s_0, a_0, r_0, \ldots, s_t)$ encodes the agent's history up to time $t$. The policy $\pi(a|s_t)$ must be $\mathcal{F}_t$-measurable — the agent cannot condition on future states.

## Independence and Conditional Expectation

Two random variables $X$ and $Y$ are **independent** ($X \perp Y$) if:

$$P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B) \quad \text{for all measurable } A, B$$

This is a statement about the joint measure factoring as a product — it says nothing about correlation (which only captures linear dependence).

**Conditional expectation** $\mathbb{E}[X | \mathcal{G}]$ for a sub-σ-algebra $\mathcal{G} \subseteq \mathcal{F}$ is defined as the unique (a.s.) $\mathcal{G}$-measurable random variable satisfying:

$$\int_G \mathbb{E}[X|\mathcal{G}] \, dP = \int_G X \, dP \quad \text{for all } G \in \mathcal{G}$$

Geometrically, $\mathbb{E}[X|\mathcal{G}]$ is the **orthogonal projection** of $X$ onto the space of $\mathcal{G}$-measurable $L^2$ functions. It is the best approximation to $X$ given the information in $\mathcal{G}$, in the mean-squared-error sense. This projection interpretation is central to understanding why conditional expectations minimize squared loss and why they appear throughout Bayesian inference and filtering.

> **Key insight:** Conditional expectation is not a number — it is a random variable. It depends on the conditioning information $\mathcal{G}$, which itself may be random. The familiar $\mathbb{E}[X|Y=y]$ is the special case where $\mathcal{G} = \sigma(Y)$.

## Python: Simulating Measure-Theoretic Concepts

The following code demonstrates the Dominated Convergence Theorem numerically. We construct a sequence of random variables that converge pointwise and verify that the expectations converge accordingly.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N_samples = 50_000

# Sequence X_n(omega) = omega^n on Omega = [0, 1] with Lebesgue measure
# X_n -> 0 a.s. (except at omega=1, a null set)
# Dominating function: Y(omega) = 1 (integrable on [0,1])
omega = np.random.uniform(0, 1, N_samples)

ns = np.arange(1, 51)
empirical_means = []
true_means = []  # E[X_n] = integral of x^n dx from 0 to 1 = 1/(n+1)

for n in ns:
    X_n = omega ** n
    empirical_means.append(X_n.mean())
    true_means.append(1.0 / (n + 1))

# DCT guarantees lim E[X_n] = E[lim X_n] = E[0] = 0
plt.figure(figsize=(8, 4))
plt.plot(ns, empirical_means, 'o-', ms=3, label='Empirical $\\mathbb{E}[X_n]$')
plt.plot(ns, true_means, 's-', ms=3, label='True $1/(n+1)$')
plt.axhline(0, color='red', ls='--', label='$\\mathbb{E}[\\lim X_n] = 0$')
plt.xlabel('n')
plt.ylabel('$\\mathbb{E}[X_n]$')
plt.title('Dominated Convergence: limit and expectation commute')
plt.legend()
plt.tight_layout()
plt.show()
```

This simulation constructs $X_n(\omega) = \omega^n$ on $\Omega = [0,1]$ with Lebesgue measure (i.e., uniform distribution). Each $X_n$ converges pointwise to zero (except at the single point $\omega = 1$, which has measure zero), and $|X_n| \leq 1$ provides the dominating function. The DCT guarantees $\mathbb{E}[X_n] = 1/(n+1) \to 0 = \mathbb{E}[\lim X_n]$, which the empirical means confirm.

:::quiz
question: "Why do we need σ-algebras instead of simply using all subsets of Ω?"
options:
  - "Computational efficiency — σ-algebras reduce the number of events to check"
  - "On uncountable spaces like ℝ, not all subsets can be assigned probabilities consistently with countable additivity"
  - "σ-algebras ensure that all random variables are continuous"
  - "The axiom of choice requires σ-algebras to define probability"
correct: 1
explanation: "Vitali's construction shows that on ℝ (with Lebesgue measure), assigning a translation-invariant, countably additive measure to every subset leads to contradiction. The σ-algebra restricts attention to 'well-behaved' (measurable) sets. Option A confuses mathematical necessity with computation; option C is false (discrete RVs are measurable); option D reverses the relationship (the axiom of choice is what creates non-measurable sets)."
:::

:::quiz
question: "The Dominated Convergence Theorem requires a dominating function Y with E[Y] < ∞. Which of the following is the primary ML consequence of this theorem?"
options:
  - "It guarantees that neural networks converge during training"
  - "It justifies interchanging differentiation and expectation in score function gradient estimators"
  - "It proves that the sample mean converges to the population mean"
  - "It establishes that all bounded sequences of random variables converge"
correct: 1
explanation: "The DCT justifies swapping limits (including derivatives) with expectations — this is exactly what happens in the REINFORCE estimator when we write ∇θ E[f(X)] = E[f(X)∇θ log pθ(X)]. Option A conflates convergence of RVs with optimization convergence; option C is the law of large numbers (different theorem); option D is false (boundedness does not imply convergence)."
:::

:::quiz
question: "Conditional expectation E[X|G] is best understood geometrically as:"
options:
  - "The maximum likelihood estimate of X given G"
  - "The orthogonal projection of X onto the space of G-measurable L² functions"
  - "The conditional probability P(X|G) multiplied by X"
  - "The Bayesian posterior mean, which requires a prior distribution"
correct: 1
explanation: "E[X|G] minimizes E[(X - Z)²] over all G-measurable Z — this is precisely the definition of orthogonal projection in the Hilbert space L². Option A is an optimization procedure, not a definition of conditional expectation; option C is not a meaningful expression; option D describes one application but not the general definition (conditional expectation does not require Bayesian framing)."
:::
