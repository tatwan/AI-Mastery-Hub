---
title: "Concentration Inequalities"
estimatedMinutes: 30
tags: ["concentration-inequalities", "Hoeffding", "Bernstein", "McDiarmid", "sub-Gaussian", "generalization"]
prerequisites: ["l1-measure-theory"]
---

# Concentration Inequalities

## Why Concentration?

Machine learning is fundamentally a science of finite samples. We observe $n$ data points, compute an empirical loss, and hope it reflects the true (population) loss. Concentration inequalities make this hope rigorous — they quantify, as a function of $n$, how tightly an empirical quantity clusters around its expectation.

Without concentration inequalities, we cannot state generalization bounds, we cannot prove that stochastic gradient descent converges at a particular rate, and we cannot give finite-sample guarantees for bandit algorithms. These inequalities form the quantitative backbone of statistical learning theory.

## From Weak to Strong: A Hierarchy of Tail Bounds

### Markov's Inequality

The simplest tail bound requires only that $X \geq 0$ with finite mean:

$$P(X \geq t) \leq \frac{\mathbb{E}[X]}{t}$$

This follows immediately from $\mathbb{E}[X] \geq t \cdot P(X \geq t)$. Markov's inequality is sharp — for any $t$ and $\mu$, there exists a distribution achieving equality (a two-point distribution). But it only gives an $O(1/t)$ tail decay, which is far too loose for most applications.

### Chebyshev's Inequality

Applying Markov's inequality to $(X - \mu)^2$:

$$P(|X - \mu| \geq t) \leq \frac{\text{Var}(X)}{t^2}$$

This uses second-moment information and gives $O(1/t^2)$ decay. Better, but still polynomial — insufficient for the exponential concentration we observe empirically for sums of independent random variables.

### The Chernoff Bound

The key idea: apply Markov's inequality to the exponential $e^{\lambda X}$ and optimize over $\lambda > 0$:

$$P(X \geq t) = P(e^{\lambda X} \geq e^{\lambda t}) \leq \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda t}} = e^{-\lambda t} M_X(\lambda)$$

where $M_X(\lambda) = \mathbb{E}[e^{\lambda X}]$ is the moment generating function. Minimizing over $\lambda > 0$:

$$P(X \geq t) \leq \inf_{\lambda > 0} e^{-\lambda t} M_X(\lambda)$$

This is called the **Chernoff bound** or **exponential Markov inequality**. The optimization over $\lambda$ extracts the tightest bound from the full distribution of $X$, yielding exponential tail decay for well-behaved distributions.

> **Key insight:** The Chernoff bound is the master technique. Every named concentration inequality (Hoeffding, Bernstein, sub-Gaussian bounds) is obtained by computing or bounding $M_X(\lambda)$ for specific distribution classes, then optimizing the Chernoff parameter $\lambda$.

## Sub-Gaussian Random Variables

A centered random variable $X$ (with $\mathbb{E}[X] = 0$) is **sub-Gaussian with parameter $\sigma$** if its MGF satisfies:

$$\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2 \sigma^2 / 2} \quad \text{for all } \lambda \in \mathbb{R}$$

This says the MGF is dominated by that of a $\mathcal{N}(0, \sigma^2)$ random variable — hence "sub-Gaussian." The tails of $X$ decay at least as fast as a Gaussian.

Key examples of sub-Gaussian variables:
- Gaussian $\mathcal{N}(0, \sigma^2)$: sub-Gaussian with parameter $\sigma$
- Bounded: if $X \in [a, b]$ a.s. and $\mathbb{E}[X] = 0$, then $X$ is sub-Gaussian with parameter $(b-a)/2$ (Hoeffding's lemma)
- Rademacher: $X \in \{-1, +1\}$ with equal probability, sub-Gaussian with parameter 1

The crucial algebraic property: **sums of independent sub-Gaussians are sub-Gaussian**, with $\sigma^2$ parameters adding. If $X_1, \ldots, X_n$ are independent sub-Gaussian with parameters $\sigma_1, \ldots, \sigma_n$, then $\sum X_i$ is sub-Gaussian with parameter $\sqrt{\sum \sigma_i^2}$.

Plugging the sub-Gaussian MGF bound into the Chernoff bound and optimizing over $\lambda$:

$$P\left(\sum_{i=1}^n X_i \geq t\right) \leq \exp\left(-\frac{t^2}{2\sum \sigma_i^2}\right)$$

This is the template for all the named inequalities below.

## Hoeffding's Inequality

**Theorem (Hoeffding, 1963).** Let $X_1, \ldots, X_n$ be independent random variables with $X_i \in [a_i, b_i]$ almost surely. Then:

$$P\left(\frac{1}{n}\sum_{i=1}^n (X_i - \mathbb{E}[X_i]) \geq t\right) \leq \exp\left(-\frac{2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$

For identically distributed variables with $X_i \in [a, b]$, this simplifies to:

$$P\left(\bar{X} - \mu \geq t\right) \leq \exp\left(-\frac{2nt^2}{(b-a)^2}\right)$$

The bound is exponential in $n$ — doubling the sample size squares the probability of a given deviation. The dependence on the range $(b-a)$ rather than the variance $\text{Var}(X)$ is both a strength (no variance estimation needed) and a weakness (it can be loose when variance is much smaller than the range suggests).

Hoeffding's inequality is the workhorse of learning theory. When you see a statement like "with probability at least $1 - \delta$, the empirical risk is within $\epsilon$ of the true risk," it almost certainly involves Hoeffding or a close relative.

## Bernstein's Inequality

When the variance is small relative to the range, Hoeffding's bound is pessimistic. Bernstein's inequality incorporates variance information for a tighter result.

**Theorem (Bernstein).** Let $X_1, \ldots, X_n$ be independent with $\mathbb{E}[X_i] = 0$, $\text{Var}(X_i) \leq \sigma^2$, and $|X_i| \leq M$ a.s. Then:

$$P\left(\sum_{i=1}^n X_i \geq t\right) \leq \exp\left(-\frac{t^2 / 2}{n\sigma^2 + Mt/3}\right)$$

The bound has two regimes. When $t \ll n\sigma^2/M$ (the "normal" regime), the denominator is dominated by $n\sigma^2$ and the bound behaves like $\exp(-t^2 / 2n\sigma^2)$ — a variance-dependent Gaussian tail. When $t \gg n\sigma^2/M$ (the "Poisson" regime), the $Mt/3$ term dominates and the bound becomes $\exp(-3t/2M)$ — a linear exponential tail.

> **Key insight:** Bernstein's inequality adapts to the difficulty of the problem. In the "easy" regime (small deviations or small variance), it gives tighter bounds than Hoeffding by a factor that can be $O((b-a)^2 / \sigma^2)$. This adaptivity is why Bernstein-type bounds appear in refined analyses of boosting, random forests, and empirical process theory.

## McDiarmid's Inequality (Bounded Differences)

The previous inequalities apply to sums of independent random variables. McDiarmid's inequality extends concentration to **arbitrary functions** of independent variables, provided the function does not depend too strongly on any single coordinate.

**Theorem (McDiarmid, 1989).** Let $X_1, \ldots, X_n$ be independent, and let $f: \mathcal{X}^n \to \mathbb{R}$ satisfy the **bounded differences** condition: for each $i$,

$$\sup_{x_1, \ldots, x_n, x_i'} |f(x_1, \ldots, x_i, \ldots, x_n) - f(x_1, \ldots, x_i', \ldots, x_n)| \leq c_i$$

Then:

$$P(f(X_1, \ldots, X_n) - \mathbb{E}[f] \geq t) \leq \exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right)$$

In words: if changing any single input changes the output by at most $c_i$, the function concentrates around its mean with sub-Gaussian tails. Note that this does not require $f$ to be a sum — it can be any function satisfying bounded differences.

McDiarmid's inequality is the cornerstone of generalization theory. The empirical risk $\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)$ is a function of $n$ i.i.d. samples. Replacing one sample changes $\hat{R}$ by at most $\frac{\sup \ell}{n}$, so $c_i = \frac{B}{n}$ where $B$ bounds the loss. McDiarmid then gives:

$$P(|\hat{R}(h) - R(h)| \geq t) \leq 2\exp\left(-2nt^2/B^2\right)$$

for any fixed hypothesis $h$.

## Application: PAC Learning and Uniform Convergence

The single-hypothesis bound above is useful but insufficient — we want guarantees that hold **simultaneously** for all hypotheses in a class $\mathcal{H}$. For finite $\mathcal{H}$, a union bound gives:

$$P\left(\sup_{h \in \mathcal{H}} |\hat{R}(h) - R(h)| \geq \epsilon\right) \leq 2|\mathcal{H}| \exp\left(-2n\epsilon^2/B^2\right)$$

Setting the right side to $\delta$ and solving for $\epsilon$:

$$\epsilon = B\sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}$$

This is the classic PAC (Probably Approximately Correct) bound: with probability at least $1-\delta$, every hypothesis in $\mathcal{H}$ has empirical risk within $\epsilon$ of its true risk. The sample complexity scales as $O(\log|\mathcal{H}| / \epsilon^2)$.

For infinite hypothesis classes, the union bound does not directly apply. Instead, we use **covering numbers** or **VC dimension** to control the effective size of $\mathcal{H}$. The Vapnik-Chervonenkis theorem shows:

$$P\left(\sup_{h \in \mathcal{H}} |\hat{R}(h) - R(h)| > \epsilon\right) \leq 4\left(\frac{2en}{d}\right)^d \exp(-n\epsilon^2/8)$$

where $d = \text{VC-dim}(\mathcal{H})$. The proof combines McDiarmid's inequality, symmetrization (replacing the population risk with a ghost sample), and covering arguments.

> **Key insight:** Concentration inequalities do not just tell us that averages converge — they tell us *how fast*, as a function of $n$. The rate $O(1/\sqrt{n})$ for bounded losses is a direct consequence of sub-Gaussian concentration. Faster rates ($O(1/n)$) are possible when the loss has low variance near the optimum (via Bernstein-type arguments), which is why "fast rates" in learning theory are intimately connected to noise conditions.

## Python: Empirical Verification of Hoeffding's Bound

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_values = np.arange(50, 2001, 50)
t = 0.1  # deviation threshold
n_trials = 10_000

# X_i ~ Uniform[0,1], so E[X_i]=0.5, range [0,1]
empirical_probs = []
hoeffding_bounds = []

for n in n_values:
    samples = np.random.uniform(0, 1, (n_trials, n))
    means = samples.mean(axis=1)
    # P(X_bar - 0.5 >= t)
    empirical_probs.append((means - 0.5 >= t).mean())
    # Hoeffding bound: exp(-2 n t^2 / (b-a)^2) with b-a=1
    hoeffding_bounds.append(np.exp(-2 * n * t**2))

plt.figure(figsize=(8, 5))
plt.semilogy(n_values, empirical_probs, 'o', ms=3, label='Empirical $P(\\bar{X}-\\mu \\geq 0.1)$')
plt.semilogy(n_values, hoeffding_bounds, '-', label='Hoeffding bound')
plt.xlabel('Sample size $n$')
plt.ylabel('Probability (log scale)')
plt.title('Hoeffding bound vs empirical tail probability')
plt.legend()
plt.tight_layout()
plt.show()
```

The plot reveals two important features: the Hoeffding bound is always an upper bound on the empirical probability (as guaranteed), and the gap between bound and reality narrows as $n$ grows. The bound captures the correct exponential decay rate but overestimates the constant — this is the price of distribution-free guarantees.

:::quiz
question: "A random variable X ∈ [-1, 1] with E[X] = 0 and Var(X) = 0.01 (very concentrated near zero). Which bound gives a tighter tail estimate?"
options:
  - "Hoeffding's inequality, because it uses the range [-1, 1] which fully characterizes the distribution"
  - "Bernstein's inequality, because it incorporates the small variance, giving a much tighter bound in the normal regime"
  - "Markov's inequality, because it only requires the first moment"
  - "Both give identical bounds when the distribution is symmetric"
correct: 1
explanation: "Bernstein's inequality uses σ² = 0.01 in the denominator (normal regime), while Hoeffding uses (b-a)² = 4 — a factor of 400 worse. The variance being much smaller than the range squared is exactly when Bernstein outperforms Hoeffding. Option A is wrong because using the full range ignores favorable variance structure; option C gives the weakest bound; option D is false."
:::

:::quiz
question: "McDiarmid's inequality applies to functions of independent variables with bounded differences. Which of the following is NOT a valid application?"
options:
  - "Bounding the deviation of empirical risk from true risk for a fixed hypothesis"
  - "Bounding the deviation of the k-nearest-neighbor classification error as a function of training data"
  - "Bounding the deviation of the maximum of correlated Gaussian random variables"
  - "Bounding the deviation of a leave-one-out cross-validation estimate"
correct: 2
explanation: "McDiarmid requires the random variables X₁,...,Xₙ to be independent. Correlated Gaussian variables violate this assumption, so McDiarmid does not apply. Options A, B, and D all involve functions of i.i.d. training samples (independent), and changing one sample changes the output by a bounded amount."
:::

:::quiz
question: "In the PAC learning bound, the sample complexity for a finite hypothesis class |H| scales as O(log|H|/ε²). The logarithmic dependence on |H| arises from:"
options:
  - "The central limit theorem applied to the empirical risk"
  - "A union bound over hypotheses combined with exponential concentration for each individual hypothesis"
  - "The VC dimension being at most log₂|H|"
  - "McDiarmid's bounded difference constant being inversely proportional to |H|"
correct: 1
explanation: "We apply a union bound: P(∃h: |R̂(h)-R(h)| > ε) ≤ |H| · P(|R̂(h)-R(h)| > ε for one h). The exponential concentration gives exp(-Cnε²) for each h, so we need |H|·exp(-Cnε²) ≤ δ, giving n = O(log(|H|/δ)/ε²). The log comes from inverting the exponential after multiplying by |H|. Option C is a fact but not the mechanism; option D is incorrect."
:::
