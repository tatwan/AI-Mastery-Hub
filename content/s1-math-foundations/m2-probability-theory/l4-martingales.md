---
title: "Martingales & Stopping Times"
estimatedMinutes: 30
tags: ["martingales", "stopping-times", "optional-stopping", "sequential-decision-making", "RL-theory"]
prerequisites: ["l1-measure-theory"]
---

# Martingales & Stopping Times

## The Martingale Concept

> **Intuition:** A martingale is a fair game: given everything you know now, your best prediction of the future value is the present value — $\mathbb{E}[M_{t+1} \mid \mathcal{F}_t] = M_t$. There is no systematic drift upward or downward. The process can fluctuate wildly, but on average it goes nowhere. This is the mathematical definition of "no free lunch" in sequential settings.

A **martingale** is a stochastic process that models a fair game. Formally, a sequence of random variables $(M_t)_{t \geq 0}$ adapted to a filtration $(\mathcal{F}_t)_{t \geq 0}$ is a martingale if:

1. $M_t$ is $\mathcal{F}_t$-measurable for all $t$ (the value at time $t$ depends only on information available at $t$)
2. $\mathbb{E}[|M_t|] < \infty$ for all $t$ (integrability)
3. $\mathbb{E}[M_{t+1} | \mathcal{F}_t] = M_t$ for all $t$ (the fair game property)

Condition 3 is the defining property: given everything you know up to time $t$, your best prediction of the future value $M_{t+1}$ is the current value $M_t$. There is no drift — no tendency to increase or decrease.

By the tower property of conditional expectation, the martingale condition implies $\mathbb{E}[M_t] = \mathbb{E}[M_0]$ for all $t$. The expected value is constant through time, though the realized path may fluctuate wildly.

## Sub- and Supermartingales

Relaxing the equality in condition 3 yields two important generalizations:

- **Submartingale**: $\mathbb{E}[M_{t+1} | \mathcal{F}_t] \geq M_t$ — a favorable game, trending upward on average. Convex functions of martingales are submartingales (Jensen's inequality for conditional expectation).

- **Supermartingale**: $\mathbb{E}[M_{t+1} | \mathcal{F}_t] \leq M_t$ — an unfavorable game, trending downward on average.

In reinforcement learning, cumulative regret under an algorithm that "learns" is often a supermartingale: as the agent improves, the expected additional regret per step decreases. Conversely, the squared error of a consistent estimator, viewed as a process in $n$, is a non-negative supermartingale that converges to zero.

> **Key insight:** The martingale property is about the conditional expectation given the past — not about independence. Martingale increments $M_{t+1} - M_t$ are uncorrelated (since $\mathbb{E}[M_{t+1} - M_t | \mathcal{F}_t] = 0$), but they can be highly dependent. This makes martingale theory applicable to settings far beyond i.i.d. sequences.

## Canonical Examples

**Random walk.** Let $Z_1, Z_2, \ldots$ be i.i.d. with $\mathbb{E}[Z_i] = 0$, and set $M_t = \sum_{i=1}^t Z_i$. Then $(M_t)$ is a martingale with respect to $\mathcal{F}_t = \sigma(Z_1, \ldots, Z_t)$, since $\mathbb{E}[M_{t+1} | \mathcal{F}_t] = M_t + \mathbb{E}[Z_{t+1}] = M_t$.

**Likelihood ratio process.** Given two measures $P$ and $Q$, the process $Z_t = \prod_{s=1}^t \frac{p(X_s)}{q(X_s)}$ is a martingale under $Q$. This is because $\mathbb{E}_Q[Z_{t+1} | \mathcal{F}_t] = Z_t \cdot \mathbb{E}_Q\left[\frac{p(X_{t+1})}{q(X_{t+1})}\right] = Z_t \cdot 1 = Z_t$. This martingale is the foundation of sequential hypothesis testing (Wald's sequential probability ratio test) and appears in the importance weights of off-policy RL.

> **Remember:** Given any integrable random variable $Y$, the sequence $M_t = \mathbb{E}[Y \mid \mathcal{F}_t]$ is always a martingale — the tower property guarantees this. This is Doob's martingale construction. In Bayesian learning, posterior means are Doob martingales in the number of observations. In attention mechanisms, $M_t$ approximates the conditional expectation of the target given the context seen so far.

**Doob's martingale.** For any integrable random variable $Y$ and filtration $(\mathcal{F}_t)$, the process $M_t = \mathbb{E}[Y | \mathcal{F}_t]$ is a martingale. This is the "best prediction" process: as more information is revealed, the conditional expectation updates but remains a martingale. In Bayesian learning, the posterior mean of a parameter is a Doob martingale in the number of observations.

**Exponential martingale.** If $Z_i$ are i.i.d. with MGF $M(\lambda) = \mathbb{E}[e^{\lambda Z_i}]$, then $\exp\left(\lambda \sum_{i=1}^t Z_i - t \log M(\lambda)\right)$ is a martingale for any $\lambda$. This construction is the engine behind the Chernoff bound and Azuma-Hoeffding inequality.

## Martingale Convergence Theorem

> **Intuition:** Bounded martingales always converge — they cannot oscillate forever without violating the bounded-expectation condition. The key is that the process cannot "explore" indefinitely if its values are constrained. This is why modeling the squared distance to an optimum as a non-negative supermartingale automatically implies convergence of the algorithm, without needing to track the detailed trajectory.

**Theorem.** If $(M_t)$ is a martingale (or non-negative supermartingale) with $\sup_t \mathbb{E}[|M_t|] < \infty$, then $M_t$ converges almost surely to a finite limit $M_\infty$.

This is a remarkable result: bounded martingales must converge. The proof uses the **upcrossing inequality** — counting how many times the process crosses from below $a$ to above $b$ — and showing that the expected number of upcrossings is finite.

The convergence theorem explains why many sequential learning algorithms converge. If you can model the error (or a related quantity) as a non-negative supermartingale, convergence is automatic. For instance, in stochastic approximation (which includes SGD as a special case), the squared distance to the optimum, under appropriate step-size conditions, forms a non-negative supermartingale.

One subtlety: convergence of $M_t$ does not mean $\mathbb{E}[M_t] \to \mathbb{E}[M_\infty]$. That requires **uniform integrability** — a stronger condition ensuring no probability mass escapes to infinity. A uniformly integrable martingale converges in $L^1$ and satisfies $M_t = \mathbb{E}[M_\infty | \mathcal{F}_t]$, making it a Doob martingale for its own limit.

## Stopping Times

A random variable $\tau: \Omega \to \{0, 1, 2, \ldots\} \cup \{\infty\}$ is a **stopping time** with respect to $(\mathcal{F}_t)$ if:

$$\{\tau \leq t\} \in \mathcal{F}_t \quad \text{for all } t$$

The decision to stop at or before time $t$ must depend only on information available at time $t$. You can decide to stop now based on everything you have seen, but you cannot peek at the future.

Examples of stopping times: the first time a random walk hits zero; the first time a confidence interval is sufficiently narrow; the first time a loss drops below a threshold. Non-examples: "the last time the process exceeds 5" (requires knowing the entire future path) or "stop one step before the maximum" (requires future knowledge).

## Optional Stopping Theorem

> **Refresher:** You cannot beat a fair game by cleverly choosing when to stop — the Optional Stopping Theorem says $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$ as long as the stopping rule satisfies certain integrability conditions. This is the gambler's ruin in formal clothing: no stopping strategy generates a positive expected gain from a martingale. The conditions matter — unbounded increments or infinite expected stopping times can break the result.

**Theorem (Doob's Optional Stopping).** Let $(M_t)$ be a martingale and $\tau$ a stopping time. If any of the following conditions hold:

1. $\tau$ is bounded (i.e., $\tau \leq N$ a.s. for some $N$), or
2. $\mathbb{E}[\tau] < \infty$ and the increments $|M_{t+1} - M_t|$ are bounded a.s., or
3. $(M_{t \wedge \tau})$ is uniformly integrable

then $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$.

The theorem states that you cannot beat a fair game by choosing when to stop, provided you cannot wait forever under "just right" conditions. This is the mathematical formalization of the gambler's ruin: no betting strategy with a stopping rule can generate positive expected profit from a fair game.

> **Key insight:** The optional stopping theorem has a sharp converse in practice. If $\mathbb{E}[M_\tau] \neq \mathbb{E}[M_0]$, one of the conditions has been violated — usually because the stopping time has heavy tails or the martingale has unbounded increments. The classic "double your bet until you win" strategy fails because it requires unbounded wealth and unbounded time.

## Applications in Reinforcement Learning

The connection between martingales and RL runs deep. Consider an MDP with policy $\pi$, value function $V^\pi(s) = \mathbb{E}_\pi[\sum_{k=0}^\infty \gamma^k r_{t+k} | s_t = s]$, and the process:

$$M_t = \sum_{k=0}^{t-1} \gamma^k r_k + \gamma^t V^\pi(s_t)$$

This is a martingale under $\pi$: the sum of discounted rewards collected plus the discounted value of the current state. The martingale property holds because:

$$\mathbb{E}_\pi[M_{t+1} | \mathcal{F}_t] = \sum_{k=0}^{t-1} \gamma^k r_k + \gamma^t \mathbb{E}_\pi[r_t + \gamma V^\pi(s_{t+1}) | s_t] = M_t$$

where the last equality uses the Bellman equation $V^\pi(s_t) = \mathbb{E}_\pi[r_t + \gamma V^\pi(s_{t+1}) | s_t]$.

> **Intuition:** TD errors in reinforcement learning are martingale difference sequences — their conditional expectation given the current history is zero. This zero-mean property is exactly why temporal difference learning converges: the updates are noisy but unbiased, and the noise averages out over time. The martingale structure is what distinguishes convergent TD learning from divergent "residual gradient" methods that break this property.

The TD error $\delta_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$ is a **martingale difference**: $\mathbb{E}[\delta_t | \mathcal{F}_t] = 0$. This is exactly why TD learning converges — the updates are noisy but unbiased, and the noise forms a martingale difference sequence. Convergence proofs for TD($\lambda$), Q-learning, and actor-critic methods all rely on this structure.

## Azuma-Hoeffding Inequality

The Azuma-Hoeffding inequality extends Hoeffding-type concentration to martingales — sequences with dependent increments.

**Theorem (Azuma-Hoeffding).** Let $(M_t)$ be a martingale with bounded increments: $|M_{t+1} - M_t| \leq c_t$ a.s. Then:

$$P(M_n - M_0 \geq t) \leq \exp\left(-\frac{t^2}{2\sum_{s=0}^{n-1} c_s^2}\right)$$

The bound has the same form as Hoeffding's inequality but applies to martingales rather than sums of independent variables. Since sums of independent centered variables are martingales (with independent increments), Azuma-Hoeffding generalizes Hoeffding.

This inequality is indispensable for analyzing online learning algorithms, where the loss sequence is not i.i.d. but adaptive. In online convex optimization, the regret $R_T = \sum_{t=1}^T [\ell_t(w_t) - \ell_t(w^*)]$ can be decomposed into a predictable (bias) term and a martingale (variance) term. Azuma-Hoeffding bounds the latter, giving high-probability regret bounds even when the adversary is adaptive.

## Online Learning and Martingale Differences

In the online learning setting, at each round $t$ the learner chooses $w_t$ based on $\mathcal{F}_{t-1}$, then observes loss $\ell_t(w_t)$. Define:

$$D_t = \ell_t(w_t) - \mathbb{E}[\ell_t(w_t) | \mathcal{F}_{t-1}]$$

This is a martingale difference sequence: $\mathbb{E}[D_t | \mathcal{F}_{t-1}] = 0$ by construction. The cumulative sum $\sum_{t=1}^T D_t$ is a martingale, and Azuma-Hoeffding (or Freedman's inequality, which also uses variance) gives:

$$P\left(\sum_{t=1}^T D_t \geq \epsilon\right) \leq \exp\left(-\frac{\epsilon^2}{2\sum c_t^2}\right)$$

This separates the regret analysis into a deterministic component (handled by the algorithm's guarantee) and a stochastic component (handled by martingale concentration). The framework applies uniformly to Follow-the-Leader, Mirror Descent, and Thompson Sampling analyses.

> **Key insight:** Martingale concentration inequalities are the bridge between worst-case online learning (where regret bounds are deterministic) and stochastic settings (where high-probability bounds are needed). They let us analyze adaptive, sequential processes without assuming independence.

## Python: Simulating Martingales and the Optional Stopping Theorem

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_paths = 20_000
max_steps = 500

# Simple random walk: M_t = sum of Rademacher +-1
steps = np.random.choice([-1, 1], size=(n_paths, max_steps))
walks = np.cumsum(steps, axis=1)

# Stopping time: first time |M_t| >= 10
barrier = 10
stopped_values = []
for i in range(n_paths):
    hit = np.where(np.abs(walks[i]) >= barrier)[0]
    if len(hit) > 0:
        stopped_values.append(walks[i, hit[0]])
    # Paths not hitting barrier within max_steps are excluded

# Optional stopping: E[M_tau] should equal E[M_0] = 0
# (for bounded stopping times / bounded barriers this holds)
print(f"Paths reaching barrier: {len(stopped_values)}/{n_paths}")
print(f"E[M_0] = 0")
print(f"E[M_tau] = {np.mean(stopped_values):.4f}  (should be ≈ 0)")
print(f"Std error: {np.std(stopped_values)/np.sqrt(len(stopped_values)):.4f}")

# Verify Azuma-Hoeffding: P(M_n >= t) <= exp(-t^2 / 2n)
n_fixed = 200
final_values = walks[:, n_fixed - 1]
t_vals = np.linspace(1, 25, 50)
empirical_tail = [(final_values >= t).mean() for t in t_vals]
azuma_bound = [np.exp(-t**2 / (2 * n_fixed)) for t in t_vals]

plt.figure(figsize=(8, 5))
plt.semilogy(t_vals, empirical_tail, 'o', ms=3, label='Empirical tail')
plt.semilogy(t_vals, azuma_bound, '-', label='Azuma-Hoeffding bound')
plt.xlabel('Threshold $t$')
plt.ylabel('$P(M_n \\geq t)$')
plt.title(f'Azuma-Hoeffding bound for random walk, n={n_fixed}')
plt.legend()
plt.tight_layout()
plt.show()
```

The simulation confirms two results: the optional stopping theorem (the mean at the stopping time is approximately zero, matching the initial value), and the Azuma-Hoeffding bound (the empirical tail is always below the theoretical bound).

:::quiz
question: "The likelihood ratio process Z_t = ∏ₛ₌₁ᵗ p(Xₛ)/q(Xₛ) is a martingale under which measure?"
options:
  - "Under P, the numerator measure"
  - "Under Q, the denominator measure"
  - "Under both P and Q simultaneously"
  - "Under neither — it is only a supermartingale"
correct: 1
explanation: "E_Q[p(X_{t+1})/q(X_{t+1})] = ∫ (p(x)/q(x)) q(x) dx = ∫ p(x) dx = 1, so Z_{t+1}/Z_t has conditional expectation 1 under Q, making Z_t a Q-martingale. Under P, the ratio q/p would form the martingale. The likelihood ratio is a martingale under the measure in the denominator."
:::

:::quiz
question: "In TD learning, the TD error δ_t = r_t + γV^π(s_{t+1}) - V^π(s_t) has E[δ_t | F_t] = 0 under policy π. This means:"
options:
  - "TD learning converges in one step"
  - "The TD errors form a martingale difference sequence, so their cumulative sum is a martingale"
  - "The value function V^π must be linear in the state features"
  - "The rewards r_t must be independent of the states"
correct: 1
explanation: "E[δ_t | F_t] = 0 is exactly the definition of a martingale difference. The cumulative sum of TD errors is therefore a martingale, which is the key structural property used in convergence proofs for TD methods. Option A is nonsensical; option C confuses function approximation with the MDS property; option D is false and unnecessary."
:::

:::quiz
question: "The optional stopping theorem states E[M_τ] = E[M_0] for a martingale M_t and stopping time τ. This fails for the 'doubling strategy' in a fair coin game because:"
options:
  - "The coin flips are not independent"
  - "The stopping time τ is not measurable with respect to the filtration"
  - "The martingale increments are unbounded (bets grow as 2^t), violating the conditions of the OST"
  - "The game is not actually fair — the house has an edge"
correct: 2
explanation: "The stopping time itself has *finite* expectation ($\\mathbb{E}[\\tau] = 2$ for the geometric distribution). The doubling strategy fails because the **martingale increments are unbounded** — the bet at step $t$ is $2^t$, growing exponentially. This violates the bounded increments condition required by the Optional Stopping Theorem. The stopped process $M_{t \\wedge \\tau}$ is not uniformly integrable, so none of the sufficient conditions for $\\mathbb{E}[M_\\tau] = \\mathbb{E}[M_0]$ are satisfied."
:::
