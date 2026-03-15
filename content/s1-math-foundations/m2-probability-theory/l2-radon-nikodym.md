---
title: "Radon-Nikodym Theorem & Change of Measure"
estimatedMinutes: 25
tags: ["Radon-Nikodym", "change-of-measure", "likelihood-ratio", "KL-divergence", "importance-sampling"]
prerequisites: ["l1-measure-theory"]
---

# Radon-Nikodym Theorem & Change of Measure

## Absolute Continuity of Measures

> **Intuition:** $P$ is absolutely continuous with respect to $Q$ ($P \ll Q$) means: if $Q$ says something is impossible, $P$ agrees. Wherever $Q$ assigns zero probability, $P$ must too. This is the condition that makes the density $dP/dQ$ well-defined — you cannot have a reweighting factor at a point that $Q$ never visits.

Given two measures $P$ and $Q$ on the same measurable space $(\Omega, \mathcal{F})$, we say $P$ is **absolutely continuous** with respect to $Q$, written $P \ll Q$, if:

$$Q(A) = 0 \implies P(A) = 0 \quad \text{for all } A \in \mathcal{F}$$

In words: every $Q$-null set is also a $P$-null set. If $P$ assigns positive probability to some event, then $Q$ must also assign it positive probability. This rules out the degenerate situation where we try to estimate properties of $P$ by sampling from $Q$, but $Q$ places zero mass on regions where $P$ concentrates.

If both $P \ll Q$ and $Q \ll P$, the measures are **equivalent** — they agree on which events are possible, though they may disagree on the probabilities. Equivalent measures arise naturally in change-of-measure arguments: if two measures are equivalent, any expectation under one can be rewritten as an expectation under the other.

> **Key insight:** Absolute continuity is the minimal condition for importance sampling to work. If $P$ is not absolutely continuous with respect to the proposal $Q$, there exist events with positive probability under $P$ that $Q$ never generates — making unbiased estimation impossible.

## The Radon-Nikodym Theorem

> **Remember:** The Radon-Nikodym derivative $dP/dQ$ is the likelihood ratio — it tells you how to reweight samples drawn from $Q$ so they behave as if drawn from $P$. When densities exist, $dP/dQ = p/q$. This single object is the common thread behind importance sampling, off-policy RL, variational inference, and KL divergence.

The Radon-Nikodym theorem is one of the central results of measure theory, and it underpins nearly every density-based computation in ML.

**Theorem (Radon-Nikodym).** If $P \ll Q$ and both are σ-finite measures on $(\Omega, \mathcal{F})$, there exists a unique (up to $Q$-a.s. equality) non-negative measurable function $\frac{dP}{dQ}: \Omega \to [0, \infty)$ such that:

$$P(A) = \int_A \frac{dP}{dQ}(\omega) \, dQ(\omega) \quad \text{for all } A \in \mathcal{F}$$

The function $\frac{dP}{dQ}$ is called the **Radon-Nikodym derivative** (or density) of $P$ with respect to $Q$.

In plain terms: any absolutely continuous measure can be expressed as a "reweighting" of the reference measure. The Radon-Nikodym derivative tells you the local reweighting factor at each point $\omega$.

When both $P$ and $Q$ admit densities $p$ and $q$ with respect to Lebesgue measure, the Radon-Nikodym derivative takes the familiar form:

$$\frac{dP}{dQ}(\omega) = \frac{p(\omega)}{q(\omega)}$$

This is the **likelihood ratio** — one of the most frequently occurring quantities in statistics and ML.

## The Change-of-Measure Formula

The Radon-Nikodym derivative lets us convert expectations between measures. For any measurable function $f$:

$$\mathbb{E}_P[f(X)] = \int f(\omega) \, dP(\omega) = \int f(\omega) \frac{dP}{dQ}(\omega) \, dQ(\omega) = \mathbb{E}_Q\left[f(X) \frac{dP}{dQ}(X)\right]$$

This identity is the foundation of importance sampling, off-policy evaluation in RL, and the change-of-measure arguments in mathematical finance. It says: to compute an expectation under $P$, you can instead sample from $Q$ and reweight by the likelihood ratio.

## KL Divergence: The Measure-Theoretic Definition

The Kullback-Leibler divergence between $P$ and $Q$ (assuming $P \ll Q$) has a clean expression in terms of the Radon-Nikodym derivative:

$$D_{\text{KL}}(P \| Q) = \int \log \frac{dP}{dQ} \, dP = \mathbb{E}_P\left[\log \frac{dP}{dQ}\right]$$

This is the **general** definition of KL divergence, valid for any pair of measures with $P \ll Q$ — no assumption of densities with respect to Lebesgue measure is needed. When densities exist, it reduces to the familiar $\int p(x) \log \frac{p(x)}{q(x)} dx$.

The KL divergence measures the expected log-likelihood ratio under $P$. It quantifies how much additional information (in nats) is required, on average, to identify samples from $P$ versus $Q$. It is always non-negative (Gibbs' inequality) and equals zero if and only if $P = Q$ a.s.

> **Key insight:** The measure-theoretic KL definition makes it clear that KL divergence is fundamentally about comparing two measures via their Radon-Nikodym derivative. This perspective unifies the discrete and continuous formulas and extends naturally to singular measures, mixture models, and infinite-dimensional spaces.

## Girsanov's Theorem and Change of Drift

> **Intuition:** Girsanov's theorem says you can change the drift of a Brownian motion by reweighting path probabilities. Under the new measure $Q$, the process with drift looks like pure noise. This is how risk-neutral pricing works in finance (remove the drift, price by discounting), and it is why the score function $\nabla_x \log p_t$ enters the reverse SDE in diffusion models — it is a drift correction under a change of measure.

In continuous-time settings, the most powerful change-of-measure result is **Girsanov's theorem**. Informally, it states:

If $W_t$ is a Brownian motion under measure $P$, and we define a new measure $Q$ via the Radon-Nikodym derivative:

$$\frac{dQ}{dP}\bigg|_{\mathcal{F}_T} = \exp\left(-\int_0^T \theta_t \, dW_t - \frac{1}{2}\int_0^T \theta_t^2 \, dt\right)$$

then under $Q$, the process $\tilde{W}_t = W_t + \int_0^t \theta_s \, ds$ is a Brownian motion.

**Novikov's condition** provides a practical sufficient guarantee that $Z_t = \exp\!\left(\int_0^t \theta_s \, dW_s - \frac{1}{2}\int_0^t \theta_s^2 \, ds\right)$ is a true martingale (not merely a local martingale): if $\mathbb{E}\!\left[\exp\!\left(\frac{1}{2}\int_0^T \theta_s^2 \, ds\right)\right] < \infty$, then $Z_T$ defines a valid probability measure change. Without this condition, Girsanov's exponential can fail to integrate to 1.

The practical implication: changing the measure amounts to changing the drift of a stochastic process while preserving the diffusion structure. This is exactly the mechanism behind the reverse-time SDE in diffusion models — the score function $\nabla_x \log p_t(x)$ enters as a drift correction under a change of measure.

In reinforcement learning, the ratio $\frac{d P_\pi}{d P_\mu}$ between the trajectory distributions under a target policy $\pi$ and a behavior policy $\mu$ is a product of per-step importance weights — this is the foundation of off-policy methods like V-trace and retrace.

## Importance Sampling

> **Refresher:** Importance sampling is a Monte Carlo trick: instead of drawing samples from $P$ directly (which may be expensive or impossible), draw samples from a proposal $Q$ and multiply each evaluation $f(x_i)$ by the likelihood ratio $dP/dQ(x_i) = p(x_i)/q(x_i)$. The estimator is unbiased by the change-of-measure formula, but variance can explode if $P$ and $Q$ are poorly matched — high weights on rare samples dominate the average.

Importance sampling (IS) operationalizes the change-of-measure formula for Monte Carlo estimation. To estimate $\mathbb{E}_P[f(X)]$ when sampling from $P$ is expensive or infeasible, we draw samples $x_1, \ldots, x_n \sim Q$ and compute:

$$\hat{\mu}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n f(x_i) \frac{p(x_i)}{q(x_i)}$$

This estimator is **unbiased**: $\mathbb{E}_Q[\hat{\mu}_{\text{IS}}] = \mathbb{E}_P[f(X)]$. However, its variance depends critically on the choice of proposal $Q$:

$$\text{Var}_Q[\hat{\mu}_{\text{IS}}] = \frac{1}{n}\left(\mathbb{E}_Q\left[f(X)^2 \frac{p(X)^2}{q(X)^2}\right] - \mathbb{E}_P[f(X)]^2\right)$$

When $p/q$ varies widely (i.e., $P$ and $Q$ are poorly matched), a few samples receive enormous weights, inflating variance. The variance is related to the $\chi^2$-divergence: $\text{Var}_Q[p/q] = \chi^2(P \| Q)$. This is why self-normalized importance sampling (dividing by the sum of weights) often works better in practice — it trades a small bias for substantially lower variance. The self-normalized IS estimator $\hat{\mu}_{SN} = \sum_i w(X_i) f(X_i) / \sum_i w(X_i)$ is consistent but biased at finite $n$ — the bias is $O(1/n)$. The unnormalized estimator $\frac{1}{n}\sum_i w(X_i)f(X_i)$ is unbiased but has higher variance.

The **optimal proposal** that minimizes IS variance for estimating $\mathbb{E}_P[f(X)]$ is $q^*(x) \propto |f(x)| p(x)$. This concentrates samples where the integrand is large, but it requires knowing the very integral we are trying to compute — so in practice, we settle for proposals that approximate this distribution.

## Connection to Variational Inference

The ELBO (Evidence Lower Bound) derivation is a direct application of the Radon-Nikodym derivative and Jensen's inequality. For a latent variable model with observed $x$ and latent $z$:

$$\log p(x) = \log \int p(x, z) \, dz = \log \int \frac{p(x, z)}{q(z|x)} q(z|x) \, dz = \log \mathbb{E}_{q(z|x)}\left[\frac{p(x,z)}{q(z|x)}\right]$$

Applying Jensen's inequality ($\log$ is concave):

$$\log p(x) \geq \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] = \text{ELBO}(q)$$

The gap between $\log p(x)$ and the ELBO is exactly $D_{\text{KL}}(q(z|x) \| p(z|x))$. Maximizing the ELBO over $q$ simultaneously tightens this bound and makes $q$ a better approximation to the true posterior. The ratio $\frac{p(x,z)}{q(z|x)}$ inside the logarithm is the Radon-Nikodym derivative of the joint distribution with respect to the variational approximation — the same object that appears in importance sampling.

### Lebesgue Decomposition Theorem

> **Intuition:** Any measure splits uniquely into an absolutely continuous part (which has a density with respect to $Q$) and a singular part (which lives on a set that $Q$ ignores entirely). The GAN discriminator implicitly learns this split: early in training, the generator's distribution is singular with respect to the data distribution — they live on different manifolds — making density-ratio estimation ill-posed and motivating Wasserstein distance.

A companion result to Radon-Nikodym: any measure $P$ can be uniquely decomposed as $P = P_{ac} + P_s$ where $P_{ac} \ll Q$ (absolutely continuous part, admitting an RN derivative) and $P_s \perp Q$ (singular part, living on a $Q$-null set). This decomposition is the mathematical basis for understanding GAN training: the generator's distribution $P_G$ is initially singular with respect to the data distribution $P_{data}$ (they live on different manifolds), making the JS divergence undefined as a density ratio. This singularity problem motivates Wasserstein GANs (Semester 5), which use a distance that remains well-defined even for mutually singular measures.

> **Key insight:** Variational inference, importance sampling, and off-policy RL all share the same mathematical skeleton: reweight samples from one distribution to compute expectations under another, using the Radon-Nikodym derivative as the reweighting factor.

## ML Connections

The Radon-Nikodym theorem is the mathematical engine behind likelihood ratios, importance weighting, and the connection between different probability distributions that permeates modern ML.

- **Importance Sampling in RL and Offline Learning:** Off-policy learning estimates $\mathbb{E}_\mu[f]$ from data collected under a different policy $\nu$: $\mathbb{E}_\mu[f(x)] = \mathbb{E}_\nu[f(x) \frac{d\mu}{d\nu}(x)]$. The likelihood ratio $\frac{d\mu}{d\nu}$ is exactly the Radon-Nikodym derivative. PPO's clipped surrogate objective bounds this ratio to prevent large policy updates.
- **Variational Autoencoders:** The ELBO derivation uses $\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi \| p)$. The KL term is $\mathbb{E}_{q_\phi}[\log \frac{dq_\phi}{dp}]$ — the Radon-Nikodym derivative of $q_\phi$ w.r.t. $p$. When $q_\phi = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ and $p = \mathcal{N}(0,1)$, the RN derivative is Gaussian ratio, giving the closed-form KL.
- **Diffusion Model Reverse Process:** Girsanov's theorem underlies diffusion model training: the score function $\nabla_x \log p_t(x)$ is the key quantity needed to reverse the forward SDE. The score is the gradient of the log Radon-Nikodym derivative between $p_t$ and the reference measure. Score matching directly estimates this.
- **Reward Modeling via DPO:** Direct Preference Optimization (DPO) implicitly trains a reward model $r(x,y) \propto \log \frac{d\pi_\theta}{d\pi_\text{ref}}(y|x)$ — the log Radon-Nikodym derivative of the policy w.r.t. a reference policy. This connects RLHF to density ratio estimation.
- **GAN Discriminator:** An optimal GAN discriminator $D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}$ is a function of the density ratio $\frac{dp_\text{data}}{dp_G}$. Training the discriminator is implicitly estimating the Radon-Nikodym derivative from samples.

> **Key insight:** Likelihood ratios are everywhere in ML: importance weights in RL, KL terms in VAEs, score functions in diffusion models, DPO reward models, GAN discriminators. They are all Radon-Nikodym derivatives. Understanding this unifies what appear to be unrelated techniques into one framework: density ratio estimation.

## Python: Importance Sampling for Tail Probabilities

Estimating tail probabilities like $P(X > 4)$ under $X \sim \mathcal{N}(0,1)$ is notoriously difficult with naive Monte Carlo because the event is rare. Importance sampling with a shifted proposal dramatically reduces variance.

```python
import numpy as np

np.random.seed(42)
n = 100_000
threshold = 4.0

# Target: P(X > 4) where X ~ N(0,1)
# Naive MC: sample from N(0,1)
x_naive = np.random.randn(n)
naive_est = (x_naive > threshold).mean()

# IS with shifted proposal Q = N(4, 1)
mu_q = 4.0
x_is = np.random.randn(n) + mu_q
log_w = -0.5 * x_is**2 - (-0.5 * (x_is - mu_q)**2)  # log(p/q)
weights = np.exp(log_w)
is_est = (weights * (x_is > threshold)).mean()

# Self-normalized IS
sn_weights = weights / weights.sum()
sn_est = (sn_weights * (x_is > threshold)).sum()

from scipy.stats import norm
true_val = 1 - norm.cdf(threshold)

print(f"True P(X>4):        {true_val:.6e}")
print(f"Naive MC estimate:  {naive_est:.6e}")
print(f"IS estimate:        {is_est:.6e}")
print(f"Self-norm IS est:   {sn_est:.6e}")
```

The shifted proposal $Q = \mathcal{N}(4, 1)$ centers samples in the tail region, ensuring many samples contribute to the estimate. The naive estimator will often return exactly zero with $10^5$ samples (since $P(X > 4) \approx 3.17 \times 10^{-5}$), while the IS estimator converges reliably.

:::quiz
question: "The Radon-Nikodym derivative dP/dQ exists when:"
options:
  - "P and Q have the same support on ℝ"
  - "P is absolutely continuous with respect to Q (P ≪ Q) and both are σ-finite"
  - "P and Q are both Gaussian distributions"
  - "The KL divergence D_KL(P‖Q) is finite"
correct: 1
explanation: "The Radon-Nikodym theorem requires P ≪ Q and σ-finiteness. Option A is informal and neither necessary nor sufficient; option C is a special case; option D is a consequence (finite KL requires P ≪ Q) but not the condition for existence of the derivative."
:::

:::quiz
question: "In importance sampling, what happens when the proposal Q assigns near-zero density to a region where P has substantial mass?"
options:
  - "The IS estimator becomes biased"
  - "The IS estimator remains unbiased but has extremely high variance due to large importance weights"
  - "The IS estimator underestimates the true expectation on average"
  - "Self-normalized IS automatically corrects for this mismatch"
correct: 1
explanation: "IS is always unbiased (assuming P ≪ Q), but when q(x) is small where p(x) is large, the ratio p(x)/q(x) becomes huge for rare samples, causing the variance to explode. This is related to high χ²(P‖Q). Option A is incorrect (unbiasedness is guaranteed by the change-of-measure formula); option C confuses bias with variance; option D is false (self-normalization reduces variance but cannot fix fundamental support mismatch)."
:::

:::quiz
question: "The ELBO gap — the difference between log p(x) and the ELBO — equals:"
options:
  - "The reconstruction error of the decoder"
  - "The entropy of the variational distribution q(z|x)"
  - "D_KL(q(z|x) ‖ p(z|x)), the KL divergence from the variational posterior to the true posterior"
  - "D_KL(p(z|x) ‖ q(z|x)), the KL divergence from the true posterior to the variational posterior"
correct: 2
explanation: "By direct derivation: log p(x) = ELBO + D_KL(q(z|x) ‖ p(z|x)). The gap is the forward KL from q to the true posterior p(z|x). Option D reverses the KL arguments (this matters — KL is asymmetric). Options A and B are components of the ELBO, not the gap itself."
:::
