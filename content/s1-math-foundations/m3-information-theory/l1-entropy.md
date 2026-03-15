---
title: "Shannon Entropy & Information Content"
estimatedMinutes: 25
tags: ["information-theory", "entropy", "cross-entropy", "shannon"]
prerequisites: []
---

## Self-Information: Quantifying Surprise

The foundation of information theory rests on a deceptively simple question: how much *information* does an event carry? Shannon's answer begins with **self-information** (also called surprisal):

> **Refresher:** $\log(1/p)$ is the natural unit of surprise — an event with probability 1 gives $\log(1) = 0$ bits (no surprise), while an event with probability $1/2$ gives $\log_2(2) = 1$ bit, and an event with probability $1/1024$ gives 10 bits. The logarithm turns multiplicative probabilities into additive information: the information from two independent events adds because $\log(1/p_1 p_2) = \log(1/p_1) + \log(1/p_2)$.

$$I(x) = -\log_2 p(x)$$

This measures the information content of observing outcome $x$ in bits. An event with probability $p(x) = 1$ carries zero information — it was certain to happen. An event with $p(x) = 1/1024$ carries $\log_2(1024) = 10$ bits — highly surprising, highly informative.

Why the logarithm? Shannon proved this is the *unique* function satisfying three natural axioms:

1. **Continuity** — $I(x)$ varies smoothly with $p(x)$.
2. **Monotonicity** — less probable events carry more information.
3. **Additivity** — the information from two independent events adds: $I(x, y) = I(x) + I(y)$ when $x \perp y$.

The logarithm is the only function satisfying all three. The base determines the unit: base 2 gives bits, base $e$ gives nats (1 nat $\approx$ 1.443 bits). In ML we typically use nats because natural logarithms compose cleanly with exponential-family distributions and gradient computations.

> **Key insight:** Self-information is *axiomatic*, not merely a convenient definition. Any measure of information content satisfying continuity, monotonicity, and additivity must be a logarithm of the inverse probability.

## Shannon Entropy: Average Information

Given a discrete random variable $X$ with distribution $p$, the **Shannon entropy** is the expected self-information:

> **Intuition:** Entropy measures average surprise — rare events carry more information than common ones, and $p(x) \log(1/p(x))$ is exactly the "surprise" of the event weighted by how often it occurs. A coin flip (two equally likely outcomes) has 1 bit of entropy; a loaded coin that always lands heads has 0 bits. The formula averages these per-event surprises over the whole distribution.

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}_{X \sim p}\left[-\log p(X)\right]$$

Entropy quantifies the *average uncertainty* in $X$ before observing it, or equivalently, the average information gained upon observing it.

Key properties:

- **Non-negativity:** $H(X) \geq 0$, with equality iff $X$ is deterministic.
- **Maximum at uniform:** For $|\mathcal{X}| = n$, entropy is maximized by the uniform distribution $p(x) = 1/n$, giving $H(X) = \log n$. The uniform distribution is maximally uncertain.
- **Concavity:** $H$ is a concave function of $p$, meaning mixtures of distributions have higher entropy than the mixture of their entropies.

Shannon derived entropy from three axioms: (i) $H$ is continuous in the $p_i$, (ii) for uniform distributions $H$ increases with $n$, and (iii) decomposition — grouping outcomes doesn't change the total entropy. These uniquely determine $H$ up to a positive scalar (the choice of log base).

### The Source Coding Theorem

Shannon's first major theorem gives entropy an operational meaning: $H(X)$ is the **minimum average number of bits** required to losslessly encode samples from $X$. You cannot compress below entropy on average — any lossless code must use at least $H(X)$ bits per symbol. Huffman coding and arithmetic coding approach this limit.

This connects directly to ML: when we train a model to predict tokens, the cross-entropy loss measures how many bits our model needs per token. A perfect model achieving the entropy of the true distribution is the theoretical optimum. The perplexity of a language model is $2^{H(p,q)}$ (or $e^{H(p,q)}$ in nats) — a direct exponentiation of the cross-entropy, measuring the effective vocabulary size the model is "confused" among at each step.

### Differential Entropy: The Continuous Case

> **Intuition:** Differential entropy is the continuous analog of entropy, but it can be negative and is not invariant to reparameterization — it measures entropy relative to the Lebesgue measure (uniform density on the real line). Squeezing a distribution into a very small interval makes it highly concentrated, so its density is large, and the log of a large density is positive; averaging $-f \log f$ can then go negative. What is invariant is the *difference* of differential entropies, which is why KL divergence and mutual information remain well-defined for continuous variables.

For continuous random variables $X$ with density $f(x)$, the **differential entropy** is:

$$h(X) = -\int f(x) \log f(x) \, dx$$

Unlike discrete entropy, differential entropy can be negative. For example, a uniform distribution on $[0, a]$ has $h(X) = \log a$, which is negative when $a < 1$. The Gaussian $\mathcal{N}(\mu, \sigma^2)$ has differential entropy $h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$, and among all distributions with fixed variance, the Gaussian maximizes differential entropy — a result with deep connections to the central limit theorem.

Differential entropy lacks some properties of its discrete counterpart (e.g., it is not invariant under change of variables), but *differences* of differential entropies — such as MI and KL divergence — remain well-behaved and coordinate-invariant.

More precisely, if $Y = g(X)$ for a differentiable bijection $g$:

$$h(Y) = h(X) + \mathbb{E}_X\!\left[\log|g'(X)|\right]$$

This is the **log-Jacobian correction** — the entropy changes by the expected log-volume of the transformation. For a linear scaling $Y = cX$: $h(cX) = h(X) + \log|c|$, which gives the negative entropy of a distribution with very small support.

> **Key insight for normalizing flows:** A normalizing flow transforms a simple base distribution $p_z(z)$ via $x = g_\theta(z)$. The log-likelihood of $x$ is $\log p_x(x) = \log p_z(g^{-1}(x)) + \log|\det J_{g^{-1}}(x)|$ — exactly this Jacobian correction. This is why flow-based models require invertible transformations with tractable Jacobian determinants.

## Joint and Conditional Entropy

For two random variables $X$ and $Y$:

**Joint entropy** captures the total uncertainty in the pair:

$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

**Conditional entropy** measures the remaining uncertainty in $X$ after observing $Y$:

$$H(X|Y) = -\sum_{x,y} p(x,y) \log p(x|y) = \mathbb{E}_{Y}\left[H(X|Y=y)\right]$$

These satisfy the **chain rule of entropy**:

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

This is the information-theoretic analog of the probability chain rule $p(x,y) = p(x)p(y|x)$. It decomposes the total uncertainty of $(X,Y)$ into the uncertainty of $X$ plus the residual uncertainty of $Y$ given $X$.

> **Remember:** $H(X,Y) = H(X) + H(Y|X)$ — the chain rule for entropy. Conditioning can only reduce or preserve entropy, never increase it: $H(Y|X) \leq H(Y)$, with equality iff $X$ and $Y$ are independent. This is the information-theoretic reason that observing more data cannot make you less certain on average.

> **Key insight:** Conditioning always reduces entropy on average: $H(X|Y) \leq H(X)$, with equality iff $X$ and $Y$ are independent. Observation can only resolve uncertainty, never increase it.

:::quiz
question: "For a discrete random variable $X$ with $n$ equally likely outcomes, what is $H(X)$?"
options:
  - "$n$"
  - "$\\log_2 n$"
  - "$n \\log_2 n$"
  - "$1/n$"
correct: 1
explanation: "When all $n$ outcomes have probability $1/n$, entropy is $-\\sum_{i=1}^n \\frac{1}{n} \\log_2 \\frac{1}{n} = \\log_2 n$. This is the maximum entropy for $n$ outcomes."
:::

## Cross-Entropy: The ML Loss Function

> **Intuition:** The maximum entropy principle says: among all distributions consistent with your constraints, choose the one with the highest entropy. This is the "most honest" distribution — it assumes nothing beyond what the constraints tell you. A distribution with lower entropy would be implicitly claiming more certainty than your constraints justify. This principle uniquely selects the uniform distribution when you have no constraints, the Gaussian when you know only the mean and variance, and the Boltzmann distribution in statistical physics.

The **cross-entropy** between a true distribution $p$ and a model distribution $q$ is:

$$H(p, q) = -\sum_{x} p(x) \log q(x) = \mathbb{E}_{X \sim p}\left[-\log q(X)\right]$$

Cross-entropy measures the average number of bits needed to encode samples from $p$ using a code optimized for $q$. Since the optimal code for $p$ uses $H(p)$ bits, the cross-entropy is always at least as large:

$$H(p, q) \geq H(p)$$

The gap $H(p, q) - H(p) = D_{\text{KL}}(p \| q)$ is the KL divergence (next lesson). Minimizing cross-entropy is therefore equivalent to minimizing KL divergence to the true distribution.

### Why Cross-Entropy Is the Classification Loss

For a classification problem with true label $y$ (one-hot encoded as $p$) and predicted probabilities $q$, the cross-entropy loss reduces to:

$$\mathcal{L} = -\log q(y)$$

This is exactly the negative log-likelihood of the true class. Minimizing cross-entropy across the dataset is equivalent to maximum likelihood estimation of the model parameters. This isn't a coincidence — it's a deep connection: MLE and cross-entropy minimization are the same optimization problem, and both are grounded in the information-theoretic principle of using the fewest bits to describe the data.

> **Key insight:** Cross-entropy loss in ML is not an arbitrary choice. It is the unique loss function that corresponds to maximum likelihood estimation for categorical distributions, and it has the information-theoretic interpretation of minimizing the coding overhead of using your model instead of the true distribution.

## ML Connections

Entropy is the fundamental quantity in information-theoretic machine learning — it appears in every loss function that involves probability distributions.

- **Cross-Entropy Loss:** The standard classification loss $L = -\sum_c y_c \log \hat{p}_c$ is the cross-entropy $H(y, \hat{p})$ between the true label distribution and the predicted distribution. Minimizing cross-entropy is equivalent to maximum likelihood estimation. The gap $H(y, \hat{p}) - H(y)$ measures the KL divergence from true to predicted distribution.
- **Maximum Entropy Language Models:** Before neural LMs, maximum entropy models (MaxEnt) were the dominant NLP approach. They find the distribution with maximum entropy subject to feature expectation constraints — the logistic regression classifier is a MaxEnt model. This principle still underlies the log-linear structure of neural softmax outputs.
- **Entropy Regularization in RL:** Soft actor-critic (SAC) maximizes expected reward plus entropy: $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha H(\pi(\cdot|s_t))]$. The entropy bonus $\alpha H(\pi)$ encourages exploration and prevents premature convergence to a deterministic policy. Adjusting $\alpha$ trades off exploration vs exploitation.
- **Information Bottleneck and Representation Learning:** The information bottleneck objective $\min I(Z;X) - \beta I(Z;Y)$ explicitly trades off compression (minimizing $H(Z|Y=y)$) against prediction (maximizing $I(Z;Y)$). Mutual information is the difference of entropies: $I(Z;Y) = H(Z) - H(Z|Y)$.
- **Neural Data Compression:** Learned image codecs (BPG, JPEG-like neural codecs, Ballé et al.) are trained to minimize $R + \lambda D$ where $R = -\mathbb{E}[\log p(z)]$ is the entropy of the quantized representation and $D$ is distortion. Entropy coding (Huffman, arithmetic coding) converts this theoretical entropy into actual compressed bits.

> **Key insight:** Cross-entropy is not just a "standard loss" — it is the information-theoretic measure of how surprised your model is by the data. Every classification model, language model, and generative model trained with maximum likelihood is minimizing surprise, measured in bits. This is why entropy is the foundation of ML theory.

## Python Example: Entropy and Cross-Entropy

```python
import numpy as np
import torch
import torch.nn.functional as F

# --- Entropy and cross-entropy from scratch ---
p = np.array([0.25, 0.25, 0.25, 0.25])  # true distribution (uniform)
q = np.array([0.1, 0.2, 0.3, 0.4])       # model distribution

# Shannon entropy: H(p) — minimum achievable bits
H_p = -np.sum(p * np.log2(p))
print(f"H(p) = {H_p:.4f} bits")  # 2.0 bits (maximum for 4 outcomes)

# Cross-entropy: H(p, q) — bits needed using code optimized for q
H_pq = -np.sum(p * np.log2(q))
print(f"H(p, q) = {H_pq:.4f} bits")  # > 2.0 (penalty for wrong model)

# KL divergence: the gap
D_kl = H_pq - H_p
print(f"D_KL(p || q) = {D_kl:.4f} bits")

# --- PyTorch cross-entropy loss for classification ---
logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])  # raw model outputs
target = torch.tensor([0])                         # true class index

loss = F.cross_entropy(logits, target)  # applies softmax + NLL internally
print(f"PyTorch CE loss = {loss.item():.4f} nats")  # in nats (ln, not log2)
```

Note that PyTorch's `cross_entropy` uses natural logarithms (nats), not base-2 logarithms (bits). To convert: $1 \text{ nat} = \log_2 e \approx 1.443 \text{ bits}$.

:::quiz
question: "If $p$ is the true distribution and $q$ is a model, what does minimizing $H(p, q)$ with respect to $q$ achieve?"
options:
  - "It makes $q$ uniform"
  - "It finds the $q$ closest to $p$ in KL divergence"
  - "It maximizes the entropy of $q$"
  - "It minimizes the variance of $q$"
correct: 1
explanation: "$H(p, q) = H(p) + D_{\\text{KL}}(p \\| q)$. Since $H(p)$ is constant with respect to $q$, minimizing cross-entropy is equivalent to minimizing $D_{\text{KL}}(p \\| q)$, making $q$ as close to $p$ as possible in the KL sense."
:::

:::quiz
question: "The chain rule $H(X,Y) = H(X) + H(Y|X)$ implies what when $X$ and $Y$ are independent?"
options:
  - "$H(X,Y) = H(X)$"
  - "$H(X,Y) = H(X) \\cdot H(Y)$"
  - "$H(X,Y) = H(X) + H(Y)$"
  - "$H(X,Y) = 0$"
correct: 2
explanation: "When $X \\perp Y$, conditioning provides no information: $H(Y|X) = H(Y)$. So $H(X,Y) = H(X) + H(Y)$. Entropy is additive for independent variables — this is the additivity axiom in action."
:::

## Summary

Shannon entropy is the bedrock of information theory and, by extension, of modern ML loss functions. The key takeaways:

- **Self-information** $I(x) = -\log p(x)$ is the unique axiomatic measure of surprise.
- **Entropy** $H(X)$ is the average surprise — and the minimum achievable compression rate.
- **Cross-entropy** $H(p,q)$ measures coding cost under a wrong model and is equivalent to negative log-likelihood.
- The chain rule decomposes joint uncertainty into marginal and conditional parts.

These concepts recur throughout ML: cross-entropy losses, KL regularization in VAEs, mutual information in representation learning, and the source coding theorem as the theoretical limit of generative models.
