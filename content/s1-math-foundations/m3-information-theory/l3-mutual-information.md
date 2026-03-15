---
title: "Mutual Information & Data Processing Inequality"
estimatedMinutes: 35
tags: ["mutual-information", "DPI", "channel-capacity", "representation-learning", "MINE"]
prerequisites: ["l1-entropy", "l2-kl-divergence"]
---

## Mutual Information: Shared Uncertainty

**Mutual information** (MI) quantifies the statistical dependence between two random variables — the amount of information that observing one reveals about the other:

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

This definition admits several equivalent forms, each revealing a different facet:

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

The first form says: MI is the reduction in uncertainty about $X$ after observing $Y$. The third form says: MI is the total individual uncertainty minus the joint uncertainty — the "overlap" in the information Venn diagram.

Perhaps most illuminating is the KL divergence form:

$$I(X;Y) = D_{\text{KL}}\left(p(x,y) \,\|\, p(x)p(y)\right)$$

MI measures how far the joint distribution is from the product of marginals — how far $X$ and $Y$ are from independence.

### Fundamental Properties

- **Symmetry:** $I(X;Y) = I(Y;X)$. Unlike KL, MI is symmetric because it compares the joint to the product of marginals, where both variables play equivalent roles.
- **Non-negativity:** $I(X;Y) \geq 0$, with equality iff $X \perp Y$.
- **Relation to correlation:** For jointly Gaussian variables with correlation $\rho$, $I(X;Y) = -\frac{1}{2}\log(1 - \rho^2)$. MI captures *all* statistical dependence, not just linear correlation.
- **Self-information:** $I(X;X) = H(X)$. A variable is maximally informative about itself.

> **Key insight:** MI generalizes correlation to capture *any* form of dependence — nonlinear, non-monotonic, or otherwise. Two variables can have zero correlation but high MI (e.g., $Y = X^2$ with $X$ symmetric around zero). This makes MI the natural objective when learning representations that preserve arbitrary structure.

## Chain Rule for Mutual Information

MI satisfies its own chain rule. For three variables:

$$I(X; Y, Z) = I(X; Z) + I(X; Y | Z)$$

where the **conditional mutual information** is:

$$I(X; Y | Z) = H(X|Z) - H(X|Y,Z) = \mathbb{E}_Z\left[D_{\text{KL}}(p(x,y|z) \| p(x|z)p(y|z))\right]$$

This measures the dependence between $X$ and $Y$ that remains after accounting for $Z$. It can be zero even when $I(X;Y) > 0$ (if all dependence was mediated through $Z$), and it can be positive even when $I(X;Y) = 0$ (if conditioning on $Z$ reveals a relationship).

## The Data Processing Inequality

The **data processing inequality (DPI)** is one of the most consequential results in information theory. If $X \to Y \to Z$ forms a Markov chain (meaning $Z$ is conditionally independent of $X$ given $Y$: $p(z|x,y) = p(z|y)$), then:

$$I(X; Z) \leq I(X; Y)$$

**Proof sketch:** By the chain rule, $I(X; Y, Z) = I(X; Y) + I(X; Z|Y)$. But the Markov condition $X \to Y \to Z$ implies $I(X; Z|Y) = 0$ (once you know $Y$, $Z$ adds nothing about $X$). Also, $I(X; Y, Z) = I(X; Z) + I(X; Y|Z) \geq I(X; Z)$. Combining: $I(X; Y) = I(X; Y, Z) \geq I(X; Z)$.

The DPI says that **processing can only destroy information, never create it**. Every layer of a neural network, every function applied to data, every communication channel — all are subject to this fundamental constraint.

> **Key insight:** The DPI is why representation learning is hard. If your encoder $f$ maps input $X$ to representation $T = f(X)$, then $I(T; Y) \leq I(X; Y)$ for any target $Y$. The best you can hope for is to *preserve* the relevant information; you cannot manufacture information that wasn't in the input. The art is in discarding the right information — the irrelevant parts of $X$.

:::quiz
question: "A neural network maps input $X$ through layers $X \\to H_1 \\to H_2 \\to H_3 \\to \\hat{Y}$. By the DPI, which statement is guaranteed?"
options:
  - "$I(X; \\hat{Y}) \\geq I(X; H_1)$"
  - "$I(X; H_3) \\geq I(X; H_1)$"
  - "$I(X; H_1) \\geq I(X; H_2) \\geq I(X; H_3)$"
  - "$I(X; \\hat{Y}) = I(X; H_1)$"
correct: 2
explanation: "Each layer forms a Markov chain $X \\to H_1 \\to H_2 \\to H_3$, so DPI gives $I(X; H_1) \\geq I(X; H_2) \\geq I(X; H_3)$. Information about $X$ can only decrease (or stay the same) as we go deeper. Note: with deterministic layers and invertible activations, equality can hold."
:::

## Channel Capacity

Information theory was born from communication. A **discrete memoryless channel** takes input $X$ and produces output $Y$ according to a fixed conditional distribution $p(y|x)$. The **channel capacity** is:

$$C = \max_{p(x)} I(X; Y)$$

the maximum MI achievable over all input distributions. Shannon's channel coding theorem states that reliable communication at rates up to $C$ bits per channel use is achievable, and rates above $C$ are not.

For the binary symmetric channel with error probability $\epsilon$:

$$C = 1 - H(\epsilon) = 1 + \epsilon \log_2 \epsilon + (1-\epsilon) \log_2(1-\epsilon)$$

Channel capacity has a natural analog in ML: the **capacity** of a representation bottleneck. The maximum MI $I(X; T)$ that can flow through a bottleneck of dimension $d$ constrains the model's ability to distinguish inputs — directly connecting to the information bottleneck framework (Lesson 5).

## MI in Representation Learning

Given input $X$ and target $Y$, a representation $T = f_\theta(X)$ is useful if $I(T; Y)$ is large — the representation retains information about the target. The DPI constrains $I(T; Y) \leq I(X; Y)$, but within this bound, we want $T$ to maximize task-relevant information.

The challenge is that MI is notoriously difficult to compute in high dimensions. The joint density $p(x, y)$ is unknown, and density estimation in high dimensions is unreliable. This motivated a series of *neural MI estimators*.

### The MINE Estimator

The **Mutual Information Neural Estimation (MINE)** framework (Belghazi et al., 2018) applies the Donsker-Varadhan representation of KL divergence to MI:

$$I(X;Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y)) = \sup_T \left[\mathbb{E}_{p(x,y)}[T(x,y)] - \log \mathbb{E}_{p(x)p(y)}[e^{T(x,y)}]\right]$$

A neural network $T_\theta(x,y)$ parameterizes the test function. Joint samples $(x, y) \sim p(x,y)$ come from the data; marginal (independent) samples are obtained by shuffling $y$ values across the batch. The MINE estimator:

$$\hat{I}_{\text{MINE}}(X;Y) = \sup_\theta \left[\frac{1}{N}\sum_{i=1}^N T_\theta(x_i, y_i) - \log \frac{1}{N}\sum_{i=1}^N e^{T_\theta(x_i, y'_i)}\right]$$

where $(x_i, y_i)$ are joint samples and $(x_i, y'_i)$ are created by independently sampling $y'_i$ from the marginal.

MINE is consistent (converges to the true MI) but suffers from high variance, especially when MI is large. The $\log$-sum-exp term requires careful handling to avoid numerical instability.

**Practical note on MINE bias:** The naive MINE gradient estimator is biased because it takes gradients through the denominator $\log \mathbb{E}_{p(x)p(y)}[e^{T_\theta}]$. The standard fix is an **exponential moving average (EMA)** of the denominator: maintain $\bar{e}_t = (1-\alpha)\bar{e}_{t-1} + \alpha \cdot \mathbb{E}[e^{T_\theta}]$ and use $\bar{e}_t$ (treated as a constant) in the gradient. This reduces bias at the cost of introducing a tunable EMA coefficient $\alpha \in (0,1)$.

### The InfoNCE Connection

The **InfoNCE** loss (van den Oord et al., 2018) provides a more stable alternative. For a batch of $N$ paired samples $\{(x_i, y_i)\}$:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{f_\theta(x_i, y_i)}}{\sum_{j=1}^N e^{f_\theta(x_i, y_j)}}$$

This is a softmax cross-entropy over the "correct pair" among $N$ candidates. The connection to MI:

$$I(X;Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$$

InfoNCE provides a *lower bound* on MI, capped at $\log N$. This ceiling means that with batch size $N = 256$, InfoNCE can estimate at most $\log 256 \approx 5.5$ nats of MI. Larger batches give tighter bounds but increase computational cost.

This is the theoretical foundation of contrastive learning methods like SimCLR, MoCo, and CLIP: they all maximize a lower bound on MI between different views or modalities.

> **Key insight:** InfoNCE elegantly sidesteps the density estimation problem. Instead of estimating $p(x,y)$, it learns a *critic* $f_\theta(x,y)$ that distinguishes true pairs from random pairs. The contrastive formulation turns MI maximization into a classification problem, which neural networks handle well.

:::quiz
question: "The InfoNCE bound $I(X;Y) \\geq \\log N - \\mathcal{L}_{\\text{InfoNCE}}$ implies that with batch size $N=64$, the maximum estimable MI is approximately:"
options:
  - "64 nats"
  - "$\\log_2 64 = 6$ bits"
  - "$\\ln 64 \\approx 4.16$ nats"
  - "Unlimited, given enough training"
correct: 2
explanation: "The InfoNCE lower bound is capped at $\\log N$. With $N=64$, the maximum is $\\ln 64 \\approx 4.16$ nats. Even if the true MI is higher, InfoNCE cannot detect it with this batch size. This is a fundamental limitation: larger batches are needed to estimate larger MI values."
:::

## Python Example: MINE Estimator

```python
import torch
import torch.nn as nn

class MINENetwork(nn.Module):
    """Statistics network T_theta(x, y) for MINE estimation."""
    def __init__(self, x_dim=1, y_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))

def mine_estimate(T_net, x, y, y_shuffle):
    """Compute MINE lower bound on I(X;Y) using Donsker-Varadhan."""
    joint_scores = T_net(x, y)                    # T(x, y) for joint samples
    marginal_scores = T_net(x, y_shuffle)          # T(x, y') for marginals
    # DV bound: E_joint[T] - log E_marginal[e^T]
    mi_lb = joint_scores.mean() - torch.logsumexp(marginal_scores, 0) + torch.log(torch.tensor(float(len(x))))
    return mi_lb

# Example: correlated Gaussians with known MI
rho = 0.8
true_mi = -0.5 * torch.log(torch.tensor(1 - rho**2))  # analytical MI

T_net = MINENetwork()
optimizer = torch.optim.Adam(T_net.parameters(), lr=1e-3)

for step in range(2000):
    x = torch.randn(256, 1)
    y = rho * x + (1 - rho**2)**0.5 * torch.randn(256, 1)  # correlated
    y_shuffle = y[torch.randperm(256)]                        # break dependence

    loss = -mine_estimate(T_net, x, y, y_shuffle)  # maximize MI → minimize -MI
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if (step + 1) % 500 == 0:
        print(f"Step {step+1}: MINE={-loss.item():.3f}, True MI={true_mi.item():.3f}")
```

With $\rho = 0.8$, the true MI is $-\frac{1}{2}\ln(1-0.64) \approx 0.51$ nats. MINE should converge close to this value, though with some variance due to the $\log$-sum-exp term.

:::quiz
question: "Two random variables $X$ and $Y$ have zero Pearson correlation ($\\rho = 0$) but $Y = X^2$ where $X \\sim \\mathcal{N}(0,1)$. What can we say about $I(X;Y)$?"
options:
  - "$I(X;Y) = 0$ because they are uncorrelated"
  - "$I(X;Y) > 0$ because MI captures nonlinear dependence"
  - "$I(X;Y) = H(X)$ because $Y$ is a function of $X$"
  - "$I(X;Y)$ is undefined for continuous variables"
correct: 1
explanation: "$Y = X^2$ is a deterministic function of $X$, so knowing $X$ determines $Y$ completely — they are strongly dependent. MI captures all forms of dependence, including nonlinear ones that correlation misses. However, $I(X;Y) < H(X)$ because the mapping is not invertible: knowing $Y = x^2$ does not uniquely determine $X$ (it could be $+x$ or $-x$)."
:::

## Summary

Mutual information is the canonical measure of statistical dependence: symmetric, non-negative, and zero iff independence. The data processing inequality constrains what any representation can achieve — processing destroys information. Channel capacity sets the fundamental limit of communication, and MI estimation via MINE and InfoNCE provides the practical tools that power contrastive representation learning. The InfoNCE bound, in particular, connects information theory directly to the softmax cross-entropy losses ubiquitous in self-supervised learning.
