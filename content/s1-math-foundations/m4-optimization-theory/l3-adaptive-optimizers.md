---
title: "Adaptive Gradient Methods"
estimatedMinutes: 35
tags: ["Adam", "AdaGrad", "RMSProp", "AdamW", "learning-rate", "optimizer", "Lion"]
prerequisites: ["l2-gradient-methods"]
---

## Motivation: Why Adapt the Learning Rate?

Standard gradient descent uses the same step size for every parameter, in every direction. But in deep learning, different parameters have very different gradient magnitudes: embedding weights receive sparse, infrequent, large gradients while normalization parameters receive dense, small gradients. A single global step size is forced to be small enough to avoid exploding for the large-gradient parameters, which makes progress glacially slow for the small-gradient parameters.

**Adaptive methods** maintain a per-parameter effective learning rate, automatically scaling each coordinate's step size by its historical gradient information. This section derives the major adaptive optimizers from first principles and analyzes their theoretical and empirical properties.

## AdaGrad: Per-Coordinate Adaptation

**AdaGrad** (Duchi, Hazan & Singer, 2011) accumulates the sum of squared gradients for each coordinate and scales the step size accordingly.

> **Intuition:** Sparse features (e.g., rare words in an embedding table) should get larger updates since they appear infrequently and carry a strong signal when they do. AdaGrad achieves this automatically by dividing each coordinate's gradient by the square root of its accumulated squared history: rarely-seen coordinates accumulate little history, keeping their effective learning rate large, while frequently-seen coordinates accumulate large history and shrink their step. This is why AdaGrad transformed NLP training when it was introduced.

**Algorithm.** Initialize $x_0$, $G_0 = \varepsilon I$ (small diagonal). At each step $t$:

$$G_t = G_{t-1} + g_t g_t^\top \quad \text{(full matrix version)}$$
$$G_{t,ii} \leftarrow G_{t-1,ii} + g_{t,i}^2 \quad \text{(diagonal approximation)}$$
$$x_{t+1} = x_t - \frac{\eta}{\sqrt{G_{t,ii}} + \varepsilon} \odot g_t$$

where $\odot$ is elementwise multiplication. Parameters with large historical gradients get smaller effective step sizes; rarely updated parameters (sparse gradients) maintain a larger step size.

**Why it helps for sparse data.** In NLP with bag-of-words features or embedding tables, most gradient entries are zero at each step. AdaGrad accumulates small values in the denominator for rarely-seen features, preserving their learning rates. For frequently-seen features the denominator grows, preventing large oscillations.

**Convergence.** For convex functions, AdaGrad achieves a $O(\sqrt{T})$ regret bound in the online learning sense, which is optimal for the general case. The bound adapts to the geometry: in directions with small total gradient energy, convergence is faster.

**The monotone decay problem.** Because $G_t$ only grows, the effective step size $\eta/\sqrt{G_t}$ monotonically decreases to zero. In non-convex training (deep nets), this premature decay often stalls learning before reaching a good solution — the optimizer effectively stops learning in the middle of training. This motivated RMSProp and Adam.

> **Key insight:** Adaptive methods accelerate early training by adjusting per-coordinate step sizes, but may generalize worse than SGD — the "adaptive ≠ better generalization" phenomenon observed empirically across many image classification benchmarks.

## RMSProp: Fixing Monotone Decay

**RMSProp** (Hinton, unpublished Coursera notes, 2012) replaces the cumulative sum in AdaGrad with an **exponential moving average (EMA)** of squared gradients, controlled by a decay rate $\rho$:

$$v_t = \rho v_{t-1} + (1-\rho) g_t^2 \quad \text{(elementwise)}$$
$$x_{t+1} = x_t - \frac{\eta}{\sqrt{v_t} + \varepsilon} g_t$$

Typical value: $\rho = 0.9$ or $0.99$. The EMA forgets old gradients with time constant $1/(1-\rho)$, so the effective step size remains bounded away from zero even after many steps. This allows continued learning throughout training.

**Interpretation.** $v_t$ estimates the second moment $\mathbb{E}[g^2]$ of the gradient. The update $g_t / \sqrt{v_t}$ normalizes the gradient by its root-mean-square (RMS), making the effective update scale-invariant with respect to the gradient magnitude.

## Adam: First and Second Moments with Bias Correction

**Adam** (Kingma & Ba, 2014) combines the momentum idea (first moment) with RMSProp's adaptive scaling (second moment), with a crucial **bias correction** step.

**Algorithm.** Initialize $x_0$, $m_0 = 0$, $v_0 = 0$, hyperparameters $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. At step $t$:

**First moment (momentum):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment (adaptive scale):**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias-corrected estimates:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**
$$x_{t+1} = x_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

**Why bias correction?** At $t=1$, $m_1 = (1-\beta_1)g_1 = 0.1 g_1$, which severely underestimates the true gradient because the EMA is initialized at zero. Dividing by $(1 - \beta_1^t)$ corrects this: $\hat{m}_1 = g_1$. After many steps, $\beta_1^t \approx 0$ and bias correction has negligible effect.

> **Remember:** Adam initializes both moment estimates at zero, creating a systematic bias toward zero in the early steps. The correction factors $1/(1-\beta_1^t)$ and $1/(1-\beta_2^t)$ undo this by rescaling the EMAs toward their true values. For $\beta_1 = 0.9$, the first-moment correction is already near 1 after about 50 steps; for $\beta_2 = 0.999$, the second-moment correction takes roughly 5000 steps to become negligible. This is precisely why warmup matters for long training runs.

**Interpretation of the update.** The effective step at time $t$ is:

$$\Delta x_t = -\eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} \approx -\eta \cdot \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}$$

This is a signal-to-noise ratio: when the gradient is consistently pointing in one direction (large $|\hat{m}_t|$ relative to $\sqrt{\hat{v}_t}$), the effective step is $\approx \pm\eta$. When the gradient is noisy ($\hat{m}_t \approx 0$, $\hat{v}_t > 0$), the step shrinks. The effective learning rate per coordinate stays in $[-\eta, \eta]$ (approximately).

## Adam's Convergence Issues and AMSGrad

**Non-convergence counterexample.** Reddi, Kale & Kumar (2018) showed that Adam can fail to converge even on simple convex online problems. The issue: the effective step size $\eta / \sqrt{v_t}$ can *increase* at some steps if $v_t$ decreases (which can happen because $v_t$ is an EMA, not a cumulative sum). This violates the decreasing step size requirement for convergence.

**Counterexample sketch.** Consider a 1D problem where gradients cycle: mostly small, occasionally large. The EMA $v_t$ tracks recent gradients — after a large gradient, $v_t$ is large (small step), but as subsequent small gradients arrive, $v_t$ shrinks (effective step grows). This can create a systematic bias in the wrong direction.

**AMSGrad fix.** Replace $\hat{v}_t$ with $\hat{v}_t^{\max} = \max(\hat{v}_{t-1}^{\max}, \hat{v}_t)$, ensuring the denominator is non-decreasing:

$$x_{t+1} = x_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t^{\max}} + \varepsilon}$$

AMSGrad has convergence guarantees but performs similarly to or worse than Adam in practice — the non-convergence issue rarely matters for typical deep learning objectives.

## AdamW: Decoupled Weight Decay

A critical insight due to Loshchilov & Hutter (2019): **L2 regularization and weight decay are not equivalent in Adam**.

**L2 regularization** adds $\frac{\lambda}{2}\|x\|^2$ to the loss. In gradient descent, this modifies the gradient: $\tilde{g}_t = g_t + \lambda x_t$, and the update is $x_{t+1} = x_t - \eta(\tilde{g}_t) = (1 - \eta\lambda)x_t - \eta g_t$.

In Adam, the regularized gradient $\tilde{g}_t = g_t + \lambda x_t$ enters both the numerator ($\hat{m}_t$) and denominator ($\sqrt{\hat{v}_t}$). The denominator adapts to the regularization term, which changes the *effective* magnitude of regularization per coordinate — parameters with large gradients get less regularization than intended.

> **Intuition:** With L2 regularization baked into the gradient, the regularization term $\lambda x_t$ enters the adaptive denominator $\sqrt{\hat{v}_t}$. This means parameters with large gradient history get their weight decay divided by a large number — they receive less shrinkage than intended. AdamW decouples this by applying weight decay directly to the parameters ($\theta \leftarrow (1-\lambda\eta)\theta$) before the Adam step, giving every parameter the same proportional shrinkage regardless of its gradient history.

**Decoupled weight decay (AdamW).** Apply weight decay directly to the parameters, separate from the adaptive gradient update:

$$x_{t+1} = (1 - \eta\lambda) x_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

This ensures every parameter is shrunk by the same factor $(1 - \eta\lambda)$ regardless of its gradient history. AdamW is the default optimizer for transformer pre-training (GPT, BERT, LLaMA) because it provides proper regularization.

> **Key insight:** Weight decay in Adam must be decoupled (AdamW) — L2 regularization in the gradient does not actually regularize in the same way because the adaptive denominator absorbs part of the regularization signal.

## Lion: Sign-Based Updates

**Lion** (Chen et al., 2023) emerged from program search (EvoLM) and has strong empirical performance with reduced memory requirements.

**Algorithm.** With momentum $m_t$ and hyperparameters $\beta_1, \beta_2$:

$$\Delta x_t = -\eta \cdot \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t)$$
$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$$

Key properties:
- Update is always $\pm\eta$ per coordinate — **constant effective step size**
- Requires storing only $m_t$ (not $v_t$), saving 33% memory vs Adam
- The sign operation provides aggressive implicit gradient clipping
- Requires smaller learning rates than Adam (updates are larger in magnitude)

**Trade-offs.** Lion is competitive with AdamW on vision and language tasks but less studied theoretically. Its constant-magnitude updates make it more sensitive to learning rate choice. Best results use $\eta_{\text{Lion}} \approx \eta_{\text{Adam}}/10$.

## Learning Rate Schedules

The learning rate schedule is as important as the optimizer choice.

**Constant.** $\eta_t = \eta$. Simple but oscillates near the minimum.

**Step decay.** $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$. Drop by factor $\gamma$ every $k$ steps. Common in ResNet training.

**Cosine annealing.** Smooth decay following a cosine:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

> **Refresher:** Cosine annealing starts at $\eta_{\max}$, decreases smoothly to $\eta_{\min}$ following the shape of a cosine curve, and can optionally restart (cosine restarts). The "linear warmup" phase before cosine decay increases the learning rate linearly from 0 to $\eta_{\max}$ over the first few hundred steps. Warmup stabilizes early training when the second moment estimate $\hat{v}_t$ is still small and the effective step size would otherwise be erroneously large.

At $t=0$: $\eta = \eta_{\max}$. At $t=T$: $\eta = \eta_{\min}$. The smooth decay avoids abrupt changes and works well empirically.

**Linear warmup + cosine decay.** The standard schedule for LLM pre-training:

$$\eta_t = \begin{cases} \eta_{\max} \cdot t / T_{\text{warmup}} & t \leq T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1 + \cos(\pi (t - T_{\text{warmup}}) / (T - T_{\text{warmup}}))) & t > T_{\text{warmup}} \end{cases}$$

Warmup is critical: at the start of training, the second moment estimate $v_t$ is small (near zero), so even with bias correction the effective step size is large. Large initial updates with a random network can destroy structure in the early loss landscape. Warmup ramps up the learning rate slowly, giving $v_t$ time to accumulate meaningful estimates.

> **Key insight:** Warmup prevents large early gradient steps that destabilize training — it is not heuristic engineering but a theoretically motivated response to the bias correction instability of Adam at initialization.

**1cycle policy.** Increase $\eta$ from $\eta_{\min}$ to $\eta_{\max}$ over 30% of training, then decrease to $\eta_{\min}/100$ (and possibly reduce momentum). Empirically accelerates training and allows larger learning rates overall (superconvergence).

**Practical guidance:**
- **Transformers** (LLMs, ViT): AdamW + linear warmup + cosine decay. $\eta = 1\text{e-}3$ to $3\text{e-}4$, $\lambda = 0.1$, warmup $= 1\%$–$5\%$ of steps
- **Vision CNNs** (ResNet, EfficientNet): SGD + momentum ($0.9$) + cosine decay often competitive with Adam; weight decay $5\text{e-}4$
- **Large-scale training**: Lion + cosine decay for memory efficiency

## ML Connections

Adaptive gradient methods are the workhorses of modern deep learning — AdamW trains virtually every large language model, and understanding their theory explains the most important hyperparameter choices in ML practice.

- **Large Language Model Training (AdamW + Cosine Decay):** GPT-4, LLaMA, Gemini, and Claude are all trained with AdamW + linear warmup + cosine decay schedule. The warmup addresses Adam's bias correction instability in early steps; cosine decay anneals the learning rate as training converges; AdamW's decoupled weight decay provides uniform regularization across all coordinates.
- **Vision Model Training (SGD + Momentum):** ResNet, EfficientNet, and most CNN-based classifiers use SGD with momentum (not Adam) because empirical evidence shows SGD finds flatter minima and generalizes better for image classification. The implicit noise of SGD (proportional to learning rate / batch size) provides regularization that adaptive methods reduce.
- **LoRA and Adapter Training:** Fine-tuning with LoRA uses AdamW on only the low-rank matrices B, A. The decoupled weight decay is critical — with Adam+L2, large-gradient coordinates (common in B matrices initialized as Gaussian) would receive weaker regularization, causing overfitting to small fine-tuning datasets.
- **Reward Model Training (RLHF Pipeline):** Reward models in RLHF use AdamW with small learning rates (1e-5 to 1e-6) and aggressive warmup (10% of steps). The very small learning rate compensates for the small, noisy preference datasets; warmup prevents early divergence from the pretrained checkpoint.
- **Diffusion Model Training:** Diffusion models (DDPM, Stable Diffusion, FLUX) use AdamW with β₁=0.9, β₂=0.999 and a constant or cosine learning rate (no warmup needed as the score matching objective is relatively smooth). The ε=1e-8 term prevents division by zero in the early steps when second moment estimates are near zero.

> **Key insight:** AdamW + warmup + cosine decay is the de facto standard for training transformers because it solves three problems simultaneously: bias correction instability (warmup), non-uniform regularization (decoupled weight decay), and learning rate annealing (cosine). Every component has a theoretical motivation — this is not a heuristic but an engineered solution to known optimization problems.

## Python: Adam and AdamW from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic regression problem
n, d = 200, 30
X = np.random.randn(n, d)
w_true = np.random.randn(d) * 0.5
y = X @ w_true + 0.1 * np.random.randn(n)

def mse_loss(w):
    return 0.5 * np.mean((X @ w - y)**2)

def mse_grad(w):
    return X.T @ (X @ w - y) / n


# --- Adam from scratch ---
def adam(grad_fn, x_init, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, T=500):
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []
    eff_lrs = []  # track effective learning rates for a single coordinate

    for t in range(1, T + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        x = x - step
        losses.append(mse_loss(x))
        eff_lrs.append(lr / (np.sqrt(v_hat[0]) + eps))  # coord 0

    return x, losses, eff_lrs


# --- AdamW from scratch ---
def adamw(grad_fn, loss_fn, x_init, lr=1e-3, beta1=0.9, beta2=0.999,
          eps=1e-8, weight_decay=0.01, T=500):
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []

    for t in range(1, T + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # Decoupled weight decay: shrink parameters BEFORE gradient step
        x = (1 - lr * weight_decay) * x - lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(loss_fn(x))

    return x, losses


# --- Adam with L2 in gradient (incorrect way) ---
def adam_l2_regularized(grad_fn, loss_fn, x_init, lr=1e-3, lam=0.01, T=500):
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, T + 1):
        g = grad_fn(x) + lam * x  # L2 reg added to gradient
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(loss_fn(x))

    return x, losses


x0 = np.zeros(d)

_, losses_adam, eff_lrs = adam(mse_grad, x0, lr=1e-2, T=500)
_, losses_adamw = adamw(mse_grad, mse_loss, x0, lr=1e-2, weight_decay=0.1, T=500)
_, losses_adam_l2 = adam_l2_regularized(mse_grad, mse_loss, x0, lr=1e-2, lam=0.1, T=500)

# Cosine annealing schedule
T_total = 500
eta_max, eta_min = 1e-2, 1e-5
cosine_lrs = [eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T_total))
              for t in range(T_total)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Convergence comparison
axes[0].semilogy(losses_adam, label='Adam (no regularization)')
axes[0].semilogy(losses_adamw, label='AdamW (decoupled WD=0.1)')
axes[0].semilogy(losses_adam_l2, label='Adam + L2 in gradient (WD=0.1)')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('MSE Loss (log scale)')
axes[0].set_title('Adam vs AdamW vs Adam+L2')
axes[0].legend(fontsize=8)
axes[0].grid(True)

# Effective learning rate dynamics (shows bias correction effect)
axes[1].plot(eff_lrs[:100])
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Effective LR for coordinate 0')
axes[1].set_title('Effective Learning Rate in Adam\n(bias correction effect visible early)')
axes[1].grid(True)

# Cosine annealing schedule
axes[2].plot(cosine_lrs)
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title(f'Cosine Annealing Schedule\n(η_max={eta_max:.0e}, η_min={eta_min:.0e})')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('adaptive_optimizers.png', dpi=150)
plt.show()

# Show weight decay behavior difference
w_adam_no_wd = adam(mse_grad, x0, lr=1e-2, T=1000)[0]
w_adamw_wd, _ = adamw(mse_grad, mse_loss, x0, lr=1e-2, weight_decay=0.5, T=1000)
w_adam_l2, _ = adam_l2_regularized(mse_grad, mse_loss, x0, lr=1e-2, lam=0.5, T=1000)

print(f"||w||₂ — Adam no WD:    {np.linalg.norm(w_adam_no_wd):.3f}")
print(f"||w||₂ — AdamW (λ=0.5): {np.linalg.norm(w_adamw_wd):.3f}")
print(f"||w||₂ — Adam+L2 (λ=0.5): {np.linalg.norm(w_adam_l2):.3f}")
print("AdamW and Adam+L2 produce different norms — they are not equivalent!")
```

:::quiz
question: "Why does Adam require bias correction (dividing m_t and v_t by 1 − β^t), and when does bias correction have negligible effect?"
options:
  - "Bias correction prevents numerical overflow in the denominator; it becomes negligible when gradients are small"
  - "Bias correction rescales the EMA estimates to approximate the true moments; it becomes negligible when t is large and β^t ≈ 0"
  - "Bias correction prevents the learning rate from decaying too fast; it becomes negligible when the loss plateaus"
  - "Bias correction is only needed for non-stationary gradient distributions; it becomes negligible after warmup"
correct: 1
explanation: "At t=1, m₁ = (1−β₁)g₁, which underestimates the true first moment g₁ by a factor of (1−β₁) because the EMA was initialized at zero. Dividing by (1−β₁¹) = (1−β₁) corrects this to m̂₁ = g₁. Similarly for v̂₁. As t grows, β^t → 0 (since β < 1), so the bias correction factor 1/(1−β^t) → 1 and has negligible effect. For β₁=0.9, this happens around t ≈ 50; for β₂=0.999, around t ≈ 5000. This is why warmup is needed for long training runs."
:::

:::quiz
question: "Consider training a transformer with AdamW. Why is it incorrect to implement weight decay as L2 regularization (adding λw to the gradient) in this case?"
options:
  - "L2 regularization is non-convex in the transformer architecture due to skip connections"
  - "Adding λw to the gradient means the regularization term is divided by √v̂_t in the update, so parameters with large gradient history receive less shrinkage than intended"
  - "L2 regularization prevents the bias correction from working correctly at early training steps"
  - "The transformer loss landscape is non-smooth, making gradient-based regularization ineffective"
correct: 1
explanation: "When λw is added to the gradient, it enters both the numerator (m̂_t) and denominator (√v̂_t) of the Adam update. The effective regularization per coordinate is λ/(√v̂_t + ε), which varies across coordinates based on gradient history. Coordinates with large gradients (large v̂_t) receive less regularization than intended, while rarely-updated coordinates receive more. AdamW bypasses this by applying the weight decay directly: x ← (1−ηλ)x − η·m̂_t/(√v̂_t+ε), ensuring uniform shrinkage regardless of gradient history."
:::

:::quiz
question: "The cosine annealing schedule η_t = η_min + ½(η_max − η_min)(1 + cos(πt/T)) is preferred over linear decay for transformer training. What mathematical property makes it advantageous?"
options:
  - "Cosine annealing satisfies the Robbins-Monro conditions Ση_t = ∞ and Ση_t² < ∞ for SGD convergence, unlike linear decay"
  - "The cosine schedule has zero derivative at t=0 and t=T (smooth start and end), spending more time at intermediate learning rates than linear decay"
  - "Cosine annealing provably achieves a lower loss than any monotonically decreasing schedule"
  - "The cosine function's periodicity enables the optimizer to escape local minima at each cycle"
correct: 1
explanation: "The cosine schedule has dη/dt = 0 at t=0 (maximum, slow initial decay) and t=T (minimum, slow final decay), with the fastest change in the middle of training. This means the schedule spends proportionally more time at high learning rates (effective exploration early) and transitions smoothly to low learning rates (fine-tuning convergence late). Linear decay drops the learning rate too aggressively early on. Additionally, the smooth start avoids the instability that can occur when momentum-based optimizers experience sudden learning rate changes."
:::
