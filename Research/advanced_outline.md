# Advanced AI/ML Graduate Curriculum: 8-Semester Learning Journey

This curriculum is designed as a rigorous, graduate-level program spanning 8 semesters. It builds from mathematical foundations through frontier research topics, ensuring graduates are prepared for both academic research and industry leadership in AI/ML. Each semester contains 4–6 modules with indicative subtopics. The final semester consolidates several critical areas—including time series, uncertainty quantification, scientific ML, and more—that the original 7-topic outline was missing.

***

## Semester 1: Mathematical Foundations for Advanced LLMs and ML

This semester establishes the mathematical language underpinning all modern ML. It goes well beyond introductory linear algebra and calculus, targeting the specific mathematics that appear in transformer theory, generative models, and optimization landscapes.

### Module 1.1 — Advanced Linear Algebra and Matrix Theory
- Spectral decomposition, SVD, and low-rank approximations (foundation for LoRA, PCA, and attention)[1]
- Random matrix theory: Marchenko-Pastur law, eigenvalue distributions of weight matrices
- Tensor algebra and Tucker/CP decompositions for multi-dimensional data
- Matrix calculus: Jacobians, Hessians, and their role in backpropagation

### Module 1.2 — Probability Theory and Stochastic Processes
- Measure-theoretic probability: σ-algebras, Lebesgue integration, Radon-Nikodym derivatives
- Concentration inequalities: Hoeffding, Bernstein, McDiarmid, and sub-Gaussian tail bounds
- Martingales and stopping times (used in sequential decision-making and RL theory)
- Stochastic differential equations (SDEs): Itô calculus, Fokker-Planck equations (critical for diffusion models)[2]

### Module 1.3 — Information Theory for Machine Learning
- Entropy, KL divergence, mutual information, and the data-processing inequality[3]
- f-divergences: total variation, Hellinger, chi-squared — and their variational representations[3]
- Rate-distortion theory and connections to model compression
- Information-theoretic lower bounds: Fano's inequality, Le Cam's method, Assouad's lemma[3]
- Information bottleneck principle and its role in deep learning representations[4]

### Module 1.4 — Optimization Theory
- Convex optimization: duality, KKT conditions, proximal methods, ADMM
- Non-convex optimization landscape: saddle points, loss surface geometry in deep networks
- Stochastic gradient methods: SGD, Adam, AdaGrad, and modern variants (LAMB, Muon)[5]
- Convergence theory: rates under convexity, PL-condition, and overparameterized regimes
- Natural gradient and Fisher information geometry

### Module 1.5 — Optimal Transport
- Monge and Kantorovich formulations, Brenier's theorem[6][7]
- Wasserstein distances and their use as loss functions in generative models
- Entropic regularization and Sinkhorn algorithm[8]
- Gradient flows and Benamou-Brenier dynamic formulation[7]
- Applications: GANs (Wasserstein GAN), domain adaptation, token dynamics in transformers[8]

### Module 1.6 — Functional Analysis and Approximation Theory
- Reproducing Kernel Hilbert Spaces (RKHS): Mercer's theorem, representer theorem
- Universal approximation theorems for neural networks (Cybenko, Barron)
- Neural Tangent Kernel (NTK) framework and infinite-width limits
- Sobolev spaces and their relevance to Physics-Informed Neural Networks

***

## Semester 2: Advanced Classical Machine Learning

Before diving into deep learning, this semester ensures mastery of the classical techniques that remain indispensable — and that deeply inform modern architectures.

### Module 2.1 — Bayesian Machine Learning
- Bayesian inference: conjugate priors, posterior computation, predictive distributions[9][10]
- Gaussian Processes: kernels, hyperparameter optimization, GP classification[10]
- Approximate inference: Laplace approximation, variational Bayes, expectation propagation[9]
- MCMC methods: Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS
- Bayesian model comparison and Bayes factors
- Bayesian neural networks and weight uncertainty

### Module 2.2 — Kernel Methods and Nonparametric Models
- Kernel trick, Mercer kernels, and kernel construction (polynomial, RBF, Matérn)[11]
- Support Vector Machines: primal/dual, soft-margin, structured output SVMs[12]
- Kernel PCA, kernel CCA, and maximum mean discrepancy (MMD)
- Multiple kernel learning and deep kernel learning
- Connections between kernels and neural networks (NTK, Random Features approximations)

### Module 2.3 — Probabilistic Graphical Models
- Bayesian Networks: d-separation, structure learning, exact and approximate inference[13][12]
- Markov Random Fields: factor graphs, belief propagation, junction tree algorithm[13]
- Hidden Markov Models and Conditional Random Fields
- Variational inference: mean-field, structured variational methods, amortized inference[12]
- The EM algorithm: theory, convergence, and applications to mixture models

### Module 2.4 — Ensemble Methods and Advanced Supervised Learning
- Boosting theory: AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost
- Random Forests and feature importance analysis
- Stacking, blending, and model averaging
- Learning theory: VC dimension, Rademacher complexity, PAC-Bayes bounds
- Bias-variance tradeoff and double descent in modern overparameterized models

### Module 2.5 — Dimensionality Reduction and Representation Learning
- Spectral methods: PCA, ICA, CCA, and their kernel variants
- Manifold learning: t-SNE, UMAP, diffusion maps, Laplacian eigenmaps
- Sparse coding and dictionary learning
- Non-negative matrix factorization
- Autoencoders as a bridge to deep representation learning

***

## Semester 3: Deep Learning — Theory, Architecture, and Optimization

This semester covers the core architectures, theoretical insights, and optimization strategies that underpin modern deep learning.

### Module 3.1 — Foundations of Deep Learning Theory
- Expressivity: depth vs. width, benefits of overparameterization
- Generalization: implicit regularization, flat minima, and PAC-Bayes bounds for deep networks
- Loss landscape geometry: mode connectivity, lottery ticket hypothesis, linear mode connectivity
- Neural Tangent Kernel regime vs. feature learning (rich) regime
- Scaling laws: Kaplan and Chinchilla power-law relationships between compute, data, and parameters[14][15]
- Grokking and delayed generalization phenomena[16]

### Module 3.2 — Convolutional and Recurrent Architectures
- CNNs: residual connections (ResNet), dense connections (DenseNet), depth-wise separable convolutions
- RNNs: LSTM, GRU, and the vanishing/exploding gradient problem[17]
- Sequence-to-sequence models and the encoder-decoder paradigm[18]
- Historical context leading to attention mechanisms

### Module 3.3 — Attention Mechanisms and Transformer Architectures
- Self-attention, multi-head attention, and positional encoding (sinusoidal, RoPE, ALiBi)
- The Transformer architecture: encoder-only, decoder-only, and encoder-decoder variants[19]
- Efficient attention: linear attention, FlashAttention, sparse attention patterns[20]
- Vision Transformers (ViT), Audio Spectogram Transformers

### Module 3.4 — State Space Models and Beyond Transformers
- Structured State Space Models (S4): continuous-time LTI systems, HiPPO initialization[21][22]
- Mamba: selective state spaces with input-dependent dynamics[23][24]
- Comparison with Transformers: linear-time complexity vs. quadratic attention[22]
- Hybrid architectures: Mamba-Transformer combinations, Jamba, Zamba[22]
- RetNet, RWKV, and linear recurrent units (LRUs)

### Module 3.5 — Advanced Optimization for Deep Networks
- Batch normalization, layer normalization, RMSNorm, and their theoretical justifications[17]
- Learning rate scheduling: warmup, cosine annealing, cyclical learning rates
- Gradient clipping, gradient accumulation, and mixed-precision training
- Distributed training: data parallelism, model parallelism, pipeline parallelism, ZeRO
- Second-order methods: K-FAC, Shampoo, and natural gradient approximations

### Module 3.6 — Regularization and Generalization in Practice
- Dropout, DropPath, stochastic depth, and weight decay
- Data augmentation: Mixup, CutMix, RandAugment, test-time augmentation
- Self-supervised pre-training as regularization
- Early stopping, model selection, and cross-validation at scale

***

## Semester 4: Deep Reinforcement Learning

This semester covers RL from foundations through frontier topics, with emphasis on deep function approximation and real-world applications.

### Module 4.1 — Foundations of RL
- Markov Decision Processes: Bellman equations, value iteration, policy iteration[25][26]
- Dynamic programming and its approximations[25]
- Monte Carlo prediction and control methods[25]
- Temporal-Difference learning: TD(0), TD(λ), SARSA, Q-learning[25]

### Module 4.2 — Deep Value-Based Methods
- Deep Q-Networks (DQN): experience replay, target networks, double DQN[26]
- Dueling DQN, prioritized experience replay
- Distributional RL: C51, QR-DQN, IQN
- Rainbow and beyond: combining improvements

### Module 4.3 — Policy Gradient and Actor-Critic Methods
- REINFORCE, variance reduction via baselines[26]
- Actor-Critic: A2C, A3C, and importance sampling corrections
- Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO)[27]
- Soft Actor-Critic (SAC) and maximum entropy RL
- Deterministic policy gradients (DDPG, TD3)

### Module 4.4 — Model-Based RL
- World models: learning environment dynamics[26]
- Dyna-style architectures and model-predictive control
- Monte Carlo Tree Search (AlphaGo, AlphaZero, MuZero)[26]
- Dreamer and latent-space planning

### Module 4.5 — Advanced RL Topics
- Offline RL: conservative Q-learning (CQL), decision transformers[27][25]
- Multi-agent RL: cooperative, competitive, and mixed settings[26]
- Hierarchical RL: options framework, goal-conditioned policies[26]
- Imitation learning and inverse RL[25]
- Meta-RL and few-shot adaptation[26]
- RL from Human Feedback (RLHF) — bridge to Semester 6[28]

***

## Semester 5: Generative AI — VAEs, GANs, Diffusion Models, and Flow Matching

This semester provides deep theoretical and practical coverage of all major generative modeling paradigms.

### Module 5.1 — Variational Autoencoders (VAEs)
- Latent variable models and the evidence lower bound (ELBO)
- Reparameterization trick and amortized inference
- β-VAE, VQ-VAE, and discrete latent spaces
- Hierarchical VAEs (NVAE, VDVAE) and their role in high-resolution generation
- VAE-based anomaly detection and time series applications[29]

### Module 5.2 — Generative Adversarial Networks (GANs)
- Minimax formulation and Nash equilibria
- Training instabilities: mode collapse, vanishing gradients, and spectral normalization
- Wasserstein GAN and gradient penalty (connection to optimal transport from Semester 1)
- Conditional GANs: cGAN, Pix2Pix, CycleGAN, StyleGAN
- Evaluation metrics: FID, IS, precision-recall for generative models

### Module 5.3 — Normalizing Flows
- Change-of-variables formula and invertible architectures
- Coupling layers (RealNVP, Glow) and autoregressive flows (MAF, IAF)
- Continuous Normalizing Flows (CNFs) and Neural ODEs[2]
- Residual flows and free-form Jacobian computations

### Module 5.4 — Diffusion Models
- Denoising Diffusion Probabilistic Models (DDPMs) and score-based generative models[30][2]
- Score matching and denoising score matching[2]
- DDIM: deterministic sampling and accelerated generation
- Latent diffusion models (Stable Diffusion) and classifier-free guidance[30]
- Consistency models and few-step generation[2]
- Design space of diffusion models: noise schedules, network architectures (U-Net, DiT)[2]
- Masked diffusion models for discrete data[2]
- Diffusion for language modeling[2]

### Module 5.5 — Flow Matching and Rectified Flows
- Continuous normalizing flows revisited: flow matching objective[2]
- Rectified flow and optimal transport connections[2]
- Stochastic interpolants and Schrödinger Bridge[2]
- Flow matching on general geometries and in discrete state spaces[2]
- Mini-batch optimal transport coupling and hierarchical rectified flow[2]

### Module 5.6 — Applications and Frontiers of Generative Models
- Text-to-image generation: DALL-E, Imagen, Stable Diffusion architecture deep-dives
- Video generation: temporal consistency, autoregressive vs. diffusion approaches
- Generative AI for molecular and protein sciences[2]
- Imaging inverse problems: diffusion posterior sampling[2]
- Probabilistic forecasting with diffusion models[2]

***

## Semester 6: Large Language Models and Fine-Tuning

This semester covers the full LLM pipeline from pre-training theory through alignment, efficient inference, and agentic systems.

### Module 6.1 — LLM Pre-Training and Architecture
- Tokenization: BPE, SentencePiece, WordPiece, and byte-level tokenization
- Pre-training objectives: next-token prediction, masked language modeling, span corruption
- Architecture decisions: decoder-only (GPT), encoder-only (BERT), encoder-decoder (T5)
- Scaling laws and compute-optimal training (Chinchilla scaling)[14]
- Emergent abilities and phase transitions in LLMs[15][16]
- Mixture of Experts (MoE): routing mechanisms, expert balancing, Switch Transformer, Mixtral, DeepSeek-V3[31][32]

### Module 6.2 — Fine-Tuning Paradigms
- Full fine-tuning vs. parameter-efficient fine-tuning (PEFT)[33][20]
- LoRA, QLoRA, DoRA, VB-LoRA, and adapter methods[34][20]
- Instruction tuning: dataset construction, formatting, and quality principles[34]
- Domain adaptation and continued pre-training[35]
- Multi-task fine-tuning and task-specific adapters

### Module 6.3 — Alignment and Preference Optimization
- Reward modeling and RLHF: PPO-based alignment pipelines[36][28]
- Direct Preference Optimization (DPO) and its variants (KTO, BCO, IPO)[37][36]
- Safety alignment as divergence estimation[37]
- Constitutional AI and RLAIF (AI-generated feedback)
- Red-teaming, jailbreak attacks, and safety benchmarks[28]
- Balancing helpfulness, harmlessness, and honesty[38]

### Module 6.4 — Reasoning, Planning, and Test-Time Compute
- Chain-of-thought prompting and self-consistency[39]
- Tree-of-Thought, Graph-of-Thought, and Forest-of-Thought reasoning[40]
- Test-time compute scaling: budget forcing, thinking-optimal scaling[41][42]
- Reasoning models: o1-style inference-time reasoning paradigm[43]
- Verification and self-correction mechanisms[39]

### Module 6.5 — Efficient Inference and Model Compression
- Quantization: GPTQ, AWQ, GGUF, INT8/INT4 methods[5]
- Knowledge distillation for LLMs[44]
- Pruning: unstructured vs. structured, optimal compression ordering (P→KD→Q)[45][44]
- KV-cache optimization: paged attention, multi-query attention, grouped-query attention[45]
- Speculative decoding and parallel inference strategies

### Module 6.6 — Retrieval-Augmented Generation and Agentic Systems
- RAG architectures: dense retrieval, HippoRAG, GraphRAG[20][39]
- Agentic AI: planning, memory, tool use, and multi-agent collaboration[46]
- Agent frameworks: ReAct, function calling, and code generation agents[46]
- Long-context processing: efficient attention for 100K+ contexts[20]
- Evaluation of LLM-based systems: benchmarks, human evaluation, LLM-as-judge

***

## Semester 7: Advanced AI Topics — GNNs, Causal ML, and MLOps

This semester covers three pillars of applied AI: graph-based learning, causal reasoning, and production ML systems.

### Module 7.1 — Graph Neural Networks: Foundations
- Graph representation: adjacency, Laplacian, spectral properties[47]
- Spectral GNNs: graph Fourier transform, ChebNet, spectral convolutions
- Spatial GNNs: message passing neural networks (MPNN), GraphSAGE, GAT, GIN[47]
- Over-smoothing, over-squashing, and expressive power of GNNs
- Graph Transformers and positional encodings for graphs

### Module 7.2 — Advanced GNN Topics
- Heterogeneous graphs, temporal graphs, and dynamic graphs[47]
- Link prediction, node classification, and graph-level tasks[47]
- Graph generation: molecular design, drug discovery[48]
- Equivariant GNNs for 3D molecular and protein structures
- GNNs for combinatorial optimization
- Graph contrastive learning and self-supervised GNNs[49]

### Module 7.3 — Causal Machine Learning
- Structural Causal Models (SCMs) and the do-calculus[49]
- Causal discovery from observational data: PC algorithm, GES, NOTEARS
- Treatment effect estimation: propensity scores, doubly robust estimators, CATE
- Instrumental variables and regression discontinuity
- Causal inference with GNNs: using learned representations as controls[50][51]
- Counterfactual reasoning and causal fairness
- Causal ML for decision-making in A/B testing and policy evaluation

### Module 7.4 — MLOps: Production Machine Learning Systems
- ML lifecycle management: experiment tracking, model versioning (MLflow, DVC, Weights & Biases)[52][53]
- CI/CD for ML: automated training pipelines, model validation gates[53]
- Model serving: REST/gRPC APIs, BentoML, TorchServe, Triton Inference Server[53]
- Containerization and orchestration: Docker, Kubernetes for ML workloads[53]
- Feature stores and data pipelines (Feast, dbt)
- Infrastructure: GPU cluster management, cost optimization, spot instances[54]

### Module 7.5 — Monitoring, Governance, and Responsible AI
- Model monitoring: data drift, concept drift, performance degradation (Evidently AI, WhyLabs)[53]
- Explainability: SHAP, LIME, attention visualization, mechanistic interpretability
- Fairness: demographic parity, equalized odds, calibration across groups
- Model governance: model cards, audit trails, reproducibility standards
- Regulatory compliance: EU AI Act, NIST AI RMF, and sector-specific requirements
- AI safety at the systems level: deployment guardrails and human-in-the-loop design[38]

***

## Semester 8: Frontier and Cross-Cutting Topics

This capstone semester fills the critical gaps in the curriculum and addresses the topics that are rapidly becoming essential for a well-rounded AI researcher or practitioner.

### Module 8.1 — Time Series Forecasting and Sequential Data
- Classical foundations: ARIMA, exponential smoothing, state-space models
- Deep learning for time series: temporal CNNs, LSTM/GRU architectures[29][18]
- Transformer-based forecasting: TFT (Temporal Fusion Transformer), Informer, Autoformer[55]
- State-space models for time series: Mamba and S4 applied to sequential data[55]
- Probabilistic forecasting: quantile regression, distributional outputs, Gaussian processes[29]
- Foundation models for time series: TimeGPT, Chronos, Lag-Llama
- Anomaly detection: VAE-based approaches, Matrix Profile for motif/discord discovery[29]
- Ensemble methods for combining forecasts[29]

### Module 8.2 — Uncertainty Quantification and Conformal Prediction
- Epistemic vs. aleatoric uncertainty and their modeling[56]
- Bayesian deep learning revisited: MC dropout, deep ensembles, SWAG
- Conformal prediction: split conformal, full conformal, and their guarantees[57][56]
- Distribution-free coverage guarantees: finite-sample validity without distributional assumptions[57]
- Conformal prediction for classification (prediction sets) and regression (prediction intervals)[58]
- Conformal methods for time series, NLP, computer vision, and RL[57]
- Calibration: temperature scaling, Platt scaling, isotonic regression[59]
- Practical pitfalls: distribution shift, efficiency-validity tradeoffs, and adaptive conformal inference[60]

### Module 8.3 — Multimodal Foundation Models
- Vision-Language Models: CLIP, LLaVA, BLIP-2, Flamingo, Janus-Pro[61][62]
- Architecture patterns: linear projection, querying transformers, gated cross-attention[61]
- Multimodal reasoning and open-vocabulary perception[62]
- Vision-Language-Action models for embodied AI[62]
- Audio-language and video-language integration
- Multimodal pre-training objectives and contrastive learning (CLIP loss)

### Module 8.4 — Self-Supervised and Contrastive Learning
- Self-prediction methods: masked image modeling (MAE, BEiT), masked language modeling[63]
- Contrastive learning: SimCLR, MoCo, BYOL, Barlow Twins, and DINO/DINOv2[63]
- Theoretical foundations: spectral graph theory perspective, augmentation-aware bounds[64][65]
- Self-supervised learning for speech, video, and scientific domains[63]
- Joint embedding predictive architectures (JEPA) and I-JEPA

### Module 8.5 — Scientific Machine Learning
- Physics-Informed Neural Networks (PINNs): loss formulation, boundary/initial conditions[66][67]
- PINNs limitations: failure modes, training challenges, and extensions[68]
- Neural operators: Fourier Neural Operator (FNO), DeepONet, Convolutional Neural Operators[48]
- Operator Transformers and graph-based neural operators[48]
- Neural ODEs and neural differential equations[68]
- Symbolic regression and equation discovery[68]
- Foundation models for PDEs and hybrid physics-ML workflows[48]
- Applications: weather/climate, computational biology, materials science[48]

### Module 8.6 — Federated Learning and Privacy-Preserving ML
- Federated learning topologies: cross-silo vs. cross-device FL[69]
- Federated averaging (FedAvg) and adaptive federated optimizers[69]
- Differential privacy for ML: privacy budgets, noise calibration, DP-SGD[70]
- Secure aggregation, homomorphic encryption, and multi-party computation[69]
- Privacy attacks: membership inference, model inversion, gradient leakage[70]
- Personalization in FL: local adaptation, meta-learning approaches[71]
- Regulatory context: GDPR, HIPAA, and privacy-by-design principles[72]

### Module 8.7 — Neuro-Symbolic AI and Knowledge Graphs
- Knowledge graph construction: entity recognition, relation extraction, graph databases[73]
- Neural reasoning over knowledge graphs: embedding methods (TransE, RotatE, ComplEx)[74]
- Combining neural and symbolic reasoning: rule learning, logical consistency, deductive inference[75][73]
- LLM + KG integration: knowledge-grounded generation, structured verification[74]
- Applications: fact-checking, healthcare reasoning, scientific discovery[75]

### Module 8.8 — Emerging Frontiers
- Test-time training and adaptive inference
- Mechanistic interpretability: circuits, features, and superposition in neural networks
- Continual and lifelong learning: catastrophic forgetting, elastic weight consolidation
- AI for code: code generation, formal verification, and program synthesis
- Synthetic data generation and its role in training
- Multi-agent LLM systems: debate, collaboration, and verification architectures
- Energy efficiency and sustainable AI: carbon-aware training, green compute

***

## Curriculum Structure Overview

| Semester | Focus Area | Key Distinguishing Topics |
|----------|-----------|--------------------------|
| 1 | Mathematical Foundations | Optimal transport, information theory, SDEs, functional analysis |
| 2 | Advanced Classical ML | Bayesian methods, kernel methods, graphical models, learning theory |
| 3 | Deep Learning Theory & Architecture | State space models, MoE, scaling laws, NTK theory |
| 4 | Deep Reinforcement Learning | Offline RL, multi-agent RL, meta-RL, RLHF bridge |
| 5 | Generative AI | Flow matching, Schrödinger Bridge, masked diffusion, molecular GenAI |
| 6 | LLMs and Fine-Tuning | Alignment (DPO/RLHF), test-time compute, MoE, agentic systems |
| 7 | GNNs, Causal ML, MLOps | Equivariant GNNs, causal discovery, production ML systems |
| 8 | Frontier Topics | Time series, conformal prediction, scientific ML, federated learning, multimodal, neuro-symbolic AI |

***

## Recommended Textbooks and Resources

| Semester | Key References |
|----------|---------------|
| 1 | Cover & Thomas — *Elements of Information Theory*; Boyd & Vandenberghe — *Convex Optimization*; Peyré & Cuturi — *Computational Optimal Transport*[7] |
| 2 | Rasmussen & Williams — *Gaussian Processes for ML*; Murphy — *Probabilistic ML*; Bishop — *Pattern Recognition and ML*[10] |
| 3 | Goodfellow, Bengio & Courville — *Deep Learning*; Gu & Dao — Mamba paper (arXiv:2312.00752)[24] |
| 4 | Sutton & Barto — *Reinforcement Learning*; Plaat — *Deep Reinforcement Learning* (Springer)[26] |
| 5 | UIUC ECE 598ZZ course materials[2]; CMU 10799 Spring 2026[76]; Fudan diffusion course[30] |
| 6 | Hugging Face LLM Course[77]; Stanford CS 224R[27]; Alignment survey (arXiv)[28] |
| 7 | Hamilton — *Graph Representation Learning*; Pearl — *Causality*; MLOps community resources[52][53] |
| 8 | Angelopoulos & Bates — *Conformal Prediction tutorial*[57]; ETH AI in the Sciences course[48]; Berkeley Neuro-Symbolic AI[75] |

***

## Design Principles

This curriculum was designed with several guiding principles:

- **Math-first**: Semester 1 ensures every subsequent topic is grounded in rigorous theory, including often-neglected areas like optimal transport and information theory that are critical for understanding modern generative models.
- **Classical before deep**: Semester 2 prevents the common pitfall of treating deep learning as a black box by building intuition through Bayesian methods, kernels, and graphical models.
- **Architecture-aware**: Semester 3 goes beyond "how to use PyTorch" to cover why architectures work, including the emerging post-Transformer landscape (Mamba, SSMs, hybrids).
- **Production-ready**: Semester 7's MLOps coverage ensures graduates can deploy, monitor, and govern models — not just train them.
- **Future-proof**: Semester 8 captures the frontier topics — conformal prediction, scientific ML, multimodal models, federated learning, and neuro-symbolic AI — that are rapidly becoming essential but are absent from most existing curricula.