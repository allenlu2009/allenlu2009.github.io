---
title: Test or Inference Time Compute
date: 2025-01-01 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-03-26-Transformer_LLM]]"
categories:
  - Science
tags:
  - AI
  - LLM
  - Optimization
---




## Source

- Google DeepMind:  Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. https://arxiv.org/pdf/2408.03314v1
- Fudan and Shanghai AI Lab: Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective.  https://arxiv.org/pdf/2412.14135v1
- good overview of Google paper: https://www.youtube.com/watch?v=QWoslkjR9W4&ab_channel=AdamLucek 
- good overview of Fudan and Shanghai AI lab paper!  https://www.youtube.com/watch?v=-haWhgmUheA&ab_channel=PurpleMind ,  https://www.youtube.com/watch?v=LyKRUwLNPO8&ab_channel=TheAIGRID

## 引言

### Training Scaling Law

•LLM performance scales with mode and dataset during training.

•Training scaling law issues: (1) huge data, model, and compute become very expensive;  (2) additional performance gain diminish.

![[Pasted image 20250101205400.png]]

Two Issues:
- Too expensive for the compute
- Run out of fossil fuel (Data).   我比較不同意這個觀點。


### Inferencing Time Scaling Law



![[Pasted image 20250101205725.png]]

LLM performance scales with computation sources (e.g. TOPS, memory) allocated during inference.
Inferencing scaling law: (1) use same model and data, only inrease computation to process more tokens (prompt and generation); (2) it requires more TOPS and memory footprint/BW to trade-off the latency. 

#### Methodology of test-time compute
- Parallel selection: Best-of-N, Fast-and-Slow (System1/2)
- Sequential selection: In-context learning (few shots), Chain-Of-Thought (CoT), Read-Twice
- Hybrid:  Beam search, MCTS (Monte-Carlo Tree Search)



## Two Method

1. Fine-tune the model usually using reinforcement learning (RL)
2. Use existing model and inferencing only method
 