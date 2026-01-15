---
title: Math AI - Diffusion Acceleration Phases
date: 2025-05-10 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Diffusion
  - Fisher
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---





## 如何加速 Diffusion Process

如同宋颺在 interview 所提到，當初在發展 Diffusion SDE (continuous model) 是被 DDPM (denoising diffusion probabilistic model, discrete model) 刺激，可以統一解釋 score matching and DDPM.   

他知道 diffusion SDE 可以轉換成 ODE using Fokker-Planck equation.  不過他的目的是計算 likelihood 可以比較不同的 diffusion methods performance (NLL - negative likelihood)。很快他發現 ODE 可以加速，因為 (1) ODE solver 本來執行速度就比 SDE solver 快；（2) 宋颺另外針對 diffusion 提出 predictor (score ?) and corrector (Langevin dynamics) 的方法。
Q: (2) 是 improve quality or speed?

Diffusion ODE 後來發展成另一支，包含 Neural ODE,  CM (Consistency Model), 甚至和 Normalized Flow Model 結合成為 PF (Probability Flow), and Flow Matching, 而開枝散葉。宋颺甚至認為 CM (or PF, Flow, etc.) 這是除了Diffusion 和 AR (Auto-Regressive) 兩大 Generative AI 宗派之外的第三個宗派。（Variational Autoencoder, GAN 都有各自的問題而無法開宗立派）。


Flow Matching = Continuous Normalizing Flows + Diffusion Models

Flow match : Diffusion,   OT (Optimal Transport)



## SDE (samples) to ODE (probability flow) (DDPM to DDIM?)

Lagenvin SDE 提供的是 samples,  就是像布朗運動的 samples.   但是速度很慢。如果要加速，一個方法是轉換成 ODE (Ordinary Differential Equation),  就有種種不同的加速工具。

![[Pasted image 20250202115412.png]]


## Diffusion 加速:  ODE to CTM (Flow Matching) to Distillation (Long Jump, CTM)
Teacher model: ODE
Student model: long jump, CTM


![[Pasted image 20250202151459.png]]


### **Central Concept: PF-ODE (2021)**

At the top, the slide references **PF-ODE (Probability Flow ODE)** from 2021 as a foundation for viewing diffusion models as ODEs. 

### **1. Training-free: Sophisticated Solvers (2021–2022+)**

- **Key Idea**: **Gradient**-based iteration.   Improve sampling speed and quality **without retraining the model** by using more advanced solvers for the ODE derived from the diffusion process.

- 為什麼說是 training-free?  DDPM trained model 可以直接用於 DDIM?
    
- **Techniques Cited**:
    
    - **DDIM (ICLR 2021)**: Deterministic version of diffusion.
        
    - **DPM, DPM++ (NeurIPS 2022)**: Higher-order solvers.
        
    - **DEIS (ICLR 2023)**: More accurate integrators.
        
- **Gradient:**  trajectory (curved black line) taken using gradients, 已經比 DDPM, NCSN 快很多 （stochastic process).  但是有兩個問題：(1) inaccurate in training; (2) slow in sampling.


如何從 DDPM to DDIM? https://arxiv.org/pdf/2010.02502

似乎是再 sampling 的時候設定不同的 $\sigma_t$.  設定一回到 DDPM.   設定二 ($\sigma_t=0$) 得到 DDIM.
所以 training 的 denoiser 應該是同一個。

![[Pasted image 20250506224511.png]]



### **2. Training-based: Learning Optimal Trajectories (2023+)**

- **Key Idea**: 1-step (NO) 還是 better flow (Yes), 例如 optimal flow?  Train models specifically to learn better (faster/shorter) trajectories for sampling rather than relying on fixed solvers.
    
- **Techniques Cited**:
    
    - **Rectified Flow (ICLR 2023)**
        
    - **Flow Matching (ICLR 2023)**
        
- **Illustration**: Shows a learned smoother green trajectory that better approximates the underlying data-to-noise mapping by training on trajectory data.



### **3. Distillation from DM (2023–)**

- **Key Idea**: Use **distillation techniques** to compress the diffusion model into a simpler, faster model (e.g., one-step or few-step models).
    
- **Techniques Cited**:
    
    - **Consistency Model (ICML 2023, Song+)**
        
    - **Kim & Lai (ICLR 2024)**
        
    - **Consistency Trajectory Model**
        
- **Illustration**: Suggests that distilled models follow a straight-line (green) trajectory, enabling fast sampling in a single or few steps from noise to data.





ODE from Fokker-Planck equation?  **Or the SDE with g(t) = 0????**


ODE - 比 SDE 好。但是兩個問題
3. NN is hard to learn function with large Lipschitzness (score match when the noise is small, 接近 data manifold)

![[Pasted image 20250505233937.png]]

1. Slow, 因爲 ODE 需要 local 斜率: Gradient (Wrong!)  這裏是 time 斜率，不是 spatial gradient, score?!!  Large Lipschitzness.
![[Pasted image 20250505234053.png]]



![[Pasted image 20250505234206.png]]

FP-Difussion 解決第一個問題；CTM 解決第二個問題！

FP-Diffusion
![[Pasted image 20250505234530.png]]
![[Pasted image 20250505234930.png]]


2nd problem, use CTM (CM is from OpenAI)

![[Pasted image 20250505235526.png]]

![[Pasted image 20250506001116.png]]
This slide titled **"Fast Sampling in Diffusion Model"** provides an overview of three major approaches to accelerating sampling in diffusion models, based on the evolution of techniques and research trends over time.


![[Pasted image 20250506001233.png]]


---

**DDIM (Denoising Diffusion Implicit Models)** can use the **same trained model** as **DDPM (Denoising Diffusion Probabilistic Models)**.

---

### ✅ Why This Works:

- **DDIM and DDPM share the same noise prediction model** — typically a U-Net trained to predict the noise added to clean data during the diffusion process.
    
- DDIM changes only the **sampling process**, not the training process.
    
- Specifically, DDIM introduces a **deterministic sampling path** by modifying the reverse diffusion equation, but it **reuses the same noise prediction network** trained via the DDPM objective.
    

---

### 🔧 What DDIM Does Differently:

- DDPM: Fully stochastic reverse process; needs many steps (e.g., 1000) for good results.
    
- DDIM: Deterministic or semi-deterministic sampling using a non-Markovian formulation, allowing **much fewer steps** (e.g., 50 or even 10) with competitive quality.
    

---

### In Summary:

> **Yes**, DDIM uses the same trained model as DDPM — it only modifies how sampling is done from that model.

Would you like to see a diagram comparing DDIM and DDPM sampling steps?


### Summary

This slide categorizes fast sampling strategies in diffusion models into:

2. **Training-free solvers** (improve ODE integration).
    
3. **Training-based optimal paths** (learn better transitions).
    
4. **Distillation-based models** (compress models for fast inference).
    

Would you like a deeper explanation of any of these three categories?

## Reference

Yang Song, PPDM, ICLR 2021: https://www.youtube.com/watch?v=L9ZegT87QK8&ab_channel=ArtificialIntelligence

SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS
https://arxiv.org/pdf/2011.13456

Yang Song interview https://www.youtube.com/watch?v=ud6z5SkjoZI&t=2098s&ab_channel=BainCapitalVentures

Yang Song CM paper: https://arxiv.org/pdf/2410.11081

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)



Lai's youtube about CTM
https://www.youtube.com/watch?v=9fW8nS6Lkzo&ab_channel=%E6%B8%85%E8%8F%AF%E5%A4%A7%E5%AD%B8%E6%95%B8%E5%AD%B8%E7%B3%BBDepartmentofMathematics%2CNTHU

https://www.youtube.com/watch?v=Bp2t8IFmDGU&ab_channel=nnabla%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%83%81%E3%83%A3%E3%83%B3%E3%83%8D%E3%83%AB

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

## Appendix



