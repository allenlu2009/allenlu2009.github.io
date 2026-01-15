---
title: Math AI - Expand Score Matching to Flow Matching
date: 2025-05-15 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Diffusion
  - Fisher
  - Flow-Model
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---






# Flow Matching 與 Score Matching 的理論基礎比較

Flow matching（FM）與 score matching（SM）是兩種生成模型方法，理論上密切相關但目標不同。在 **score-based（擴散）模型** 中，我們學習一個 *score function*（機率密度的梯度），或等效地學習一個 *noise predictor*，透過最小化一個去雜訊的均方誤差（MSE）損失。例如，在 DDPM 類型的模型中，目標是最小化：

$$
\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]
$$

以預測加上去的高斯噪聲 。

相較之下，**flow matching** 直接對一個預先定義的傳輸路徑中的 *velocity field* $v(x,t)$ 進行回歸。具體而言，FM 定義一個從雜訊到資料的條件分布族 $p_t(x\mid x_1)$（如高斯內插），然後導出該條件分布對應的真實 velocity field，再訓練一個網路 $f_\theta(x,t)$ 來擬合它。FM 的損失為：

$$
\mathcal{L}_{FM} = \mathbb{E}_{t,x\sim p_t}[\,\|f_\theta(x,t) - v(x,t)\|^2\,]
$$

這使得 $f_\theta$ 能精確擬合真實 flow 。

總結來說，score matching 是訓練一個對 diffusion 過程中的噪聲或 score 進行預測的模型，而 flow matching 是訓練一個描述資料傳輸 ODE 的向量場模型。FM 可以被視為對 diffusion 模型的一般化形式：如果選擇 diffusion-style 的高斯分布路徑，FM 模型可以模擬 score matching 效果，但通常更穩定  。

需要注意的還有 **時間方向差異**：score-based 模型通常從乾淨資料（t=0）逐步加噪聲到雜訊（t=1），並在生成時反向。而 FM 模型則多為雜訊到資料的方向（0→1），透過 ODE 積分前向產生資料 。

---

## 實作差異

* **模型結構與維度要求**：score matching 模型運作於資料空間中，不要求可逆性；而 flow matching 通常實作為 continuous normalizing flow（CNF），因此要求模型是可逆的，且輸入輸出維度一致。這使得 FM 更自然地應用於連續資料上，而 SM 可以更靈活地擴展到潛空間甚至離散資料。

* **訓練穩定性**：兩者都使用簡單的 MSE 損失函數，不需要模擬 forward/reverse 流程。在實作上，FM 常被認為比 score-based diffusion 模型更穩定，尤其是當選用 diffusion path 進行 flow matching 訓練時 。

* **最佳化與成本**：SM 與 FM 每個訓練步驟通常只需一個 forward 和 backward pass，因此每次迭代的成本相當。但傳統 CNF 的最大似然訓練需要昂貴的 ODE 求解，而 FM 可免去此成本。SM 本身也不需 ODE 求解，這使得兩者在訓練時成本相近，但 FM 更具效率。

* **生成速度**：最大差異在於抽樣速度。score-based 模型（如 DDPM）需經過數百或數千步反向推理，而 FM 通常只需少數 ODE 積分步驟。Liu 等人提出的 rectified flow 僅用一個 Euler step 就能生成高品質影像 ，FM 的效率優勢明顯。

* **高維資料的效率**：在高維資料（如影像）中，diffusion 模型需大量噪聲更新步驟，而 flow matching（特別是設計良好的 optimal transport path）可透過直線式流動（shorter paths）大幅減少所需的神經網路評估次數（NFEs）  。

---

## Flow Matching 與 Score Matching 可以合併嗎？

可以，且目前已有理論與實務上的合併方向。許多文獻指出，當 flow matching 使用高斯路徑時，它和 diffusion 模型在數學上是等價的 。FM 與 SM 其實是在不同框架下參數化相同的生成過程；因此，也可以交叉使用彼此的技巧。例如可以使用 diffusion path 進行 flow matching 訓練，並透過 deterministic ODE 或 stochastic SDE 進行抽樣。

此外，像 *rectified flow* 就是明確將 diffusion 轉化為 flow matching 框架的一種方式 。這些研究說明 FM 與 SM 並非競爭，而是可以互補與轉換的框架。

---

## Flow Matching 可否應用於 VAE 的 Latent Space？

可以。由於 flow matching 需處理高維輸入，因此許多研究將其應用在低維的潛空間（latent space）中，特別是與 VAE 搭配。

例如 **Dao 等人（2023）** 提出 *“Flow Matching in Latent Space”*：先訓練一個 VAE 以將資料編碼成潛空間，再在潛空間中使用 FM 進行生成建模。這種方法能有效降低計算成本，且維持高品質的圖像生成 。

同樣地，**Samaddar 等人（2025）** 的 *Latent-CFM* 利用預訓練模型提取的潛在特徵進行 flow matching。他們指出，在潛空間中訓練 FM 不僅加快訓練速度，還能提升生成品質 。這些 latent flow 模型也可與條件資訊結合（如分類標籤、inpainting、語義標籤）來實現條件生成。

與此類似的還有 diffusion 模型中的 latent 變體，例如 NVidia 的 *Latent Score-based Generative Model (LSGM)*，將 score matching 訓練在 VAE 的潛空間中以加速生成 。這些工作都顯示，將 flow matching 應用於潛空間是可行且有效的策略。

---

## 應用與結合模型實例

Flow matching 與 score matching 均已應用於各種生成與下游任務，其中包括：

* **無條件與有條件生成**：兩種方法皆能實現高品質的圖像、音訊等資料生成。例如，Lipman 等人基於 FM 的模型在 ImageNet 上達到 SOTA 表現 。Dao 的 latent FM 模型也支援高解析度條件生成（如 inpainting） 。SM 模型如 latent diffusion 也應用於文本到圖像生成（如 Stable Diffusion）。

* **表徵學習（Representation Learning）**：VAE 所學習的潛空間可與 FM 結合，透過在潛空間中訓練 flow，可學得有用的資料表徵，並進行如分類、生成、資料操控等下游任務。

* **半監督學習**：雖然目前使用 flow matching 進行半監督學習的工作較少，但過去已有 flow-based 模型（如 FlowGMM）用於建模資料與標籤的聯合分布，可望結合 FM 進一步擴展。

* **應用於物理、語言與 3D 資料**：FM 已延伸應用於影片生成、分子構象預測、離散資料建模等領域。例如 Polyak 等人將 FM 用於 foundation video models ，Hassan 等人應用於分子生成 。

* **結合 VAE 與 Flow Matching 的模型**：除了上述 latent FM 外，還有如 *Variational Rectified Flow Matching*（Guo & Schwing, 2025）結合 latent mixture 與 rectified flow，以及 *FlowLLM*（Sriram 等人, 2024）用 flow matching 微調 LLM 的輸出。這些皆顯示 FM 可與潛變數模型靈活結合，形成高效生成器。

---




From 8-Gaussian to moons.

![[Pasted image 20250522235631.png]]

![[Pasted image 20250515212250.png]]

**Flow process:** 
- **物理意義:  就是讓 noise 移動變成 image.
- Training:
	- Input: scale input + noise, t
	- Output: vector field
	- $x_n = x_{n-1} + v_t(x_t, \theta)$


```python
# Flow sampling
dt = T / N
x = x0  # initial sample from prior

for i in range(N):
    t = i * dt
    v = model(x, t)  # predict vector field at (x, t)
    x = x + v * dt  # Euler integration step

```


**Diffusion process:** **物理意義: 就是 denoise 變成 image.**
- Training:
	- Input: scale input + noise, t
	- Output: noise $\epsilon_{\theta}$
- Sampling:
	- Start from noise N(0, I)
	- $x_n = k x_{n-1} - (1-k) \epsilon_{\theta}$

For each timestep $t$, you use the reverse process:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

Where:

* $\epsilon_\theta(x_t, t)$: predicted noise
* $\alpha_t$, $\bar{\alpha}_t$: noise schedule values
* $\sigma_t$: standard deviation of the noise added at step $t$
* $z \sim \mathcal{N}(0, I)$: noise (used for stochasticity)

---

### 🧾 **DDPM Sampling Algorithm (Step-by-Step)**

```python
x = torch.randn(batch_size, channels, height, width)  # Start with Gaussian noise

for t in reversed(range(1, T + 1)):  # T to 1
    z = torch.randn_like(x) if t > 1 else 0  # no noise added at final step
    eps = model(x, t)  # Predict the noise

    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]
    sigma_t = compute_sigma(t)  # depends on schedule

    x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps) + sigma_t * z
```

---

## Combine Flow Matching and Score Matching

考慮 g(t) = 0, 可以 degenerate 得到 ODE.  這是 1st order 的 flow?

![[Pasted image 20250530185251.png]]

再加上 2nd order score stochastic function

![[Pasted image 20250530185557.png]]

這是 objective function.  基本是兩個 neural networks?
![[Pasted image 20250530190205.png]]


## Takeaway

**Diffusion process:** 
- **物理意義/現象是從頭到脚的 stochastic SDE**
- 經由對應的 Fokker-Planck ODE equation 可以得到每個時間的 $p_t(x)$, 以及 forward/reverse dynamics.   
- 其中 deterministic score function (vector field) $\nabla_{x} \log p_t(x)$ 扮演關鍵的角色。發動者是 forward path 加 noise; 關鍵是 reverse path score function 的引導者。
- **訓練 neural network 近似 score function:  score matching!**   
- Sampling 時則是利用 random sampling from initial distribution -> Langevin dynamics + denoiser (score function) 產生 samples
- Likelihood: 還是要經過 Fokker-Planc ODE 計算出.

**Flow process:** 
- **物理意義/現象是 deterministic ODE + random initial condition/distribution** 
- 物理順序是: vector field $u_t$ + initial condition 產生-> trajectory $X_t$ 產生-> flow $\phi_t(x_t)$ 產生-> distribution $p_t(x_t)$
- Flow equation 是非常簡單的 ODE:  $X_0 = x_0, \frac{d X_t}{dt} = u_t(X_t)$,  $X_t$ 是每一個 **trajectory** 對應不同的 initial condition $x_0$. 
- 把所有的 initial condition trajectories 集合起來就是 flow: $\phi_t(X_t)$，當然也滿足 $\frac{d \phi_t(X_t)}{dt} = u_t(\phi_t(X_t))$ 
- **訓練 neural network 近似 vector fild $u_t$ (是 flow 的微分):  flow matching!** 
- Sampling 時則是利用 **random sample initial distribution** ->  flow ODE 產生 samples
- Likelihood: 基本經過 reversed flow 可以計算出.


Flow matching is simulation-free (不用 ODE solver) in training! but need ODE solver in sampling!
Consistency mode is simulation-free in sampling!

## Flow matching vs. Score Matching

我用 ChatGPT 做了一個比較：

| **Aspect**                 | **Score Matching (e.g., Diffusion Models)**                             | **Flow Matching**                                                                           |
| -------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Training Objective**     | Match score ∇ₓ log pₜ(x) using denoising or noise-perturbed samples.    | Match vector field uₜ(x) using interpolated paths between base and target.                  |
| **Training Data Pairs**    | Uses perturbations of **single samples** x ∼ q(x).                      | Uses **sample pairs** (x₀, x₁) ∼ p₀ × q.                                                    |
| **Stochasticity**          | Involves stochastic reverse-time SDE or ODE.                            | Can be trained and used deterministically (via ODE).                                        |
| **Interpretability**       | Score function is less interpretable.                                   | Vector field can be directly visualized as transport directions.                            |
| **Sample Quality**         | High quality, especially for complex distributions.                     | May suffer in multi-modal or complex geometries due to poor interpolation.                  |
| **Interpolation Issues**   | None — doesn't interpolate between mismatched samples.                  | **Yes** — interpolates between randomly sampled pairs, which may be semantically unrelated. |
| **Manifold Respect**       | ✅ Yes — local noise keeps dynamics near the data manifold.              | ❌ No — interpolations may go off-manifold, especially in high dimensions.                   |
| **Simulation-Free**        | No — often requires simulation of forward diffusion process.            | **Yes** — training can be done without simulation of trajectories.                          |
| **Training Stability**     | Stable and well-understood, but computationally intensive.              | Often more efficient, but may suffer from spurious vector fields.                           |
| **Mode Coverage**          | Good mode coverage due to noise-based training.                         | Can miss modes if interpolation crosses low-density areas.                                  |
| **Theoretical Guarantees** | Strong links to Fokker-Planck, optimal denoising, and score estimation. | Tied to optimal transport and flow ODEs.                                                    |
| **Applications**           | Denoising diffusion, image synthesis, audio generation.                 | Neural ODEs, flow-based generative modeling, fast training.                                 |
Add how to do sampling for both score matching and flow matching
- Score matching - Annealed Langevin dynamics based on score function denoise
- Flow matching - 利用 vector field (不是 flow) 可以得到 samples


### Summary:

- **Score Matching** is more robust in complex distributions and provides high sample quality, but can be computationally heavy.
    
- **Flow Matching** is efficient and deterministic, but can suffer from poor interpolation unless guided or regularized carefully.
    


## Appendix D: Flow Matching vs. Score Matching



---

### **What's the issue?**

In **flow matching**, you're learning a vector field $u_t(x)$ such that the flow transforms samples from a base distribution $p_0$ to a target distribution $q$, by matching trajectories defined by conditional fields (e.g., between $x_0 \sim p_0$ and $x_1 \sim q$).

But during training:

* You sample **independent** pairs $(x_0, x_1)$ from $p_0(x_0) \times q(x_1)$.
* The vector field $u_t(x)$ is trained to point from $x_0$ to $x_1$ (linearly or via some transport plan).

This means:

> You're interpolating **between randomly paired samples**, not samples that lie on a meaningful trajectory between modes.

---

### **Why is this a problem?**

1. **Mismatch in Semantics**: If $x_0$ and $x_1$ lie in different modes (e.g., digits “1” and “8”), the interpolated vector field may not follow a realistic path — it may go through "garbage" regions.

2. **Conflicting Training Signals**: In high dimensions, many training pairs induce vector directions that contradict each other in overlapping regions of space.

3. **Generalization in Between**: The learned field might look good at data points but behaves poorly **between** them — i.e., the model has no guarantee of generating valid intermediate samples during inference.

---

### **Contrast with Score-based Models**

Score-based models (e.g., diffusion models) estimate the score $\nabla_x \log p_t(x)$, which is locally defined and smooth, and avoid the "random pairing" problem — they only rely on perturbing data and denoising, not interpolating arbitrary sample pairs.

---

### **Possible Fixes or Improvements**

* **Conditional Flow Matching (CFM)**: Instead of random pairing, use conditional transport or kernel matching to align similar samples.

* **Use trajectories from known dynamics**: If you have access to optimal transport maps (e.g., displacement interpolation), training becomes much more stable.

* **Guidance during interpolation**: Add constraints or learned alignment that discourage implausible transitions.


Here's a table comparing **Score Matching** (used in diffusion models) and **Flow Matching** (used in flow-based models):

Certainly! Here's the updated comparison table with **"Manifold Respect"** moved right after **"Interpolation Issues"**:

---

| **Aspect**                 | **Score Matching (e.g., Diffusion Models)**                             | **Flow Matching**                                                                           |
| -------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Training Objective**     | Match score ∇ₓ log pₜ(x) using denoising or noise-perturbed samples.    | Match vector field uₜ(x) using interpolated paths between base and target.                  |
| **Training Data Pairs**    | Uses perturbations of **single samples** x ∼ q(x).                      | Uses **sample pairs** (x₀, x₁) ∼ p₀ × q.                                                    |
| **Stochasticity**          | Involves stochastic reverse-time SDE or ODE.                            | Can be trained and used deterministically (via ODE).                                        |
| **Interpretability**       | Score function is less interpretable.                                   | Vector field can be directly visualized as transport directions.                            |
| **Sample Quality**         | High quality, especially for complex distributions.                     | May suffer in multi-modal or complex geometries due to poor interpolation.                  |
| **Interpolation Issues**   | None — doesn't interpolate between mismatched samples.                  | **Yes** — interpolates between randomly sampled pairs, which may be semantically unrelated. |
| **Manifold Respect**       | ✅ Yes — local noise keeps dynamics near the data manifold.              | ❌ No — interpolations may go off-manifold, especially in high dimensions.                   |
| **Simulation-Free**        | No — often requires simulation of forward diffusion process.            | **Yes** — training can be done without simulation of trajectories.                          |
| **Training Stability**     | Stable and well-understood, but computationally intensive.              | Often more efficient, but may suffer from spurious vector fields.                           |
| **Mode Coverage**          | Good mode coverage due to noise-based training.                         | Can miss modes if interpolation crosses low-density areas.                                  |
| **Theoretical Guarantees** | Strong links to Fokker-Planck, optimal denoising, and score estimation. | Tied to optimal transport and flow ODEs.                                                    |
| **Applications**           | Denoising diffusion, image synthesis, audio generation.                 | Neural ODEs, flow-based generative modeling, fast training.                                 |

---

### 🌐 **Manifold Assumption Recap**

The **manifold hypothesis** states that high-dimensional data (e.g., images) actually lies on or near a **low-dimensional manifold** embedded in the ambient space (e.g., ℝⁿ).

---

### 🌀 Score Matching (Diffusion Models)

* **Preserves the manifold assumption** — at least during inference:

  * The forward diffusion process **adds noise**, pushing data **off** the manifold.
  * But during **reverse-time generation**, the score function learns to **bring samples back toward the data manifold**.
  * The model gradually denoises toward realistic-looking images on or near the original manifold.
* At each denoising step, small updates guided by the **score** field respect local structure.

✅ **Bottom line**: **Score matching implicitly respects the data manifold** by learning how to denoise back to it.

---

### 🔁 Flow Matching

* **Potentially breaks the manifold**:

  * Flow matching defines **vector fields** between a base distribution (e.g., Gaussian) and the data distribution, often through **interpolated paths**.
  * These interpolations (e.g., linear interpolations in pixel space between two image samples) often leave the manifold — they can pass through unrealistic regions that are **far from the true data manifold**.
  * The resulting vector field **pushes probability mass through low-density regions**, which is unnatural for image data.

🚨 **Problem**: If you interpolate between two real images of, say, a cat and a car, the middle of the interpolation may not correspond to any real image-like structure — it's **off-manifold**.

---

### 🧠 Summary

|                      | **Score Matching**                       | **Flow Matching**                                         |
| -------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Manifold Respect** | ✅ Mostly stays on/near the data manifold | ❌ May generate off-manifold trajectories                  |
| **Interpolation**    | Implicit via small local noise steps     | Explicit interpolation between possibly unrelated samples |
| **Risk**             | Lower — local denoising follows manifold | Higher — global flows may cross unrealistic regions       |

---

### 🧩 Final Thought

Flow matching **can respect the manifold**, *but only if interpolation is carefully chosen*. Some recent work uses **manifold-aware interpolations** (e.g., via encoders or geodesic paths in latent space) to mitigate this issue.

Would you like to see a visual example comparing on- vs. off-manifold interpolations?


## Reference

Schrodinger bridge
https://arxiv.org/pdf/2307.03672
