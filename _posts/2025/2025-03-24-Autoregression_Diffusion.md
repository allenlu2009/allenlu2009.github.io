---
title: Autoregression vs. Diffusion
date: 2025-03-24 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2023-02-08-GenAI_Diffusion
categories:
  - GenAI
tags:
  - Diffusion
  - GenAI
  - Graph
  - Language
  - Laplacian
  - Math-AI
---




## Generative AI 兩大流派：CV (Computer Vision) and NL (Natural Language)

AI 的兩大流派是 CV and NL，對應劍宗和氣宗。劍宗花俏，生成各式美麗圖案。特別 diffusion model generate image from noise, 頗有無招勝有招之態。氣宗則是穩扎穩打，步步為營。正如同 NL 的 auto-regression 永遠只看下一步如何產生。


# How Does AI Generate X, Article/Image/… Samples?

Analytic AI is cheap, generative AI is difficult.  A quick comparison **Analytic AI vs. Generative AI Comparison Table**

|                               | **Analytic/Discriminative AI**                                                              | **Generative AI**                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Number of Parameters**      | < 10's M<br>                                                                                | > 1000 M                                                                          |
| **Computation per Inference** | 1-10's TOPS                                                                                 | 100's - 1000's TOPS                                                               |
| **Core Neural Network**       | CNN                                                                                         | Transformer                                                                       |
| **Learning Methodology**      | Supervised training                                                                         | Self-Supervised training + Supervised fine-tuning                                 |
| **Math Insight**              | $P(c\vert \mathbf{x})$: given an image/article **x**, what’s the probability of a dog/spam? | $P(\mathbf{x})\sim X$: generate image/article sample $X$ with text/image guidance |

- **Generative AI is extremely difficult** because of **x**’s high dimensionality:
    - The probability of a Monkey typing Hamlet (30K words, or 130K letters) is $1/26^{-130000} \sim 3\times 10^{-183,945}$.
    - The probability of randomly generating a 512×512=262,144 pixels natural image is $1/(2^{24})^{512\times 512}\sim 1/16,777,216^{262,144}$.
    - Brute force 的機率是 $S^{-N}$,  $N$ 是 token/pixel length, $S$ 是每個 token/pixel 的 discrete values/states.  我們一般稱 $N$ 是這個問題的 dimension.  Monday typing Hamlet 算是 130K dimensions, 512x512 image 則是 262K dimensions.  隨著文章長度變長，或是 image size 變大，問題的 dimension 也隨之 exponentially 增加。 

  ![[Pasted image 20250324171721.png]]

- **How to obtain $P(\mathbf{x})$ to generate sample *X* (article, image, …) effectively and efficiently?**
    - **(Mainstream) Auto-regression for text**: OpenAI GPT-2, 2019.
    - **(Mainstream) Diffusion for image**: Stanford NCSN[^1]/DDIM [^2](Ermon, Song), 2019.
    - **Other (past) generation techniques:**
        - **VAE (Variational Autoencoder)**: U. of Amsterdam (Kingma, Welling), **2013**.
        - **GAN (Generative Adversarial Network)**: U. of Montreal (Goodfellow), **2014**.
        - **MAE (Masked Autoencoder)**: Meta (Kaiming He), **2021**.



- 2013, 最早的 generative AI 始於 CV VAE 用於 image generation.
- 2019, OpenAI GPT-2, 使用 AR + Transformer 於 natural language generation.
- 2019-2020, Stanford (Ermon and Song) and Berkeley (Jonathan Ho) 使用 diffusion 取代 GAN 產生 image
- 2020?,  .. use token for image patch like ViT for image understanding.  Tokenization wins!
- 

- NL camp: Attention, Transformer, Tokenization, AR
- CV camp: CNN for patchfication, VAE (use for encode/decode image), 

## **Auto-regression (generate next token) vs. Diffusion (generate from noise)**

### Auto-regression: $P(\mathbf{x})=P(x_1)P(x_2\vert x_1)...P(x_n\vert x_1,...,x_{n−1})$
上式是完全等價, 沒有做任何的簡化。複雜度仍然時 $S^N$
因為是一個一個 token (text, image) 產生，比較類似循序漸進，稱為氣宗。
#### Text generation:

- $P(\mathbf{x}) : P(動物園) = P(\text{動}) P(\text{物} \vert \text{動}) P(\text{園} \vert \text{動}, \text{物})$
#### Pros:

- Discrete or continuous distribution (text, image, video)
- Variable length (text, video)
- Scalable and also meet scaling law

#### Cons:

- **Slow – Sequential generation**
- Sampling "drifts" - accumulated errors
- Inductive bias:  對於非語言應用 (e.g. DNA sequence) 不一定從左到右
- Constrained architecture: causal attention mask

#### Models:

- Most popular LLMs and LMMs based on transformer (金剛掌), Mamba (蛇形拳), ViT


### Diffusion: $\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\epsilon}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \mathbf{z}_t$

#### Pros:

- **Fast – Parallel generation**

#### Cons:

- Continuous distribution (image, speech)
- Fixed length (image)

#### Models:

- DDPM, DDIM, Stable Diffusion (w/ Encoder/Decoder), DiT (patched tokens)

#### Image generation via SDE: Stochastic Differential Equation

- **Forward SDE ($0 \to T$):**
    - $d\mathbf{x}_t = \sigma (t)\, d\mathbf{w}_t$ 
    - $\mathbf{x}_t$ is random walk, 或稱為布朗運動。$\sigma(t)$ 是 noising/denoising scheduler，控制退火的速度。
    
- **Reverse SDE ($T \to 0$):**
    - $d\mathbf{x}_t = -\sigma(t)^{2} \underbrace{\nabla_\mathbf{x} \log p(\mathbf{x}_t)}_{\text{Score Function}} \,dt + d\mathbf{w}_t$
    
    **Score Function:**  log likelihood gradient 是一個 vector field。使用 neural network 模擬。看起來變複雜，因為 output 的 dimension 從 scalar (dimension = 1) 變成 high dimension as the input (e.g.  512x512=262K for image, or 130K for text).  不過實務上不會變更複雜，因為在 training 時，本來就會計算 (high dimension) gradient.  同時在 training 和 inferencing 變的更簡單。



AR 和 Diffusion 原來是兩個 local kings, 各自有擅長的領域 in text and image。但事情開始改變。

# **Episode 2: Auto-Regression for Image Generation**

- **AR (Auto-regression)** has unified all modalities by tokenizing text, image, audio, video, etc., and predicting the next token.
    
- **The last hurdle:** AR image generation is **(1) slow** and **(2) low quality**
    
    - **VAR (Visual AutoRegression)** provides image generation via next-scale prediction

### **Three Different Autoregressive Generative Models**

1. **Autoregressive Transformer (GPT, Llama, PaLM, etc.)**
    
    - AR: Text generation by next-token prediction
2. **AR Transformer (GPT, VQGAN, PaRI, etc.)**
    
    - AR: Image generation by next-image-token prediction
3. **Visual Autoregressive Transformer (Our VAR)**
    
    - VAR: Image generation by next-scale (not next-resolution) prediction








## Citation



[^1]: Noise Conditioned Score Network

[^2]: Denoising Diffusion Implicit Model
