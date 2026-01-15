---
title: AI Nonlinear History
date: 2024-11-03 23:10:08
description: LLM Output Token Rate
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-17-AI_Evolution]]"
categories:
  - Science
tags:
  - Boltzman_Machine
  - Hopfield
  - Ising
---


## Source

1. https://www.youtube.com/watch?v=_bqa_I5hNAo
2. Cross-Entropy loss, excellent video: https://youtu.be/KHVR587oW8I
3. 人工智能諾貝爾獎  https://mbd.baidu.com/newspage/data/videolanding?nid=sv_5737170245335408568&sourceFrom=share


## AI 的非線性歷史

AI 擁有不同的分支。我們專注於機器學習或深度學習，這是最突出的分支之一，甚至成為 AI 的同義詞。這一分支的時間線：

### 物理分支：Ising 模型

1. **1-D Ising 模型 (1925)**：Ising 模型最初由 **Lenz** 在1920年提出，他的學生 **Ising ** 在1925年解決了 1 維情況。他顯示在 1D 中，沒有相變，這意味著沒有自發磁化發生的臨界溫度。

2. **2-D Ising 模型與相變 (1944)**：在1944年，**翁薩格 Onsager**** 解決了2D Ising 模型並證明其在非零臨界溫度下確實會出現相變。這是統計物理中的一個重大結果，顯示即使是簡單模型也能展示複雜的相行為。

|                    | Ising  模型          | Hopfield  神經網絡      | Hinton BM/RBM                   |
| ------------------ | ------------------ | ------------------- | ------------------------------- |
| Bit Representation | 兩種自旋               | 二進位制                | 二進位制                            |
| Connectivity       | 鄰近作用 visible nodes | 神經元連接 visible nodes | 神經元連接, visible and hidden nodes |
| Objective          | 能量最低               | 損失函數最低              | 損失函數最低                          |
| Randomness         | Yes                | No                  | Yes                             |
| Distribution       | Boltzmann          | No                  | Data Distribution               |

### AI 分支：神經網絡中的 Ising 模型

1. **Hopfield 網絡 (1982)**：**Hopfield** 提出了基於Ising 模型原則的遞歸神經網絡，如今稱為 Hopfield 網絡。網絡的每一種配置對應於一個能量狀態，穩定模式對應於這個“能量”函數的局部最小值。該網絡通過穩定到最低能量狀態來尋找模式，類似於 Ising 模型中的基態。

2. **玻爾茲曼機器 (BM) (1985)**：**Hinton 和  Sejnowski**  引入了玻爾茲曼機器。該模型擴展了Hopfield 網絡中的想法，通過引入**隱藏單元**並運用**玻爾茲曼統計學**來進行擴展。玻爾茲曼機器使用基於熱力學的概率方法，使其能夠通過類似熱波動的過程探索狀態空間。

3. **限制玻爾茲曼機器 (RBM)**：Hinton後來（2000年）對BM進行了改進，使其更適合訓練和實際應用。RBM 的效率最終導致了 **深度信念網絡（DBN）** 的發展，並奠定了促成2000年代深度學習復興的基礎思想。

<img src="/media/image-20240817101556.png" alt="20240817101556" style="zoom:60%;" />

**Hopfield 和 Hinton 共同獲得2024年諾貝爾物理獎。**

**Hopfield （所有可見節點，全連接）-> Hinton BM（可見 + 隱藏節點，全連接）-> Hinton RBM（可見 + 隱藏節點，Bipartite 連接）**

玻爾茲曼機器（BM）和限制玻爾茲曼機器（RBM）之間的關鍵區別在於它們節點之間的連接性，這影響它們的計算效率和學習能力。
### 1. 波茲曼機（BM）

- **結構**：在標準的波茲曼機中，每個節點（神經元）可以與其他所有節點相連，這意味著它是一個完全連接的無向圖。
- **節點類型**：波茲曼機可以有兩種類型的節點：**可見節點**（表示輸入）和**隱藏節點**（捕捉潛在特徵），儘管有些波茲曼機不包括隱藏節點。
- **訓練難度**：由於連接不受限制，訓練波茲曼機的計算開銷很大。每個節點的激活依賴於所有其他節點，導致複雜的依賴關係，使得計算概率和進行推斷變得緩慢。
- **學習算法**：波茲曼機通常使用**Gibbs 抽樣**或**對比散度**，但由於連接不受限制，這些方法在實際應用中可能效率低下。

### 2. 限制波茲曼機（RBM）

- **結構**：限制波茲曼機引入了一種限制：同一層內部之間沒有連接。相反，**可見節點**僅與**隱藏節點**相連，反之亦然，形成一個二分圖無向圖。
- **節點類型**：限制波茲曼機同樣有可見和隱藏節點，但與完整的波茲曼機不同，隱藏節點不與其他隱藏節點相連，可見節點也不與其他可見節點相連。
- **訓練效率**：連接的限制簡化了模型，使得計算狀態概率和進行推斷變得更容易。這使得限制波茲曼機的訓練速度顯著提高且效率更高，通常使用**對比散度**來近似梯度。
- **應用領域**：因為其高效性，限制波茲曼機更常用於特徵提取、降維以及生成模型（如深度信念網絡）等應用。

### 主要差異總結

| 特徵         | 波茲曼機（BM）   | 限制波茲曼機（RBM）      |
| ---------- | ---------- | ---------------- |
| **連接方式**   | 完全連接（所有節點） | 限制：可見 ↔ 隱藏僅限於二分圖 |
| **訓練複雜性**  | 高（低效）      | 低（高效）            |
| **推斷方法**   | Gibbs 抽樣   | 對比散度             |
| **常見應用領域** | 由於複雜性而較少使用 | 特徵提取、生成建模        |

RBMs中的這種「限制」簡化了訓練和應用，使其在現實世界任務中更加實用，也促使其相對於傳統波茲曼機獲得了更大的普及。

**Hopfield （確定性） ->  Hinton BM/RBM（隨機性）**

## Hinton 如何將隨機性加入玻爾茲曼機器（BM）？

Geoffrey Hinton 透過引入 **神經元激活的隨機性**，使 **玻爾茲曼機器（BM）** 增加了 **隨機性**。這個隨機元素使得 BM 能夠探索更廣泛的狀態，避免卡在局部最小值中，更有效地找到表示有用模式或特徵的低能狀態。以下是其運作方式：

1. **概率激活**：
   - 在玻爾茲曼機器中，每個神經元（或單元）的激活是基於 **概率規則** 而非確定性規則。單元 $i$ 被「開啟」（激活）的概率由其淨輸入的 **sigmoid 函數** 給出：
    $$
     P(v_i = 1) = \sigma\left(\frac{\sum_j w_{ij} v_j}{T}\right) = \frac{1}{1 + e^{-\frac{\sum_j w_{ij} v_j}{T}}}
    $$
   其中：
   - $w_{ij}$ 是單元 $i$ 和 $j$ 之間的權重，
   - $v_j$ 是單元 $j$ 的狀態，
   - $T$ 是一個稱為 **溫度** 的參數（來自統計力學），控制激活中的隨機性水平。

2. **玻爾茲曼分佈**：
   - 這種概率激活遵循 **玻爾茲曼分佈**，該分佈對較低能量狀態賦予更高的概率，同時仍然允許偶爾探索較高能量狀態。系統處於特定狀態 $s$ 的概率由以下公式給出：
    $$
     P(s) = \frac{e^{-\frac{E(s)}{T}}}{Z}
    $$
   其中：
   - $E(s)$ 是狀態 $s$ 的能量，
   - $T$ 是溫度，控制隨機性，
   - $Z$ 是配分函數，用於正規化概率。

3. **溫度參數**：
   - **溫度 $T$** 控制隨機性的水平：
     - 在 **高溫下**，單元更可能隨機翻轉，導致更大的探索。
     - 在 **低溫下**，網絡更專注於低能量狀態，減少隨機波動並穩定在特定配置中。
   - 可以使用退火（逐漸降低溫度）幫助網絡安定在較低能量狀態中，類似於優化中的模擬退火。

4. **Gibbs 取樣**：
   - 訓練 BM 通常涉及到 **Gibbs 取樣**，這是一種隨機過程，其中每個單元根據概率規則進行迭代更新。這使得網絡可以從能量景觀中取樣並學習數據的潛在分佈。

### 為什麼隨機性很重要

BM 的隨機特性使其能夠透過偶爾「跳躍」到較高能量狀態來克服局部最小值，這可能會導致它們發現更多全局最小值或較低能量配置，以代表更好的解決方案。這種隨機方法是玻爾茲曼機器與確定性神經網絡之間的區別所在，使其能夠捕捉數據中的複雜概率分佈。


## Appendix A: English Version

## Nonlinear History of AI

AI has different branches.  We focus on machine learning or deep learning, one of the most prominant branch or even becomes the synonym of AI.

The timeline of this branch:

### Physics Branch: Ising  Model

1. **1-D Ising  Model (1925)**: The Ising  model was first developed by **Wilhelm Lenz** in 1920, with his student **Ernst Ising ** solving the 1-dimensional case in 1925. He showed that in 1D, there is no phase transition, meaning there’s no critical temperature at which spontaneous magnetization occurs.
    
2. **2-D Ising  Model with Phase Transition (1944)**: In 1944, **Lars Onsager** solved the 2D Ising  model and demonstrated that it does exhibit a phase transition at a non-zero critical temperature. This was a major result in statistical physics, showing that even simple models could demonstrate complex phase behavior.

|                    | Ising  模型          | Hopfield  神經網絡      | Hinton BM/RBM                   |
| ------------------ | ------------------ | ------------------- | ------------------------------- |
| Bit Representation | 兩種自旋               | 二進位制                | 二進位制                            |
| Connectivity       | 鄰近作用 visible nodes | 神經元連接 visible nodes | 神經元連接, visible and hidden nodes |
| Objective          | 能量最低               | 損失函數最低              | 損失函數最低                          |
| Randomness         | Yes                | No                  | Yes                             |
| Distribution       | Boltzmann          | No                  | Data Distribution               |

### AI Branch: Ising  Model in Neural Networks

1. **Hopfield  Networks (1982)**: **John Hopfield** introduced a recurrent neural network, now known as the Hopfield  network, which is based on the principles of the Ising  model. Each configuration of the network corresponds to an energy state, with stable patterns corresponding to local minima of this "energy" function. The network finds patterns by settling into the **lowest energy states, similar to the ground states in the Ising  model.**
    
2. **Boltzmann Machine, BM (1985)**: **Geoffrey Hinton and Terry Sejnowski** introduced the Boltzmann machine. This model expanded on the ideas in Hopfield  networks by incorporating **hidden units** and employing **Boltzmann statistics**. The Boltzmann machine uses a probabilistic approach based on thermodynamics, allowing it to explore the state space using a process resembling thermal fluctuations.
    
3. **Restricted Boltzmann Machine, RBM:** Hinton later (2000) refined BM with the RBM to make it more practical for training and real-world applications. The RBM’s efficiency eventually led to the development of **Deep Belief Networks (DBNs)** and laid foundational ideas that contributed to the deep learning renaissance in the 2000s.


<img src="/media/image-20240817101556.png" alt="20240817101556" style="zoom:60%;" />

**Hopfield  and Hinton shared the 2024 Nobel Physics Prize.**

**Hopfield  (all visible nodes, fully connected) -> Hinton BM (visible + hidden nodes, fully connected) -> Hinton RBM (visible + hidden nodes, bipartite connected)**

The key difference between a **Boltzmann Machine** (BM) and a **Restricted Boltzmann Machine** (RBM) lies in the connectivity of their nodes, which affects both their computational efficiency and learning capabilities.


### 1. Boltzmann Machine (BM)

- **Structure**: In a standard Boltzmann Machine, every node (neuron) can be connected to every other node, meaning it is a fully connected, undirected graph.
- **Types of Nodes**: BMs can have two types of nodes: **visible nodes** (representing the input) and **hidden nodes** (capturing latent features), though some BMs don’t include hidden nodes.
- **Training Difficulty**: Because of the unrestricted connections, training BMs is computationally expensive. Every node's activation depends on all other nodes, leading to complex dependencies that make calculating probabilities and performing inference slow.
- **Learning Algorithm**: BMs often use **Gibbs sampling** or **contrastive divergence**, but due to the unrestricted connections, these methods can be inefficient for practical use.

### 2. Restricted Boltzmann Machine (RBM)

- **Structure**: RBMs introduce a restriction: there are no connections between nodes within the same layer. Instead, **visible nodes** only connect to **hidden nodes** and vice versa, resulting in a bipartite, undirected graph.
- **Types of Nodes**: RBMs also have visible and hidden nodes, but unlike in a full BM, hidden nodes are not connected to other hidden nodes, and visible nodes are not connected to other visible nodes.
- **Training Efficiency**: The restriction in connections simplifies the model, making it easier to compute the probability of the states and perform inference. This makes RBMs significantly faster and more efficient to train, often using **contrastive divergence** to approximate the gradient.
- **Applications**: Because of their efficiency, RBMs are more commonly used in applications like feature extraction, dimensionality reduction, and generative models (such as in Deep Belief Networks).

### Summary of Key Differences

| Feature                 | Boltzmann Machine (BM)        | Restricted Boltzmann Machine (RBM)                     |
| ----------------------- | ----------------------------- | ------------------------------------------------------ |
| **Connections**         | Fully connected (all nodes)   | Restricted: visible ↔ hidden only, **bipartite graph** |
| **Training Complexity** | High (inefficient)            | Lower (efficient)                                      |
| **Inference Method**    | Gibbs sampling                | Contrastive divergence                                 |
| **Common Applications** | Rarely used due to complexity | Feature extraction, generative modeling                |
|                         |                               |                                                        |

The **restriction** in RBMs simplifies both training and application, making them more practical for real-world tasks and leading to their popularity over traditional Boltzmann Machines.

**Hopfield  (deterministic) -> Hinton BM/RBM (stochasticity)**


## How does Hinton adds stochasticity into the BM?

Geoffrey Hinton added **stochasticity** to the **Boltzmann Machine (BM)** by introducing **randomness in the activation of neurons**. This stochastic element allows the BM to explore a broader range of states, avoiding getting stuck in local minima and more effectively finding low-energy states that represent useful patterns or features. Here’s how it works:

1. **Probabilistic Activation**: 
   - In a Boltzmann Machine, each neuron (or unit) is activated based on a **probabilistic rule** rather than a deterministic one. The probability that a unit$i$will be "on" (active) is given by a **sigmoid function** of its net input:
    $$
     P(v_i = 1) = \sigma\left(\frac{\sum_j w_{ij} v_j}{T}\right) = \frac{1}{1 + e^{-\frac{\sum_j w_{ij} v_j}{T}}}
    $$
   where:
   -$w_{ij}$is the weight between units$i$and$j$,
   -$v_j$is the state of unit$j$,
   -$T$is a parameter called the **temperature** (from statistical mechanics), which controls the level of randomness in the activations.

2. **Boltzmann Distribution**:
   - This probabilistic activation follows the **Boltzmann distribution**, which assigns higher probabilities to states with lower energy, while still allowing exploration of higher-energy states occasionally. The probability of the system being in a particular state$s$is given by:
    $$
     P(s) = \frac{e^{-\frac{E(s)}{T}}}{Z}
    $$
   where:
   -$E(s)$is the energy of state$s$,
   -$T$is the temperature, controlling randomness,
   -$Z$is the partition function, normalizing the probabilities.

3. **Temperature Parameter**:
   - The **temperature$T$** controls the level of stochasticity:
     - At **high temperatures**, units are more likely to flip randomly, leading to greater exploration.
     - At **low temperatures**, the network focuses more on low-energy states, reducing random fluctuations and stabilizing in a particular configuration.
   - Annealing (gradually lowering the temperature) can be used to help the network settle into lower-energy states, similar to simulated annealing in optimization.

4. **Gibbs Sampling**:
   - Training a BM typically involves **Gibbs sampling**, a stochastic process where each unit is iteratively updated based on the probabilistic rule. This allows the network to sample from the energy landscape and learn the underlying distribution of the data.

### Why Stochasticity Matters
The stochastic nature of BMs allows them to overcome local minima by occasionally "jumping" to higher-energy states, which may lead them to discover more global minima, or lower-energy configurations, representing better solutions. This stochastic approach is what differentiates Boltzmann Machines from deterministic neural networks and enables them to capture complex probability distributions in data.

### How Probabilistic Activation and Boltzmann Distribution Connect

- The **probabilistic activation of individual neurons** enables the network to sample from different configurations. **When the network is trained, the collective probabilistic activations approximate the Boltzmann distribution over the network’s states.**
- Together, these probabilistic activations allow the network to explore various states and **converge towards a distribution over states that matches the training data's distribution (or the underlying patterns it captures).**


### In summary:

- **Probabilistic Activation** is about the stochastic behavior at the **neuron level**.
- **Boltzmann Distribution** describes the **overall system's behavior** in terms of state probabilities, based on energy levels.

In a Boltzmann Machine, as long as each neuron's probabilistic activation is governed by a specific activation rule (usually the sigmoid function based on its input and temperature), the **Boltzmann distribution over the network's states** is automatically derived as the system evolves.

Here's how it works in detail:

1. **Neuron-Level Probabilistic Activation**:
    
    - Each neuron independently decides its state based on a probability function, typically a sigmoid, which depends on its local inputs and a temperature parameter.
    - This stochastic activation allows neurons to randomly turn on or off in a way that is biased by their net input (the weighted sum of inputs from connected neurons).
    
1. **Emergence of the Boltzmann Distribution**:
    
    - When all neurons follow this probabilistic activation rule, the entire network begins to explore different states (configurations of all neurons) in a way that aligns with **thermodynamic principles**.
    - As the network iterates through these states (through processes like **Gibbs sampling**), it starts to **approximate the Boltzmann distribution** over its states.
    - The network will naturally spend more time in lower-energy states, thus higher-probability configurations, because lower-energy states are more likely under the Boltzmann distribution.
    
1. **Training and Equilibrium**:
    
    - During training, the network adjusts its weights so that the probability of different states under the Boltzmann distribution matches the probability of those configurations in the training data.
    - When the system reaches **equilibrium** (where the probabilities are stable), it has effectively learned a Boltzmann distribution that approximates the underlying data distribution.

**Boltzmann distribution** in the context of a Boltzmann Machine, it refers to the **probability distribution over all possible states of the entire network**, not just individual 000 and 111 states of each neuron.

Here's what this means in detail:

1. **Total Number of States**:
    
    - Each possible configuration (or state) of the network, consisting of a unique combination of 000s and 111s for all neurons, is considered a distinct **state** of the system.
    - If there are NNN neurons, there are 2N2^N2N possible configurations (states) in total, and the Boltzmann distribution describes the probability of the system being in any one of these configurations.
2. **Boltzmann Distribution over Configurations**:
    
    - The Boltzmann distribution gives the **probability of each entire configuration (state) based on its energy**. Lower-energy configurations are more probable, while higher-energy configurations are less probable, according to: P(s)=e−E(s)TZP(s) = \frac{e^{-\frac{E(s)}{T}}}{Z}P(s)=Ze−TE(s)​​
    - Here, P(s)P(s)P(s) is the probability of the system being in configuration sss, E(s)E(s)E(s) is the energy of configuration sss, TTT is the temperature, and ZZZ (the partition function) normalizes the probabilities across all 2N2^N2N states.
3. **Configuration vs. Individual Neuron States**:
    
    - The Boltzmann distribution does not apply to the individual states of neurons (0 or 1 for each neuron in isolation); instead, it applies to the **entire configurations of the network**. Each configuration has a specific energy level, and the Boltzmann distribution tells us the probability of the network being in any one of these configurations.
    - In other words, the Boltzmann distribution is over all possible **global states** (complete configurations of all neurons), not just the binary states of individual neurons.
4. **Sampling and Inference**:
    
    - During training, Boltzmann Machines use techniques like **Gibbs sampling** to approximate this distribution. By sampling from the Boltzmann distribution, the model can learn to represent patterns in the data, as it captures the likelihood of different configurations based on their energy levels.