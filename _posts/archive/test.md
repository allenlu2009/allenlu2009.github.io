

### Mathematical Explanation of `linear_attention`

`linear_attention` computes attention in a more efficient way by leveraging properties of softmax kernels and linear algebra. Here's the step-by-step mathematical formulation:

#### Standard Attention
The standard attention mechanism computes:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$
Where:
- $Q$ (query) has shape $(B, N, d)$,
- $K$ (key) has shape $(B, N, d)$,
- $V$ (value) has shape $(B, N, d)$,
- $B$ is the batch size,
- $N$ is the sequence length,
- $d$ is the feature dimension.

#### Linear Attention Approximation
Linear attention bypasses the quadratic complexity of the standard softmax attention by re-arranging the computation:

1. **Softmax Kernel Reformulation**:
   Rewrite the softmax operation as:
   $$
   \text{Softmax}(x) = \frac{e^x}{\sum e^x}
   $$
   This lets us decompose the denominator and numerator separately for efficiency.

2. **Linearized Softmax Attention**:
   Compute the attention as:
   $$
   \text{LinearAttention}(Q, K, V) = D^{-1}(Q \cdot (K^T \cdot V))
   $$
   Where:
   - $D$ is the normalization term defined as:
     $$
     D = Q \cdot K^T \cdot \mathbf{1}
     $$
     This sums over the keys.

3. **Steps in Linear Attention**:
   - Compute the cumulative sum of the keys:
     $$
     K_{\text{cumsum}} = \sum K
     $$
   - Normalize using $D^{-1}$:
     $$
     D^{-1} = \frac{1}{Q \cdot K_{\text{cumsum}}}
     $$
   - Calculate the context vector:
     $$
     \text{Context} = \sum K \cdot V
     $$
   - Combine to produce the output:
     $$
     \text{Output} = D^{-1} \cdot (\text{Context} \cdot Q)
     $$

#### Efficiency
The linearized version avoids explicit computation of the full $QK^T$ matrix (size $N \times N$) by computing cumulative sums and reusing intermediate results, reducing the complexity from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N d)$. 

This makes `linear_attention` particularly suited for long sequences where $N$ is large.







[Skip to main content](#__docusaurus_skipToContent_fallback)What's new in AutoGen? Read [this blog](/autogen/blog/2024/03/03/AutoGen-Update) for an overview of updates[![AutoGen](/autogen/img/ag.svg)![AutoGen](/autogen/img/ag.svg)**AutoGen**](/autogen/)[Docs](/autogen/docs/Getting-Started)[API](/autogen/docs/reference/agentchat/conversable_agent)[Blog](/autogen/blog)[FAQ](/autogen/docs/FAQ)[Examples](/autogen/docs/Examples)[Notebooks](/autogen/docs/notebooks)[Gallery](/autogen/docs/Gallery)Other Languages

* [Dotnet](https://microsoft.github.io/autogen-for-net/)
[GitHub](https://github.com/microsoft/autogen)`ctrl``K`Recent posts

* [What's New in AutoGen?](/autogen/blog/2024/03/03/AutoGen-Update)
* [StateFlow - Build LLM Workflows with Customized State-Oriented Transition Function in GroupChat](/autogen/blog/2024/02/29/StateFlow)
* [FSM Group Chat -- User-specified agent transitions](/autogen/blog/2024/02/11/FSM-GroupChat)
* [Anny: Assisting AutoGen Devs Via AutoGen](/autogen/blog/2024/02/02/AutoAnny)
* [AutoGen with Custom Models: Empowering Users to Use Their Own Inference Mechanism](/autogen/blog/2024/01/26/Custom-Models)
* [AutoGenBench -- A Tool for Measuring and Evaluating AutoGen Agents](/autogen/blog/2024/01/25/AutoGenBench)
* [Code execution is now by default inside docker container](/autogen/blog/2024/01/23/Code-execution-in-docker)
* [All About Agent Descriptions](/autogen/blog/2023/12/29/AgentDescriptions)
* [AgentOptimizer - An Agentic Way to Train Your LLM Agent](/autogen/blog/2023/12/23/AgentOptimizer)
* [AutoGen Studio: Interactively Explore Multi-Agent Workflows](/autogen/blog/2023/12/01/AutoGenStudio)
* [Agent AutoBuild - Automatically Building Multi-agent Systems](/autogen/blog/2023/11/26/Agent-AutoBuild)
* [How to Assess Utility of LLM-powered Applications?](/autogen/blog/2023/11/20/AgentEval)
* [AutoGen Meets GPTs](/autogen/blog/2023/11/13/OAI-assistants)
* [EcoAssistant - Using LLM Assistants More Accurately and Affordably](/autogen/blog/2023/11/09/EcoAssistant)
* [Multimodal with GPT-4V and LLaVA](/autogen/blog/2023/11/06/LMM-Agent)
* [AutoGen's Teachable Agents](/autogen/blog/2023/10/26/TeachableAgent)
* [Retrieval-Augmented Generation (RAG) Applications with AutoGen](/autogen/blog/2023/10/18/RetrieveChat)
* [Use AutoGen for Local LLMs](/autogen/blog/2023/07/14/Local-LLMs)
* [MathChat - An Conversational Framework to Solve Math Problems](/autogen/blog/2023/06/28/MathChat)
* [Achieve More, Pay Less - Use GPT-4 Smartly](/autogen/blog/2023/05/18/GPT-adaptive-humaneval)
* [Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH](/autogen/blog/2023/04/21/LLM-tuning-math)
# Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH

April 21, 2023 · 6 min read[![Chi Wang](https://github.com/sonichi.png)](https://www.linkedin.com/in/chi-wang-49b15b16/)[Chi Wang](https://www.linkedin.com/in/chi-wang-49b15b16/)Principal Researcher at Microsoft Research

![level 2 algebra](/autogen/assets/images/level2algebra-659ba95286432d9945fc89e84d606797.png)

**TL;DR:**

* **Just by tuning the inference parameters like model, number of responses, temperature etc. without changing any model weights or prompt, the baseline accuracy of untuned gpt-4 can be improved by 20% in high school math competition problems.**
* **For easy problems, the tuned gpt-3.5-turbo model vastly outperformed untuned gpt-4 in accuracy (e.g., 90% vs. 70%) and cost efficiency. For hard problems, the tuned gpt-4 is much more accurate (e.g., 35% vs. 20%) and less expensive than untuned gpt-4.**
* **AutoGen can help with model selection, parameter tuning, and cost-saving in LLM applications.**

Large language models (LLMs) are powerful tools that can generate natural language texts for various applications, such as chatbots, summarization, translation, and more. GPT-4 is currently the state of the art LLM in the world. Is model selection irrelevant? What about inference parameters?

In this blog post, we will explore how model and inference parameter matter in LLM applications, using a case study for [MATH](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html), a benchmark for evaluating LLMs on advanced mathematical problem solving. MATH consists of 12K math competition problems from AMC-10, AMC-12 and AIME. Each problem is accompanied by a step-by-step solution.

We will use AutoGen to automatically find the best model and inference parameter for LLMs on a given task and dataset given an inference budget, using a novel low-cost search & pruning strategy. AutoGen currently supports all the LLMs from OpenAI, such as GPT-3.5 and GPT-4.

We will use AutoGen to perform model selection and inference parameter tuning. Then we compare the performance and inference cost on solving algebra problems with the untuned gpt-4. We will also analyze how different difficulty levels affect the results.

## Experiment Setup[​](#experiment-setup "Direct link to Experiment Setup")

We use AutoGen to select between the following models with a target inference budget $0.02 per instance:

* gpt-3.5-turbo, a relatively cheap model that powers the popular ChatGPT app
* gpt-4, the state of the art LLM that costs more than 10 times of gpt-3.5-turbo

We adapt the models using 20 examples in the train set, using the problem statement as the input and generating the solution as the output. We use the following inference parameters:

* temperature: The parameter that controls the randomness of the output text. A higher temperature means more diversity but less coherence. We search for the optimal temperature in the range of [0, 1].
* top_p: The parameter that controls the probability mass of the output tokens. Only tokens with a cumulative probability less than or equal to top-p are considered. A lower top-p means more diversity but less coherence. We search for the optimal top-p in the range of [0, 1].
* max_tokens: The maximum number of tokens that can be generated for each output. We search for the optimal max length in the range of [50, 1000].
* n: The number of responses to generate. We search for the optimal n in the range of [1, 100].
* prompt: We use the template: "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \boxed{{}}." where {problem} will be replaced by the math problem instance.

In this experiment, when n > 1, we find the answer with highest votes among all the responses and then select it as the final answer to compare with the ground truth. For example, if n = 5 and 3 of the responses contain a final answer 301 while 2 of the responses contain a final answer 159, we choose 301 as the final answer. This can help with resolving potential errors due to randomness. We use the average accuracy and average inference cost as the metric to evaluate the performance over a dataset. The inference cost of a particular instance is measured by the price per 1K tokens and the number of tokens consumed.

## Experiment Results[​](#experiment-results "Direct link to Experiment Results")

The first figure in this blog post shows the average accuracy and average inference cost of each configuration on the level 2 Algebra test set.

Surprisingly, the tuned gpt-3.5-turbo model is selected as a better model and it vastly outperforms untuned gpt-4 in accuracy (92% vs. 70%) with equal or 2.5 times higher inference budget.
The same observation can be obtained on the level 3 Algebra test set.

![level 3 algebra](/autogen/assets/images/level3algebra-94e87a683ac8832ac7ae6f41f30131a4.png)

However, the selected model changes on level 4 Algebra.

![level 4 algebra](/autogen/assets/images/level4algebra-492beb22490df30d6cc258f061912dcd.png)

This time gpt-4 is selected as the best model. The tuned gpt-4 achieves much higher accuracy (56% vs. 44%) and lower cost than the untuned gpt-4.
On level 5 the result is similar.

![level 5 algebra](/autogen/assets/images/level5algebra-8fba701551334296d08580b4b489fe56.png)

We can see that AutoGen has found different optimal model and inference parameters for each subset of a particular level, which shows that these parameters matter in cost-sensitive LLM applications and need to be carefully tuned or adapted.

An example notebook to run these experiments can be found at: <https://github.com/microsoft/FLAML/blob/v1.2.1/notebook/autogen_chatgpt.ipynb>. The experiments were run when AutoGen was a subpackage in FLAML.

## Analysis and Discussion[​](#analysis-and-discussion "Direct link to Analysis and Discussion")

While gpt-3.5-turbo demonstrates competitive accuracy with voted answers in relatively easy algebra problems under the same inference budget, gpt-4 is a better choice for the most difficult problems. In general, through parameter tuning and model selection, we can identify the opportunity to save the expensive model for more challenging tasks, and improve the overall effectiveness of a budget-constrained system.

There are many other alternative ways of solving math problems, which we have not covered in this blog post. When there are choices beyond the inference parameters, they can be generally tuned via [`flaml.tune`](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function).

The need for model selection, parameter tuning and cost saving is not specific to the math problems. The [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) project is an example where high cost can easily prevent a generic complex task to be accomplished as it needs many LLM inference calls.

## For Further Reading[​](#for-further-reading "Direct link to For Further Reading")

* [Research paper about the tuning technique](https://arxiv.org/abs/2303.04673)
* [Documentation about inference tuning](/autogen/docs/Use-Cases/enhanced_inference)

*Do you have any experience to share about LLM applications? Do you like to see more support or research of LLM optimization or automation? Please join our [Discord](https://discord.gg/pAbnFJrkgZ) server for discussion.*

**Tags:**

* [LLM](/autogen/blog/tags/llm)
* [GPT](/autogen/blog/tags/gpt)
* [research](/autogen/blog/tags/research)
[Newer PostAchieve More, Pay Less - Use GPT-4 Smartly](/autogen/blog/2023/05/18/GPT-adaptive-humaneval)

* [Experiment Setup](#experiment-setup)
* [Experiment Results](#experiment-results)
* [Analysis and Discussion](#analysis-and-discussion)
* [For Further Reading](#for-further-reading)
Community

* [Discord](https://discord.gg/pAbnFJrkgZ)
* [Twitter](https://twitter.com/pyautogen)
Copyright © 2024 AutoGen Authors | [Privacy and Cookies](https://go.microsoft.com/fwlink/?LinkId=521839)





AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

Qingyun Wu , Gagan Bansal , Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Awadallah, Ryen W. White, Doug Burger, Chi Wang

# Abstract

AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic framework for building diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.

# Introduction

Large language models (LLMs) are becoming a crucial building block in developing powerful agents that utilize LLMs for reasoning, tool usage, and adapting to new observations (Yao et al., 2022; Xi et al., 2023; Wang et al., 2023b) in many real-world tasks. Given the expanding tasks that could benefit from LLMs and the growing task complexity, an intuitive approach to scale up the power of agents is to use multiple agents that cooperate. Prior work suggests that multiple agents can help encourage divergent thinking (Liang et al., 2023), improve factuality and reasoning (Du et al., 2023), and provide validation (Wu et al., 2023).

## d666f1f7-46cb-42bd-9a39-9a39cf2a509f

In light of the intuition and early evidence of promise, it is intriguing to ask the following question: how can we facilitate the development of LLM applications that could span a broad spectrum of domains and complexities based on the multi-agent approach? Our insight is to use multi-agent conversations to achieve it. There are at least three reasons confirming its general feasibility and utility thanks to recent advances in LLMs: First, because chat optimized LLMs (e.g., GPT-4) show the ability to incorporate feedback, LLM agents can cooperate through conversations with each other or human(s), e.g., a dialog where agents provide and seek reasoning, observations, critiques, and validation. Second, because a single LLM can exhibit a broad range of capabilities (especially when configured with the correct prompt and inference settings), conversations between differently configured agents can help combine these broad LLM capabilities in a modular and complementary manner. Third, LLMs have demonstrated ability to solve complex tasks when the tasks are broken into simpler subtasks. Here is a random UUID in the middle of the paragraph! 314b0a30-5b04-470b-b9f7-eed2c2bec74a Multi-agent conversations can enable this partitioning and integration in an intuitive manner. How can we leverage the above insights and support different applications with the common requirement of coordinating multiple agents, potentially backed by LLMs, humans, or tools exhibiting different capacities? We desire a multi-agent conversation framework with generic abstraction and effective implementation that has the flexibility to satisfy different application needs. Achieving this requires addressing two critical questions: (1) How can we design individual agents that are capable, reusable, customizable, and effective in multi-agent collaboration? (2) How can we develop a straightforward, unified interface that can accommodate a wide range of agent conversation patterns? In practice, applications of varying complexities may need distinct sets of agents with specific capabilities, and may require different conversation patterns, such as single- or multi-turn dialogs, different human involvement modes, and static vs. dynamic conversation. Moreover, developers may prefer the flexibility to program agent interactions in natural language or code. Failing to adequately address these two questions would limit the framework’s scope of applicability and generality.

Here is a random table for .docx parsing test purposes:

| 1   | 2   | 3                                    | 4   | 5   | 6   |
| --- | --- | ------------------------------------ | --- | --- | --- |
| 7   | 8   | 9                                    | 10  | 11  | 12  |
| 13  | 14  | 49e168b7-d2ae-407f-a055-2167576f39a1 | 15  | 16  | 17  |
| 18  | 19  | 20                                   | 21  | 22  | 23  |
| 24  | 25  | 26                                   | 27  | 28  | 29  |
|     |     |                                      |     |     |     |


嗯，我现在需要从给定的前向随机微分方程（SDE）推导出对应的反向SDE和ODE。首先，我得回忆一下相关的随机过程理论，特别是关于SDE和它们的对偶方程，比如Fokker-Planck方程，以及反向随机微分方程的概念。题目已经给出前向SDE和对应的Fokker-Planck方程，接下来需要找到反向的SDE和ODE。

首先，回顾题目中的前向SDE：

$$

d\boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) dt + g(t) d\boldsymbol{w}_t

$$

这里$\boldsymbol{w}_t$是标准的维纳过程（布朗运动），扩散系数是$g(t)$，对应的扩散项是$D(t) = \frac{g^2(t)}{2}$。对应的Fokker-Planck方程（也称为前向Kolmogorov方程）已经给出：

$$

\frac{\partial p(x,t)}{\partial t} = -\nabla \cdot [\boldsymbol{f}(x,t) p(x,t)] + \frac{g^2(t)}{2} \Delta p(x,t)

$$

这个方程描述了概率密度$p(x,t)$随时间的演化。

现在需要推导的是反向SDE和对应的ODE。这里可能需要用到时间反转的概念，或者使用Girsanov定理来进行变换。不过，反向SDE通常涉及到将时间倒流，所以可能需要调整漂移项和扩散项。让我先仔细思考一下反向SDE的一般形式。

一般来说，对于前向SDE：

$$

dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t

$$

在时间反转的情况下，比如从时间$T$倒回到0，反向SDE可能会有不同的漂移项。根据反向随机微分方程的理论，反向过程通常需要考虑当前的概率密度，或者说在已知终端条件的情况下进行推导。另外，Fokker-Planck方程在这里可能也有关联，因为它描述了概率密度的演变，而反向方程可能和这个有关。

不过，这里的前向SDE的漂移项是$\boldsymbol{f}(x,t)$，扩散项是$g(t)$，而对应的Fokker-Planck方程已经给出。现在要得到反向的SDE，可能需要通过时间反转，比如让$\tilde{t} = T - t$，然后重新表达原SDE在这个时间变量下的形式。

考虑时间反转后的过程，定义$\tilde{\boldsymbol{x}}_t = \boldsymbol{x}_{T - t}$，那么当原时间$t$从0到T时，新的时间$\tilde{t}$从T到0。这时候，原来的SDE在时间反转后，会变成怎样的形式？

根据随机微分方程在时间反转下的变换，反向过程的漂移项通常会改变符号，并且可能包含与概率密度相关的调整项。具体来说，反向过程的漂移项可能包括原漂移项加上一个由概率流或分数函数（score function）构成的项。例如，在扩散模型中，反向过程通常涉及到分数函数的梯度项。

这里可能需要用到扩散过程的时间反转公式。根据Stratonovich积分的性质，或者通过Girsanov变换调整漂移项，但可能需要更多的具体步骤。

另外，反向的Fokker-Planck方程应该满足时间反转的条件，即原来的前向方程变为反向方程，但这里可能需要调整符号。

不过，或许更直接的方法是使用Feynman-Kac公式，将Fokker-Planck方程和相应的SDE联系起来，进而推导反向方程。

不过，让我先尝试用时间反转的方法。假设我们将时间变量替换为$\tilde{t} = T - t$，那么原过程的$\boldsymbol{x}_t$在反向时间下成为$\tilde{\boldsymbol{x}}_{\tilde{t}} = \boldsymbol{x}_{T - \tilde{t}}}$。当$\tilde{t}$从0到T时，对应的原时间从T到0。

对反向过程$\tilde{\boldsymbol{x}}_{\tilde{t}}$，求其随机微分方程。根据时间反转的规则，原来的漂移项会改变符号，并且可能有一个修正项，而扩散项可能与原扩散项相同，但可能需要考虑时间反转后的维纳过程。

更正式地，根据扩散过程的时间反转定理，反向过程的SDE可以表示为：

$$

d\tilde{\boldsymbol{x}}_{\tilde{t}} = \left[ -\boldsymbol{f}(\tilde{\boldsymbol{x}}_{\tilde{t}}, T - \tilde{t}) + g(T - \tilde{t})^2 \nabla \log p(\tilde{\boldsymbol{x}}_{\tilde{t}}, T - \tilde{t}) \right] d\tilde{t} + g(T - \tilde{t}) d\tilde{\boldsymbol{w}}_{\tilde{t}}

$$

其中$\tilde{\boldsymbol{w}}_{\tilde{t}}$是反向时间下的维纳过程。不过，这里的推导是否正确呢？

这可能需要更仔细的推导。例如，根据Anderson的著作或者一些关于时间反转扩散过程的文献，反向过程的漂移项会包含原漂移项的负数，加上扩散系数平方乘以概率密度梯度的项，即分数函数。

不过，可能题目中的反向SDE应该用这种形式表达。而这里的分数函数$\nabla \log p(x,t)$来自于Fokker-Planck方程的解。因此，反向SDE的漂移项为原漂移项的负数，再加上$g(t)^2 \nabla \log p(x,t)$，而扩散项保持不变？或者说，在反向时间下，扩散项符号会有变化？

但可能更准确的是，反向过程的SDE在正向时间下写作：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

或者可能时间方向的问题需要考虑。例如，当时间反转时，维纳过程的微分会有不同的性质，但通常处理反向时间SDE时，可能需要引入逆时间的维纳过程，或者通过调整漂移项来补偿时间反转带来的变化。

这里可能需要参考反向SDE的一般形式。例如，根据扩散模型中的理论，反向过程的SDE可以写成：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - g(t)^2 \nabla \log p_t(\boldsymbol{x}_t) \right] dt + g(t) d\overline{\boldsymbol{w}}_t

$$

其中$\overline{\boldsymbol{w}}_t$是反向时间的维纳过程。不过，这里的具体符号可能取决于时间参数的设定。或者，在另一种参数化方式下，反向时间SDE的漂移项可能需要调整符号。

例如，在连续时间的情形下，假设原过程是正向时间从0到T，反向过程可以表示为从T到0，那么其SDE可能涉及对原漂移项的符号调整，以及分数项的添加。例如，根据Song Yang的论文“Score-Based Generative Modeling through Stochastic Differential Equations”，反向SDE的形式为：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - g(t)^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}_t) \right] dt + g(t) d\boldsymbol{w}_t

$$

但这里的$\boldsymbol{w}_t$可能被替换为反向时间的维纳过程，或者说在反向过程中，维纳过程的微分可能需要取负号，或者用反向的积分。不过，在形式化的SDE表达中，反向时间的维纳过程可以被视为另一个正向时间的维纳过程，但漂移项需要相应的调整。

也就是说，反向SDE的漂移项是原漂移项减去$g(t)^2$乘以分数函数（即对数概率密度的梯度），而扩散项保持不变，或者可能有符号的不同？

不过，在常规的扩散模型推导中，例如去噪扩散概率模型（DDPM）或Score SDE，反向过程通常涉及这样的调整，其中分数项的出现是为了抵消前向过程带来的噪声。

回到问题，题目中的前向SDE是：

$$

d\boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) dt + g(t) d\boldsymbol{w}_t

$$

对应的Fokker-Planck方程是：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{f} p] + D(t) \Delta p

$$

其中$D(t) = g(t)^2 / 2$。

现在需要推导反向SDE和对应的ODE。根据扩散模型中的理论，反向SDE通常的形式是：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - g(t)^2 \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

不过，这可能是在反向时间下的方程，或者说，这里的dt是反向时间的微分，因此可能需要调整符号。例如，当时间从T倒流到0时，变量替换为$\tau = T - t$，则原来的dt变为-dτ。这时候，原方程的漂移项和扩散项可能需要调整符号。

或者，更系统地进行推导，考虑时间反转后的过程。

假设我们有一个正向过程从t=0到t=T，定义反向过程为$\boldsymbol{y}_s = \boldsymbol{x}_{T - s}$，其中s从0到T。此时，当s增加时，原时间t减少。那么，我们要找到关于s的SDE。

根据Itô公式，考虑$\boldsymbol{y}_s = \boldsymbol{x}_{T - s}$，则微分dy_s为：

$$

d\boldsymbol{y}_s = d\boldsymbol{x}_{T - s} = \boldsymbol{f}(\boldsymbol{x}_{T - s}, T - s) d(T - s) + g(T - s) d\boldsymbol{w}_{T - s}

$$

但是，这里的$d(T - s) = -ds$，而$d\boldsymbol{w}_{T - s}$如何处理呢？在时间反转的情况下，维纳过程的增量会有不同的性质。事实上，反向时间的维纳过程$\boldsymbol{\tilde{w}}_s = \boldsymbol{w}_{T} - \boldsymbol{w}_{T - s}$是一个反向的维纳过程，其增量$d\boldsymbol{\tilde{w}}_s$在正向时间s下是独立同分布的，但可能和原维纳过程有不同的相关性。

不过，根据Itô积分的性质，时间反转后的维纳过程仍然是一个维纳过程，因此可以将其视为另一个维纳过程，记为$\boldsymbol{\tilde{w}}_s$。因此，原过程的微分可以重写为：

$$

d\boldsymbol{y}_s = -\boldsymbol{f}(\boldsymbol{y}_s, T - s) ds + g(T - s) d\boldsymbol{\tilde{w}}_s

$$

但这样得到的SDE似乎没有包含任何额外的项，这与之前提到的反向SDE需要包含分数函数项似乎矛盾。这说明可能我的初步推导有问题，或者需要更深入的考虑。

事实上，直接进行变量替换会导致反向过程的SDE形式似乎缺少了分数函数的修正项，这可能是由于忽略了概率密度的影响。正确的反向过程应该考虑到当前的概率流，即Fokker-Planck方程的解，从而引入额外的漂移项。

根据扩散过程的时间反转理论，反向过程的漂移项不仅包括原漂移项的负数，还需要加上一个由概率密度梯度引起的修正项。这个修正项的形式通常为$2D(t) \nabla \log p(x,t)$，即扩散系数平方除以2乘以概率密度对数的梯度，因为$D(t) = g(t)^2 / 2$，所以这里可能是$g(t)^2 \nabla \log p(x,t)$。

例如，根据反向过程的推导，正向过程的生成器是：

$$

\mathcal{L}_t = \boldsymbol{f}(x,t) \cdot \nabla + D(t) \Delta

$$

对应的Fokker-Planck方程是：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot (\boldsymbol{f} p) + D(t) \Delta p

$$

而反向过程的生成器应该满足时间反转后的动力学，即：

$$

\mathcal{\tilde{L}}_t = -\boldsymbol{f}(x,t) \cdot \nabla + D(t) \Delta + 2 D(t) \nabla \log p(x,t) \cdot \nabla

$$

这可能对应于漂移项为$-\boldsymbol{f} + 2 D(t) \nabla \log p$，而扩散项相同。不过，这个生成器对应的反向SDE应该是：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + 2 D(t) \nabla \log p(\boldsymbol{x}_t, t) \right] dt + \sqrt{2 D(t)} d\boldsymbol{w}_t

$$

但这里$D(t) = g(t)^2 / 2$，所以$2 D(t) = g(t)^2$，而$\sqrt{2 D(t)} = g(t)$，所以反向SDE的漂移项变为：

$$

-\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla \log p(\boldsymbol{x}_t, t)

$$

扩散项仍然是$g(t) d\boldsymbol{w}_t$。因此，反向SDE的形式是：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

不过，这里的时间变量可能应该以反向时间来表示，比如使用s = T - t，因此可能需要调整时间的符号。例如，在反向时间参数化下，dt会变成负的，因此反向过程的方程可能需要调整符号。

但不管怎样，反向SDE的漂移项在原漂移项的负数基础上，加上扩散系数平方乘以概率密度的梯度。这应该就是反向SDE的结构。

接下来，关于对应的ODE。在扩散模型中，除了反向SDE，还存在一个确定性的ODE，其轨迹与SDE相同，但移除了随机性。这个ODE通常被称为概率流ODE，其形式可以通过将反向SDE中的随机项去掉，并调整漂移项以保持相同的边际概率密度。

具体来说，ODE的漂移项应该等于反向SDE的漂移项减去扩散项的平方梯度的一半？或者可能需要考虑Stratonovich积分转换为ODE的形式？

或者，根据Fokker-Planck方程，如果我们有一个ODE：

$$

d\boldsymbol{x}_t = \boldsymbol{v}(\boldsymbol{x}_t, t) dt

$$

那么对应的连续性方程是：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{v} p]

$$

而Fokker-Planck方程是：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{f} p] + D(t) \Delta p

$$

因此，如果我们希望ODE的演化与SDE的Fokker-Planck方程相同，则需要满足：

$$

-\nabla \cdot [\boldsymbol{v} p] = -\nabla \cdot [\boldsymbol{f} p] + D(t) \Delta p

$$

这可以重写为：

$$

\nabla \cdot [\boldsymbol{v} p] = \nabla \cdot [\boldsymbol{f} p] - D(t) \Delta p

$$

或者：

$$

\nabla \cdot [\boldsymbol{v} p - \boldsymbol{f} p + D(t) \nabla p] = 0

$$

这意味着：

$$

\boldsymbol{v} p = \boldsymbol{f} p - D(t) \nabla p + \boldsymbol{C}

$$

其中$\boldsymbol{C}$是一个无散度的向量场。通常，如果我们假设$\boldsymbol{C}=0$，那么：

$$

\boldsymbol{v} = \boldsymbol{f} - D(t) \frac{\nabla p}{p} = \boldsymbol{f} - D(t) \nabla \log p

$$

因此，对应的ODE应为：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - D(t) \nabla \log p(\boldsymbol{x}_t, t) \right] dt

$$

或者，代入$D(t) = g(t)^2 / 2$，得到：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla \log p(\boldsymbol{x}_t, t) \right] dt

$$

但这里需要注意，这个ODE是否与反向SDE有关联。根据扩散模型的理论，概率流ODE通常与反向SDE共享相同的边缘分布，因此这里的ODE可能就是由反向SDE去掉随机项并调整漂移项得到的。

或者，根据另一种观点，当反向SDE的随机项被移除后，相应的确定性方程即为ODE。例如，在反向SDE中，如果我们使用Stratonovich积分，则对应的ODE可能会有不同的形式，但通常，在Itô积分下，ODE的漂移项可能需要调整。

不过，根据前面的推导，这里得到的ODE的漂移项是$\boldsymbol{f} - D(t) \nabla \log p$，即：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla \log p(\boldsymbol{x}_t, t) \right] dt

$$

而反向SDE的漂移项是：

$$

-\boldsymbol{f} + g(t)^2 \nabla \log p

$$

所以这里的ODE和反向SDE似乎有不同的符号。这可能需要重新考虑推导是否正确。

或者，可能我之前关于反向SDE的推导有误。例如，正确的反向SDE漂移项应该是原漂移项的负数加上扩散项的平方乘以梯度对数概率，而ODE的漂移项则可能涉及原漂移项减去扩散项平方的一半乘以梯度对数概率？

或者，可能我在这里混淆了正向和反向的时间方向。例如，在反向时间下，ODE的漂移项可能对应于反向SDE的漂移项减去扩散项的贡献？

或者，在另一篇文献中，比如Song Yang的论文中，反向SDE和对应的概率流ODE是分别定义的。例如，反向SDE的漂移项为：

$$

d\boldsymbol{x}_t = [\boldsymbol{f}(\boldsymbol{x}_t, t) - g(t)^2 \nabla \log p_t(\boldsymbol{x}_t)] dt + g(t) d\boldsymbol{w}_t

$$

而对应的概率流ODE是：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(\boldsymbol{x}_t) \right] dt

$$

但这里可能存在不同的参数化方式，需要更仔细的推导。

为了正确推导反向SDE和ODE，我需要更系统地应用时间反转定理。

根据扩散过程的时间反转定理，假设原过程满足SDE：

$$

d\boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) dt + g(t) d\boldsymbol{w}_t

$$

对应的反向过程$\boldsymbol{y}_s = \boldsymbol{x}_{T - s}$满足的SDE为：

$$

d\boldsymbol{y}_s = \left[ -\boldsymbol{f}(\boldsymbol{y}_s, T - s) + g(T - s)^2 \nabla \log p(\boldsymbol{y}_s, T - s) \right] ds + g(T - s) d\boldsymbol{w}_s

$$

这里，$\boldsymbol{w}_s$是一个新的维纳过程，与正向过程的维纳过程无关。因此，在正向时间参数化下，即令$t = T - s$，则反向过程的SDE可以重新参数化为：

当$s$从0到T时，$t$从T到0，那么用$t$代替$T - s$，则$ds = -dt$，因此反向SDE变为：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - g(t)^2 \nabla \log p(\boldsymbol{x}_t, t) \right] (-dt) + g(t) d\boldsymbol{w}_t

$$

但这似乎会导致符号问题，可能需要重新调整。

或者，更直接地，将反向SDE表示为：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{\tilde{w}}_t

$$

其中$\boldsymbol{\tilde{w}}_t$是反向时间的维纳过程，具有相同的扩散系数。这种情况下，反向SDE的漂移项是原漂移项的负数加上$g(t)^2$乘以分数函数，扩散项保持不变。

现在，关于对应的ODE，当我们需要一个确定性过程，使得其轨迹的概率分布与SDE相同，则可以通过移除扩散项并调整漂移项来实现。具体来说，ODE的漂移项应等于反向SDE的漂移项减去扩散项平方的一半乘以分数函数，或者类似的调整。

或者，根据概率流的连续性方程，ODE的漂移项应该使得：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{v} p]

$$

而原Fokker-Planck方程是：

$$

\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{f} p] + D(t) \Delta p

$$

因此，将这两个方程等同起来，得到：

$$

-\nabla \cdot [\boldsymbol{v} p] = -\nabla \cdot [\boldsymbol{f} p] + D(t) \Delta p

$$

将右边的项整理：

右边可以写成：

$$

-\nabla \cdot [\boldsymbol{f} p] + D(t) \nabla \cdot (\nabla p) = -\nabla \cdot [\boldsymbol{f} p - D(t) \nabla p]

$$

因此，对比两边，得到：

$$

\boldsymbol{v} p = \boldsymbol{f} p - D(t) \nabla p

$$

即：

$$

\boldsymbol{v} = \boldsymbol{f} - D(t) \frac{\nabla p}{p} = \boldsymbol{f} - D(t) \nabla \log p

$$

代入$D(t) = g(t)^2 / 2$，得到：

$$

\boldsymbol{v} = \boldsymbol{f} - \frac{g(t)^2}{2} \nabla \log p

$$

因此，对应的ODE是：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla \log p(\boldsymbol{x}_t, t) \right] dt

$$

这就是概率流ODE，其轨迹与原来的SDE具有相同的边缘分布$p(x,t)$。

总结一下：

反向SDE的推导需要考虑时间反转后的过程，其漂移项调整为原漂移项的负数加上扩散系数平方乘以概率密度的梯度，扩散项保持不变。对应的ODE则是通过确保概率流的连续性方程与原Fokker-Planck方程一致，得到漂移项为原漂移项减去扩散系数平方的一半乘以概率密度的梯度。

因此，最终结果应该是：

反向SDE：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

对应的ODE：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla \log p(\boldsymbol{x}_t, t) \right] dt

$$

不过，需要再次检查符号是否正确。例如，在时间反转的情况下，反向SDE的漂移项是否应该是原漂移项的负数加上分数项，而ODE的漂移项是否在原漂移项的基础上减去分数项的一半？

根据前面概率流ODE的推导，确实如此。所以，反向SDE和ODE的漂移项都涉及到分数函数$\nabla \log p(x,t)$，但系数不同。反向SDE的系数是$g(t)^2$，而ODE的系数是$\frac{g(t)^2}{2}$，同时反向SDE的漂移项是原漂移项的负数，而ODE的漂移项保持原符号。

因此，最终的答案应该是：

反向SDE：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla_x \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

对应的ODE：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla_x \log p(\boldsymbol{x}_t, t) \right] dt

$$

不过，这里是否在反向SDE中需要改变时间的参数化呢？比如，是否在反向SDE中时间应该从T到0，因此在表达式中是否需要调整？不过问题中可能并不严格要求时间的参数化方向，只需要写出方程的形式，所以可能上面的式子已经足够。

此外，需要注意，这里的反向SDE可能是在正向时间下的，即维纳过程仍然是正向的，但是漂移项被调整了。这可能需要更详细的解释，但根据扩散模型中的常规表示，反向SDE通常以这种方式写出，其中时间$t$仍然从0到T，但漂移项被调整为包含分数函数。

最后，确认一下Fokker-Planck方程是否一致。对于反向SDE，其对应的Fokker-Planck方程应该与原方程相同，但符号可能不同，因为时间方向改变。不过，当反向SDE以正向时间写出时，其Fokker-Planck方程应该与原方程相同，但此时可能概率密度是反向传播的。不过，这里可能需要更深入的验证，但由于问题已经给出了前向的Fokker-Planck方程，并且要求推导反向的SDE和ODE，因此可能只需要按照上述步骤推导即可。

**答案**

反向SDE为：

$$

d\boldsymbol{x}_t = \left[ -\boldsymbol{f}(\boldsymbol{x}_t, t) + g(t)^2 \nabla_x \log p(\boldsymbol{x}_t, t) \right] dt + g(t) d\boldsymbol{w}_t

$$

对应的ODE为：

$$

d\boldsymbol{x}_t = \left[ \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g(t)^2}{2} \nabla_x \log p(\boldsymbol{x}_t, t) \right] dt 

$$

**Lagrangian and Eulerian Views of Probability Flow in Diffusion Models**

---

## 1. Pure Diffusion

We consider the stochastic differential equation (SDE) in the pure diffusion case:

$$
d\boldsymbol{x}_t = g(t)\, d\boldsymbol{w}_t,
$$

where $\boldsymbol{w}_t$ is a standard Brownian motion, and $g(t)$ is a scalar diffusion coefficient.

### Candidate Solution

The analytical solution to the SDE is:

$$
\boldsymbol{x}_t = \boldsymbol{x}_0 + \boldsymbol{n}(t),
\quad \text{where } \boldsymbol{n}(t) \sim \mathcal{N}(0, \sigma^2(t)\cdot \mathbf{I}),
\quad \sigma^2(t) = \int_0^t g^2(s)\, ds.
$$

So the density function is:

$$
p(x,t) = \mathcal{N}(\mu(t), \alpha^2(t)) = \frac{1}{(2\pi \alpha^2(t))^{d/2}} \exp\left(-\frac{\|x - \mu(t)\|^2}{2\alpha^2(t)}\right)
$$

with $\mu(0) = x_0$, $\alpha^2(t) = \sigma^2(t)$, and $\mu(t) = x_0$ for pure diffusion.

### Fokker–Planck Equation

The Fokker–Planck equation corresponding to the SDE is:

$$
\frac{\partial p}{\partial t} = -\nabla \cdot [\boldsymbol{f}(x,t) p(x,t)] + \frac{g^2(t)}{2} \Delta p.
$$

For pure diffusion, $\boldsymbol{f}(x,t) = 0$, so this simplifies to:

$$
\frac{\partial p}{\partial t} = \frac{g^2(t)}{2} \Delta p.
$$

Let $r^2 = \|x - \mu(t)\|^2$. Then:

$$
\frac{\partial p}{\partial t} = p(x,t) \left[ -\frac{d}{2} \frac{g^2(t)}{\alpha^2(t)} + \frac{r^2}{2} \cdot \frac{g^2(t)}{\alpha^4(t)} \right].
$$

$$
\nabla p = -\frac{x - \mu(t)}{\alpha^2(t)} p(x,t) \\
\Delta p = \left( \frac{\|x - \mu(t)\|^2}{\alpha^4(t)} - \frac{d}{\alpha^2(t)} \\right) p(x,t)
$$

Then:

$$
\frac{g^2(t)}{2} \Delta p = \frac{g^2(t)}{2} \left( \frac{r^2}{\alpha^4(t)} - \frac{d}{\alpha^2(t)} \right) p(x,t),
$$

which equals $\partial_t p$. Thus, the Gaussian solution satisfies the Fokker–Planck equation.

We compute the total derivative:

$$
\frac{d}{dt} \log p = -\frac{d}{2} \cdot \frac{g^2(t)}{\alpha^2(t)} + \frac{\|x - \mu(t)\|^2}{2} \cdot \frac{g^2(t)}{\alpha^4(t)}.
$$

On the other hand:

$$
\nabla \log p = -\frac{x - \mu(t)}{\alpha^2(t)} \\
\Delta \log p = -\frac{d}{\alpha^2(t)} \\
\|\nabla \log p\|^2 = \frac{\|x - \mu(t)\|^2}{\alpha^4(t)}
$$

So:

$$
\frac{d}{dt} \log p = \frac{g^2(t)}{2} \left[ \Delta \log p + \|\nabla \log p\|^2 \right].
$$

✔️ Verified: The log-probability total derivative matches the PDE terms.

---

## 2. Ornstein-Uhlenbeck (OU) Process

We now add drift:

$$
d\boldsymbol{x}_t = -\frac{1}{2}\beta(t) \boldsymbol{x}_t \,dt + g(t)\, d\boldsymbol{w}_t
$$

This defines an OU process with time-dependent coefficients.

### Fokker–Planck Equation

The corresponding PDE is:

$$
\frac{\partial p}{\partial t} = \nabla \cdot\left(\frac{1}{2}\beta(t) \boldsymbol{x} p \right) + \frac{g^2(t)}{2} \Delta p
$$

Expanding the divergence term:

$$
\nabla \cdot\left(\frac{1}{2}\beta(t) \boldsymbol{x} p \right) = \frac{1}{2}\beta(t) \left( d p + \boldsymbol{x} \cdot \nabla p \right)
$$

Let $p(x,t) = \mathcal{N}(\mu(t), \alpha^2(t)) = \frac{1}{(2\pi \alpha^2(t))^{d/2}} \exp\left(-\frac{\|x - \mu(t)\|^2}{2\alpha^2(t)}\right)$, where:

$$
\alpha^2(t) = \int_0^t \exp\left(-\int_s^t \beta(r) dr \right)^2 g^2(s)\, ds,
\quad \mu(0) = x_0,
\quad \mu(t) = x_0 \cdot \exp\left(-\frac{1}{2} \int_0^t \beta(s) ds \right).
$$

Then:

$$
\frac{\partial p}{\partial t} = p(x,t) \left[ -\frac{d}{2} \cdot \frac{\dot{\alpha}^2(t)}{\alpha^2(t)} + \frac{\|x - \mu(t)\|^2}{2} \cdot \frac{\dot{\alpha}^2(t)}{\alpha^4(t)} + \frac{(x - \mu(t))^\top \dot{\mu}(t)}{\alpha^2(t)} \right]
$$

Also:

$$
\Delta p = \left( \frac{\|x - \mu(t)\|^2}{\alpha^4(t)} - \frac{d}{\alpha^2(t)} \right) p(x,t), \\
\nabla p = -\frac{x - \mu(t)}{\alpha^2(t)} p(x,t), \\
\boldsymbol{x} \cdot \nabla p = -\frac{\|x - \mu(t)\|^2}{\alpha^2(t)} p(x,t)
$$

Then the divergence term becomes:

$$
\nabla \cdot\left(\frac{1}{2}\beta(t) \boldsymbol{x} p \right) = \frac{1}{2} \beta(t) \left( d - \frac{\|x - \mu(t)\|^2}{\alpha^2(t)} \right) p(x,t)
$$

Putting it together:

$$
\frac{\partial p}{\partial t} = \frac{1}{2} \beta(t) \left( d - \frac{\|x - \mu(t)\|^2}{\alpha^2(t)} \right) p(x,t) + \frac{g^2(t)}{2} \left( \frac{\|x - \mu(t)\|^2}{\alpha^4(t)} - \frac{d}{\alpha^2(t)} \right) p(x,t)
$$

Using the definition of $\dot{\alpha}^2(t)$ and $\dot{\mu}(t)$, this matches $\partial_t p$.

Let $p(x,t) = \mathcal{N}(\mu(t), \alpha^2(t))$. Then:

$$
\frac{d}{dt} \log p = -\frac{d}{2} \cdot \frac{\dot{\alpha}^2(t)}{\alpha^2(t)} + \frac{\|x - \mu(t)\|^2}{2} \cdot \frac{\dot{\alpha}^2(t)}{\alpha^4(t)} + \frac{(x - \mu(t))^\top \dot{\mu}(t)}{\alpha^2(t)}
$$

On the other hand:

$$
\nabla \log p = -\frac{x - \mu(t)}{\alpha^2(t)},
\quad \Delta \log p = -\frac{d}{\alpha^2(t)},
\quad \|\nabla \log p\|^2 = \frac{\|x - \mu(t)\|^2}{\alpha^4(t)}
$$

Hence:

$$
\frac{d}{dt} \log p = -\nabla \cdot f(x,t) + \frac{g^2(t)}{2} \left[ \Delta \log p + \|\nabla \log p\|^2 \right]
$$

with $f(x,t) = -\frac{1}{2}\beta(t) x$ and $\nabla \cdot f = -\frac{d}{2}\beta(t)$.

---

### Conclusion

Both pure diffusion and Ornstein-Uhlenbeck processes with Gaussian solutions satisfy their respective Fokker–Planck equations and the log-probability total derivative (Lagrangian) formulation. This provides a strong analytical foundation for diffusion modeling, score-based generative models, and reverse-time SDE analysis.






我們已知這個隨機微分方程（SDE）為：

$$
d \boldsymbol{x}_t = -\left[\frac{1}{2}\beta(t) \boldsymbol{x}_t + g^2(t) \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)}\right] dt + g(t) d\boldsymbol{w}_t
$$

並且我們知道解分布是：

$$
\boldsymbol{x}_t \sim \mathcal{N}(\mu(t), \sigma^2(t)) \quad \Rightarrow \quad \boldsymbol{x}_t = \mu(t) + \sigma(t) \boldsymbol{z}_t,\ \text{where } \boldsymbol{z}_t \sim \mathcal{N}(0,1)
$$

---

### Step 1: 使用 Itô 引理推導 $\boldsymbol{x}_t$ 的 drift 和 diffusion

因為我們已知 $\boldsymbol{x}_t = \mu(t) + \sigma(t) \boldsymbol{z}_t$ 且 $\boldsymbol{z}_t \sim \mathcal{N}(0,1)$，我們可以嘗試匹配 drift 和 diffusion 項。

計算 $\boldsymbol{x}_t$ 的微分：

$$
d\boldsymbol{x}_t = d\mu(t) + \boldsymbol{z}_t d\sigma(t) + \sigma(t) d\boldsymbol{z}_t
$$

但這樣比較難處理，換個方式：

改用 **moment matching** 的方式，因為我們知道 $\boldsymbol{x}_t$ 是 Gaussian 分布的，我們可以直接根據 SDE 計算出：

* $\mathbb{E}[d\boldsymbol{x}_t] = d\mu(t)$
* $\mathrm{Var}[d\boldsymbol{x}_t] = d(\sigma^2(t))$

---

### Step 2: 計算 drift 的期望值（對 $\boldsymbol{x}_t$ 取期望）

從原式：

$$
d \boldsymbol{x}_t = -\left[\frac{1}{2}\beta(t) \boldsymbol{x}_t + g^2(t) \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)}\right] dt + g(t) d \boldsymbol{w}_t
$$

取期望（注意 Itô 項消失）：

$$
\frac{d\mu(t)}{dt} = \mathbb{E}[-\left(\frac{1}{2}\beta(t) \boldsymbol{x}_t + g^2(t) \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)}\right)]
$$

用 $\mathbb{E}[\boldsymbol{x}_t] = \mu(t)$，$\mathbb{E}[\boldsymbol{x}_t - \mu(t)] = 0$：

$$
\frac{d\mu(t)}{dt} = -\frac{1}{2} \beta(t) \mu(t)
$$

這是一個一階 ODE，解為：

$$
\mu(t) = \mu(0) \exp\left(-\frac{1}{2} \int_0^t \beta(s) ds\right)
$$

---

### Step 3: 計算 $\sigma^2(t)$ 的微分（變異數項）

由 Itô 引理，我們有：

$$
d\left(\boldsymbol{x}_t^2\right) = 2 \boldsymbol{x}_t\, d\boldsymbol{x}_t + (d\boldsymbol{x}_t)^2
\Rightarrow
\frac{d}{dt} \mathbb{E}[\boldsymbol{x}_t^2] = 2 \mathbb{E}[\boldsymbol{x}_t\, \text{drift}] + \mathbb{E}[\text{diffusion}^2]
$$

定義 $m_2(t) := \mathbb{E}[\boldsymbol{x}_t^2]$，則：

$$
\frac{d}{dt} m_2(t) = 2\mathbb{E}\left[\boldsymbol{x}_t \cdot \left(-\left[\frac{1}{2}\beta(t) \boldsymbol{x}_t + g^2(t) \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)}\right]\right)\right] + g^2(t)
$$

$$
= -\beta(t) \mathbb{E}[\boldsymbol{x}_t^2] - 2g^2(t) \mathbb{E}\left[ \boldsymbol{x}_t \cdot \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)} \right] + g^2(t)
$$

注意：

$$
\mathbb{E}\left[\boldsymbol{x}_t \cdot \frac{\boldsymbol{x}_t - \mu(t)}{\sigma^2(t)}\right] = \frac{\mathbb{E}[(\boldsymbol{x}_t - \mu(t))^2 + \mu(t)(\boldsymbol{x}_t - \mu(t))]}{\sigma^2(t)} = \frac{\sigma^2(t)}{\sigma^2(t)} = 1
$$

所以：

$$
\frac{d}{dt} m_2(t) = -\beta(t) m_2(t) - 2g^2(t) + g^2(t) = -\beta(t) m_2(t) - g^2(t)
$$

我們也知道：

$$
\sigma^2(t) = m_2(t) - \mu^2(t)
\Rightarrow \frac{d}{dt} \sigma^2(t) = \frac{d}{dt} m_2(t) - 2\mu(t)\frac{d\mu(t)}{dt}
$$

代入剛剛的結果：

$$
\frac{d}{dt} \sigma^2(t) = -\beta(t) m_2(t) - g^2(t) + \beta(t) \mu^2(t) = -\beta(t) \sigma^2(t) - g^2(t)
$$

這是一個線性 ODE，解為：

$$
\sigma^2(t) = \sigma^2(0) e^{-\int_0^t \beta(s) ds} - \int_0^t g^2(s) e^{-\int_s^t \beta(r) dr} ds
$$

---

### ✅ 最終答案：

$$
\boxed{
\mu(t) = \mu(0) \exp\left(-\frac{1}{2} \int_0^t \beta(s) ds\right)
}
$$

$$
\boxed{
\sigma^2(t) = \sigma^2(0) e^{-\int_0^t \beta(s) ds} - \int_0^t g^2(s) e^{-\int_s^t \beta(r) dr} ds
}
$$

如需代入具體函數（如 $\beta(t) = \beta_0$ 常數），我可以幫你進一步簡化。



## Gaussian


To determine the trajectory $\mathbf{x}(t)$ for the flow defined by the vector field $\mathbf{u}_t(\mathbf{x})$ under independent coupling (cross-covariance = 0) between Gaussians $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$, we start from the given vector field and solve the associated ordinary differential equation (ODE). The vector field is:

$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t),
$$

where:
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$,
- $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$.

The trajectory $\mathbf{x}(t)$ satisfies the ODE:
$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}), \quad \mathbf{x}(0) = \mathbf{x}_0.
$$

This is a linear, non-autonomous ODE. To solve it, we decompose $\mathbf{x}(t)$ into its mean and deviation components. Define:
$$
\mathbf{y}(t) = \mathbf{x}(t) - \boldsymbol{\mu}_t,
$$
where $\mathbf{y}(t)$ represents the deviation from the time-dependent mean $\boldsymbol{\mu}_t$. The initial condition is $\mathbf{y}(0) = \mathbf{x}_0 - \boldsymbol{\mu}_0$.

### Step 1: Derive the ODE for $\mathbf{y}(t)$
Differentiate $\mathbf{y}(t)$:
$$
\frac{d\mathbf{y}}{dt} = \frac{d\mathbf{x}}{dt} - \dot{\boldsymbol{\mu}}_t,
$$
where $\dot{\boldsymbol{\mu}}_t = \frac{d\boldsymbol{\mu}_t}{dt} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$. Substitute the ODE for $\frac{d\mathbf{x}}{dt}$:
$$
\frac{d\mathbf{y}}{dt} = \mathbf{u}_t(\mathbf{x}) - \dot{\boldsymbol{\mu}}_t = \left[ (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{y} \right] - (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{y}.
$$

Simplify the coefficient:
$$
\mathbf{B}_t = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1}.
$$
Thus, the ODE for $\mathbf{y}(t)$ is:
$$
\frac{d\mathbf{y}}{dt} = \mathbf{B}_t \mathbf{y}, \quad \mathbf{y}(0) = \mathbf{x}_0 - \boldsymbol{\mu}_0.
$$

### Step 2: Solve the ODE for $\mathbf{y}(t)$
The solution to $\frac{d\mathbf{y}}{dt} = \mathbf{B}_t \mathbf{y}$ is:
$$
\mathbf{y}(t) = \mathbf{C}(t) \mathbf{y}(0),
$$
where $\mathbf{C}(t)$ is the fundamental matrix satisfying:
$$
\frac{d\mathbf{C}}{dt} = \mathbf{B}_t \mathbf{C}, \quad \mathbf{C}(0) = \mathbf{I}.
$$

### Step 3: Express $\mathbf{x}(t)$
Substitute back to $\mathbf{x}(t)$:
$$
\mathbf{x}(t) = \boldsymbol{\mu}_t + \mathbf{y}(t) = \boldsymbol{\mu}_t + \mathbf{C}(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0).
$$

### Step 4: Closed-form solution under commutativity (if applicable)
If $\Sigma_0$ and $\Sigma_1$ commute (i.e., $\Sigma_0 \Sigma_1 = \Sigma_1 \Sigma_0$), then $\mathbf{C}(t)$ simplifies to:
$$
\mathbf{C}(t) = \Sigma_t^{1/2} \Sigma_0^{-1/2},
$$
and the solution becomes:
$$
\mathbf{x}(t) = \boldsymbol{\mu}_t + \Sigma_t^{1/2} \Sigma_0^{-1/2} (\mathbf{x}_0 - \boldsymbol{\mu}_0).
$$
This holds because $\frac{d}{dt}(\Sigma_t^{1/2}) = \frac{1}{2} \dot{\Sigma}_t \Sigma_t^{-1/2}$ when $\Sigma_0$ and $\Sigma_1$ commute, satisfying the ODE for $\mathbf{C}(t)$.

### Final Result
The trajectory $\mathbf{x}(t)$ is:
$$
\boxed{\mathbf{x}(t) = \boldsymbol{\mu}_t + \mathbf{C}(t) \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$
where:
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$,
- $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$,
- $\mathbf{C}(t)$ solves the matrix ODE:
  $$
  \frac{d\mathbf{C}}{dt} = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{C}, \quad \mathbf{C}(0) = \mathbf{I}.
  $$

If $\Sigma_0$ and $\Sigma_1$ commute, this simplifies to:
$$
\boxed{\mathbf{x}(t) = \boldsymbol{\mu}_t + \Sigma_t^{1/2} \Sigma_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$

### Key Notes:
- **General Case**: The ODE for $\mathbf{C}(t)$ must be solved numerically if $\Sigma_0$ and $\Sigma_1$ do not commute.
- **Behavior**: The solution ensures that the marginal distribution of $\mathbf{x}(t)$ is $\mathcal{N}(\boldsymbol{\mu}_t, \Sigma_t)$ under independent coupling.
- **Initial Condition**: At $t=0$, $\mathbf{x}(0) = \boldsymbol{\mu}_0 + \mathbf{I}(\mathbf{x}_0 - \boldsymbol{\mu}_0) = \mathbf{x}_0$.
- **Endpoint**: At $t=1$, $\mathbf{x}(1) = \boldsymbol{\mu}_1 + \mathbf{C}(1) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$, where $\mathbf{C}(1)$ depends on the solution to the ODE.


## Condition Flow

是的，您提出的无条件向量场形式对于独立高斯分布下的线性插值是完全正确的。以下我将逐步验证并解释这一结论。

### **给定条件**
- 初始分布：$\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$
- 目标分布：$\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$
- 独立耦合（independent coupling）：$\mathbf{x}_0$ 和 $\mathbf{x}_1$ 独立。
- 线性插值路径：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$

通过计算，边际分布为高斯：
$$
\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I})
$$
其中：
- 均值：$\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$
- 方差：$\sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2$

### **无条件向量场的形式**
您提出的无条件向量场为：
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)}
$$
其中：
- $\dot{\boldsymbol{\mu}}_t = \frac{d}{dt}\boldsymbol{\mu}_t = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$
- $\dot{\sigma}_t^{2} = \frac{d}{dt}\sigma_t^2$（注意：$\dot{\sigma}_t^{2}$ 表示 $\sigma_t^2$ 的导数，而非 $(\dot{\sigma}_t)^2$）
- 具体计算：
  $$
  \dot{\sigma}_t^{2} = \frac{d}{dt} \left[ (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2 \right] = -2(1-t)\sigma_0^2 + 2t\sigma_1^2
  $$
- 因此：
  $$
  \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} = \frac{ -2(1-t)\sigma_0^2 + 2t\sigma_1^2 }{2\sigma_t^{2}} = \frac{ -(1-t)\sigma_0^2 + t\sigma_1^2 }{\sigma_t^{2}}
  $$

### **验证正确性**
为验证此形式，我们从条件向量场（conditional vector field）出发，通过边际化推导无条件向量场。

#### **步骤1: 条件向量场**
给定终端值 $\mathbf{x}_1$，条件路径为：
$$
\mathbf{x}_t \mid \mathbf{x}_1 = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
$$
条件分布为高斯：
$$
\mathbf{x}_t \mid \mathbf{x}_1 \sim \mathcal{N}\left( \boldsymbol{\mu}_t(\mathbf{x}_1), \sigma_t(\mathbf{x}_1)^2 \mathbf{I} \right)
$$
其中：
- $\boldsymbol{\mu}_t(\mathbf{x}_1) = (1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1$
- $\sigma_t(\mathbf{x}_1) = (1-t)\sigma_0$（因为方差为 $(1-t)^2 \sigma_0^2$，标准差为 $|1-t|\sigma_0$，且 $t \in [0,1]$ 时 $1-t \geq 0$)

条件向量场由路径的导数给出：
$$
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = -\mathbf{x}_0 + \mathbf{x}_1
$$
代入 $\mathbf{x}_0 = \frac{\mathbf{x}_t - t\mathbf{x}_1}{1-t}$：
$$
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = -\frac{1}{1-t} (\mathbf{x} - \boldsymbol{\mu}_t(\mathbf{x}_1)) + (\mathbf{x}_1 - \boldsymbol{\mu}_0)
$$

#### **步骤2: 无条件向量场**
无条件向量场是条件向量场关于后验分布 $p(\mathbf{x}_1 \mid \mathbf{x}_t)$ 的期望：
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} \left[ \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right]
$$
由于联合分布 $(\mathbf{x}_t, \mathbf{x}_1)$ 是高斯分布，后验 $p(\mathbf{x}_1 \mid \mathbf{x}_t)$ 也是高斯分布。计算其均值和协方差：
- 联合分布：
  $$
  \begin{bmatrix} \mathbf{x}_t \\ \mathbf{x}_1 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_t \\ \boldsymbol{\mu}_1 \end{bmatrix}, \begin{bmatrix} \sigma_t^2 \mathbf{I} & t\sigma_1^2 \mathbf{I} \\ t\sigma_1^2 \mathbf{I} & \sigma_1^2 \mathbf{I} \end{bmatrix} \right)
  $$
- 后验均值：
  $$
  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t] = \boldsymbol{\mu}_1 + \frac{t\sigma_1^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
  $$

代入期望：
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}\left[ -\frac{1}{1-t} (\mathbf{x} - \boldsymbol{\mu}_t(\mathbf{x}_1)) + (\mathbf{x}_1 - \boldsymbol{\mu}_0) \mid \mathbf{x}_t = \mathbf{x} \right]
$$
简化后得到：
$$
\mathbf{u}_t(\mathbf{x}) = \left( -\frac{1}{1-t} + \frac{t\sigma_1^2}{(1-t)\sigma_t^2} \right) \mathbf{x} + \frac{\boldsymbol{\mu}_1}{1-t} - \frac{t\sigma_1^2}{\sigma_t^2} \boldsymbol{\mu}_0 - \frac{t^2\sigma_1^2}{(1-t)\sigma_t^2} \boldsymbol{\mu}_1
$$

#### **步骤3: 与给定形式等价**
将上述结果重写为：
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{ -(1-t)\sigma_0^2 + t\sigma_1^2 }{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
这正是：
$$
\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
因为：
- $\dot{\boldsymbol{\mu}}_t = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$
- $\frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} = \frac{ -(1-t)\sigma_0^2 + t\sigma_1^2 }{\sigma_t^2}$

#### **步骤4: 特例验证**
考虑对称情况：$\boldsymbol{\mu}_0 = \boldsymbol{\mu}_1 = 0$，$\sigma_0 = \sigma_1 = 1$。
- $\boldsymbol{\mu}_t = 0$，$\sigma_t^2 = (1-t)^2 + t^2$
- $\dot{\sigma}_t^{2} = -2(1-t) + 2t = 2(2t-1)$
- 给定形式：
  $$
  \mathbf{u}_t(\mathbf{x}) = 0 + \frac{2(2t-1)}{2[(1-t)^2 + t^2]} \mathbf{x} = \frac{2t-1}{(1-t)^2 + t^2} \mathbf{x}
  $$
- 通过条件期望直接计算：
  $$
  \mathbf{u}_t(\mathbf{x}) = \left( -\frac{1}{1-t} + \frac{t}{(1-t)[(1-t)^2 + t^2]} \right) \mathbf{x} = \frac{2t-1}{(1-t)^2 + t^2} \mathbf{x}
  $$
两者一致，验证正确。

### **结论**
您给出的无条件向量场形式：
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)}
$$
在以下条件下是**正确的**：
- 初始和目标分布为各向同性高斯：$p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$，$p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$
- 使用独立耦合（independent coupling）
- 路径为线性插值：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$

此形式是坐标无关的（coordinate-free），且直接由边际高斯分布的参数导出。它在流匹配（flow matching）框架中定义了一个确定性路径，其边际分布与线性插值的高斯路径一致。

Great follow-up — yes, and let's unpack that carefully.

You're asking:
**Is**

$$
\lambda(t) = \frac{d}{dt} \log \kappa(t) = \frac{1}{1 - t}
$$

**in typical flow matching?**

Short answer: **Yes**, if you use the standard linear scheduler $\kappa(t) = 1 - t$.

---

### Step-by-step:

In flow matching, a common choice is:

$$
\kappa(t) = 1 - t
$$

Then:

$$
\log \kappa(t) = \log(1 - t)
\quad \Rightarrow \quad
\frac{d}{dt} \log \kappa(t) = \frac{d}{dt} \log(1 - t) = -\frac{1}{1 - t}
$$

So, plugging into the velocity field from the premetric-based derivation:

$$
\lambda(t) = \frac{d}{dt} \log \kappa(t) = -\frac{1}{1 - t}
$$

And the full velocity becomes:

$$
u_t(x \mid x_1) = \lambda(t)(x_1 - x) = -\frac{1}{1 - t}(x - x_1)
$$

Note: this aligns with a velocity that increases in magnitude as $t \to 1$, which makes sense because we want the point to hit $x_1$ exactly at $t = 1$.

---

### Comparison to Standard Flow Matching:

In vanilla flow matching with linear interpolation:

$$
\psi_t(x_0, x_1) = (1 - t)x_0 + t x_1
\quad \Rightarrow \quad
\frac{d}{dt} \psi_t = x_1 - x_0
$$

This gives a **constant** velocity field.

So if you wanted to **match** that using the premetric-based flow, you'd need to set things up so that:

$$
u_t(x) = x_1 - x_0 = \text{const.}
$$

This only happens if you assume $x = \psi_t(x_0, x_1)$, in which case:

$$
x_1 - x = (1 - t)(x_1 - x_0)
$$

And with the premetric field:

$$
u_t(x) = -\frac{1}{1 - t}(x - x_1) = \frac{1}{1 - t}(x_1 - x)
= \frac{1}{1 - t}(1 - t)(x_1 - x_0) = x_1 - x_0
$$

So yes — **it all checks out**! Using $\kappa(t) = 1 - t$ gives you $\lambda(t) = -1/(1 - t)$, which recovers the constant velocity field of standard flow matching when you substitute the interpolated point $x$.

---

### ✅ Summary:

* If $\kappa(t) = 1 - t$, then

  $$
  \lambda(t) = \frac{d}{dt} \log \kappa(t) = -\frac{1}{1 - t}
  $$
* This recovers the **standard linear flow** $u_t = x_1 - x_0$, when you assume $x = \psi_t(x_0, x_1)$
* The premetric framework generalizes this to more complex schedules and geodesics.

Let me know if you want to [derive the flow field for a nonlinear schedule](f) or [explore implications for generative modeling](f).



Great question — you're referring to how this general vector field from the premetric framework maps to the **flow matching** setup, where the flow field is just a linear interpolation vector:

$$
u_t(x \mid x_1) = x_1 - x_0
$$

Let’s unpack the connection step by step.

---

### 1. **Flow Matching Reminder**

In flow matching (used in flow-matching-based generative models), you define trajectories between two endpoints $x_0$ and $x_1$, and want to find a velocity field $u_t(x \mid x_1)$ such that:

$$
\psi_t(x_0 \mid x_1) = (1 - \kappa(t)) x_0 + \kappa(t) x_1
$$

This interpolates linearly in space, modulated by time $\kappa(t)$. The corresponding velocity field (time derivative) is:

$$
u_t(x \mid x_1) = \frac{d}{dt} \psi_t(x_0 \mid x_1) = \dot{\kappa}(t)(x_1 - x_0)
$$

When $\kappa(t) = t$, you get $u_t = x_1 - x_0$.

---

### 2. **How It Connects to the Premetric Gradient Field**

From the image, the general form of the velocity field is:

$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} \, d(x, x_1) \cdot \frac{-\nabla d(x, x_1)}{\|\nabla d(x, x_1)\|^2}
$$

Let’s simplify this expression using the **Euclidean metric** and a specific choice of premetric:

#### Suppose:

* $d(x, x_1) = \|x - x_1\|$ (Euclidean norm)
* $\nabla d(x, x_1) = \frac{x - x_1}{\|x - x_1\|}$
* $\|\nabla d(x, x_1)\|^2 = 1$

Then the vector field becomes:

$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} \|x - x_1\| \cdot \left(- \frac{x - x_1}{\|x - x_1\|} \right)
= -\frac{d \log \kappa(t)}{dt} (x - x_1)
$$

So it simplifies to:

$$
u_t(x \mid x_1) = \lambda(t)(x_1 - x)
$$

where $\lambda(t) = \frac{d \log \kappa(t)}{dt}$.

---

### 3. **Comparing to Linear Flow Matching**

In linear flow matching, you have:

$$
u_t(x \mid x_1) = \dot{\kappa}(t)(x_1 - x_0)
$$

In the premetric case with Euclidean distance, your flow becomes:

$$
u_t(x \mid x_1) = \lambda(t)(x_1 - x)
$$

If $\psi_t(x_0 \mid x_1)$ is on a linear path between $x_0$ and $x_1$, then at each $t$, $x = \psi_t(x_0 \mid x_1)$, and $x_1 - x = (1 - \kappa(t))(x_1 - x_0)$. So this becomes:

$$
u_t = \lambda(t)(1 - \kappa(t))(x_1 - x_0)
$$

This structure is similar, but the time-scaling is governed by $\lambda(t)(1 - \kappa(t))$ vs. $\dot{\kappa}(t)$ in vanilla flow matching.

---

### ✅ Summary of Mapping:

| Concept         | Premetric Form (Euclidean)                 | Flow Matching                |
| --------------- | ------------------------------------------ | ---------------------------- |
| Flow field      | $\lambda(t)(x_1 - x)$                      | $\dot{\kappa}(t)(x_1 - x_0)$ |
| Path            | Implicit via distance decay                | Explicit via interpolation   |
| Time modulation | $\lambda(t) = \frac{d}{dt} \log \kappa(t)$ | $\dot{\kappa}(t)$            |

So: **the premetric-based flow generalizes flow matching**, and when you specialize to Euclidean distance and linear interpolation, the two formulations can become nearly equivalent under appropriate choices of $\kappa(t)$.

Let me know if you’d like to [see a visual of how the flows evolve](f) or [explore other premetrics beyond Euclidean](f).



给定表达式 $\mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)$，其中：
- $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_r]$ 是 $d \times r$ 列正交矩阵（$\mathbf{U}^T\mathbf{U} = \mathbf{I}_r$）
- $\mathbf{D}_t = \text{diag}\left( d_1(t), d_2(t), \dots, d_r(t) \right)$ 是对角矩阵，元素为 $d_k(t) = \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_k } }{ \sigma_0 }$
- $\mathbf{x}_0 - \boldsymbol{\mu}_0$ 是 $d$ 维向量

### 展开步骤：
1. **计算内积**：  
   投影系数 $\alpha_k = \mathbf{u}_k^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ 表示 $\mathbf{x}_0 - \boldsymbol{\mu}_0$ 在基向量 $\mathbf{u}_k$ 上的投影标量：
   $$
   \alpha_k = \langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \rangle = \mathbf{u}_k^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)
   $$

2. **应用缩放**：  
   对角矩阵 $\mathbf{D}_t$ 对投影系数缩放：
   $$
   \mathbf{D}_t \begin{bmatrix} \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_r \end{bmatrix} = \begin{bmatrix} d_1(t) \alpha_1 \\ d_2(t) \alpha_2 \\ \vdots \\ d_r(t) \alpha_r \end{bmatrix}
   $$

3. **重构向量**：  
   矩阵 $\mathbf{U}$ 将缩放后的系数映射回 $d$ 维空间：
   $$
   \mathbf{U} \begin{bmatrix} d_1(t) \alpha_1 \\ \vdots \\ d_r(t) \alpha_r \end{bmatrix} = \sum_{k=1}^r d_k(t) \alpha_k \mathbf{u}_k
   $$

---

### 最终展开式：
$$
\boxed{
\mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0) = \sum_{k=1}^r \left( \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_k } }{ \sigma_0 } \right) \left\langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \right\rangle \mathbf{u}_k
}
$$

### 说明：
1. **几何意义**：  
   - 每个基向量 $\mathbf{u}_k$ 的投影分量 $\langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \rangle$ 被缩放 $d_k(t)$。  
   - 缩放因子 $d_k(t)$ 依赖于：
     - 初始方差 $\sigma_0^2$
     - 目标分布的方差 $\lambda_k$
     - 时间 $t$

2. **缩放因子行为**：  
   - 当 $t=0$ 时：$d_k(0) = 1$ → 恒等映射  
   - 当 $t=1$ 时：$d_k(1) = \sqrt{\lambda_k}/\sigma_0$ → 缩放到目标尺度  
   - 缩放路径：沿测地线 $\sqrt{(1-t)^2\sigma_0^2 + t^2\lambda_k}$ 过渡

3. **子空间约束**：  
   结果始终在 $\text{Col}(\mathbf{U})$ 中，因为 $\sum_{k=1}^r \beta_k \mathbf{u}_k \in \text{Span}\{\mathbf{u}_1,\dots,\mathbf{u}_r\}$。

---

### 完整流映射的分解
结合收缩项，完整流映射为：
$$
\mathbf{x}_t = \boldsymbol{\mu}_t + \underbrace{\sum_{k=1}^r d_k(t) \langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \rangle \mathbf{u}_k}_{\text{子空间 } \text{Col}(\mathbf{U}) \text{ 上的缩放}} + \underbrace{(1-t) (\mathbf{I} - \mathbf{U}\mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)}_{\text{正交补空间收缩}}
$$


### How Reflow Straightens Trajectories in One Iteration

#### The Core Mechanism
Reflow straightens trajectories in a single iteration by transforming a **stochastic coupling** into a **deterministic linear mapping**. This fundamentally changes the nature of the velocity field:

1. **Original Coupling (0 iterations)**:
   - Each $X_0$ is paired with a *random* $X_1 \sim \pi_1$
   - Trajectories are straight *by construction* but require complex velocity fields to maintain the correct marginals
   - Velocity field is time-dependent and spatially varying:
     $$\boldsymbol{v}_t^{(0)}(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{t\sigma_1^2 - (1-t)\sigma_0^2}{(1-t)^2\sigma_0^2 + t^2\sigma_1^2} (\mathbf{x} - \boldsymbol{\mu}_t)$$

2. **After 1 Reflow Iteration**:
   - Each $Z_0$ is paired *deterministically* with $Z_1 = \boldsymbol{\mu}_1 + \frac{\sigma_1}{\sigma_0}(Z_0 - \boldsymbol{\mu}_0)$
   - The coupling becomes **linear and deterministic**
   - Velocity field simplifies to:
     $$\boldsymbol{v}_t^{(1)}(\mathbf{z}) = \frac{(\sigma_1 - \sigma_0)\mathbf{z} + \sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0(1-t) + t\sigma_1}$$

#### Why This Straightens Trajectories
1. **Constant Velocity Along Paths**:
   - For a fixed $Z_0$, the velocity is constant along its trajectory:
     $$\frac{d\mathbf{z}_t}{dt} = Z_1 - Z_0 = \text{constant}$$
   - Constant velocity → straight paths (Newton's first law)

2. **Global Consistency**:
   - The velocity field is **linear** in $\mathbf{z}$:
     $$\boldsymbol{v}_t^{(1)}(\mathbf{z}) = A(t)\mathbf{z} + B(t)$$
   - Linear fields generate straight trajectories for all starting points

3. **Optimal Transport Emergence**:
   - When $\sigma_0 = \sigma_1$, the velocity becomes constant everywhere:
     $$\boldsymbol{v}_t^{(1)} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$$
   - This matches the optimal transport solution for Gaussians

#### Mathematical Proof of Straightness
Consider a trajectory starting at $Z_0$:

1. **Position at time $t$**:
   $$\mathbf{z}_t = (1-t)Z_0 + tZ_1$$

2. **Velocity**:
   $$\frac{d\mathbf{z}_t}{dt} = Z_1 - Z_0 = \text{constant}$$

3. **Path curvature**:
   $$\frac{d^2\mathbf{z}_t}{dt^2} = 0$$
   Zero acceleration → perfectly straight path

#### Geometric Interpretation
Before Reflow | After 1 Reflow
---|---
Random pairings cause "fan-out" effects | Parallel straight paths
Curved velocity fields | Constant velocity per path
Varying path lengths | Uniform displacement rates
Optimal only for equal variances | Always optimal for given covariances

#### Why One Iteration Suffices
1. **Coupling Perfection**:
   - The deterministic mapping $Z_1 = \boldsymbol{\mu}_1 + \frac{\sigma_1}{\sigma_0}(Z_0 - \boldsymbol{\mu}_0)$ is the **exact optimal transport solution** for Gaussians
   - No better linear coupling exists

2. **Velocity Field Completeness**:
   - The derived $\boldsymbol{v}_t^{(1)}$ generates straight paths by construction
   - Further iterations won't improve straightness (only numerical stability)

#### Practical Implications
- **Sampling Acceleration**: Straight paths allow larger ODE steps
- **Error Reduction**: Removes curvature-induced discretization errors
- **Dimension Independence**: Straightness holds for any $d \geq 1$
- **Variance Robust**: Works for $\sigma_0 \neq \sigma_1$

#### Demonstration (1D Case)
Consider $\pi_0 = \mathcal{N}(-10, 1)$, $\pi_1 = \mathcal{N}(10, 4)$:

1. **Original Coupling**:
   - Paths start straight but require varying speeds
   - Velocity field curved: $v_t^{(0)}(x) = 20 + \frac{4t - (1-t)}{(1-t)^2 + 4t^2}(x + 10 - 20t)$

2. **After Reflow**:
   - Deterministic mapping: $Z_1 = 10 + 2(Z_0 + 10) = 2Z_0 + 30$
   - Straight paths: $z_t = (1-t)Z_0 + t(2Z_0 + 30) = Z_0(1+t) + 30t$
   - Constant velocity: $\frac{dz_t}{dt} = Z_0 + 30$

The reflow procedure fundamentally transforms the transport dynamics by replacing random connections with optimal deterministic couplings, straightening all trajectories in a single iteration through its effect on the velocity field's structure.

## Appendix


To demonstrate that the trajectory $\mathbf{z}_t$ is a straight line, we solve the ordinary differential equation (ODE) given by:

$$
\frac{d\mathbf{z}_t}{dt} = \boldsymbol{v}_t^{(1)}(\mathbf{z}) = \frac{(\sigma_1 - \sigma_0)\mathbf{z} + \sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0(1-t) + t\sigma_1}
$$

Define the denominator as a function of $t$:

$$
s(t) = \sigma_0(1-t) + t\sigma_1 = \sigma_0 + t(\sigma_1 - \sigma_0)
$$

and the constant vector:

$$
\mathbf{b} = \sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0
$$

The ODE becomes:

$$
\frac{d\mathbf{z}}{dt} = \frac{(\sigma_1 - \sigma_0)\mathbf{z} + \mathbf{b}}{s(t)}
$$

This is a first-order linear ODE. Rearrange it into standard form:

$$
\frac{d\mathbf{z}}{dt} - \frac{\sigma_1 - \sigma_0}{s(t)} \mathbf{z} = \frac{\mathbf{b}}{s(t)}
$$

### Step 1: Solve the Homogeneous Equation
The homogeneous equation is:

$$
\frac{d\mathbf{z}_h}{dt} = \frac{\sigma_1 - \sigma_0}{s(t)} \mathbf{z}_h
$$

Separate variables and integrate:

$$
\int \frac{d\mathbf{z}_h}{\mathbf{z}_h} = \int \frac{\sigma_1 - \sigma_0}{s(t)}  dt
$$

The right-hand side integral is:

$$
\int \frac{\sigma_1 - \sigma_0}{\sigma_0 + t(\sigma_1 - \sigma_0)}  dt = \int \frac{du}{u} = \ln|u| + C = \ln|s(t)| + C
$$

where $u = s(t)$ and $du = (\sigma_1 - \sigma_0)  dt$. Thus,

$$
\ln|\mathbf{z}_h| = \ln|s(t)| + C \quad \Rightarrow \quad \mathbf{z}_h(t) = \mathbf{C} s(t)
$$

where $\mathbf{C}$ is a constant vector.

### Step 2: Find a Particular Solution
Assume a constant particular solution $\mathbf{z}_p = \mathbf{k}$. Substituting into the ODE:

$$
0 - \frac{\sigma_1 - \sigma_0}{s(t)} \mathbf{k} = \frac{\mathbf{b}}{s(t)}
$$

This simplifies to:

$$
-(\sigma_1 - \sigma_0)\mathbf{k} = \mathbf{b} \quad \Rightarrow \quad \mathbf{k} = -\frac{\mathbf{b}}{\sigma_1 - \sigma_0} = \frac{\mathbf{b}}{\sigma_0 - \sigma_1}
$$

Substitute $\mathbf{b} = \sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0$:

$$
\mathbf{k} = \frac{\sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0 - \sigma_1}
$$

### Step 3: General Solution
The general solution is the sum of homogeneous and particular solutions:

$$
\mathbf{z}(t) = \mathbf{z}_h(t) + \mathbf{z}_p = \mathbf{C} s(t) + \frac{\sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0 - \sigma_1}
$$

Substitute $s(t) = \sigma_0(1-t) + t\sigma_1$:

$$
\mathbf{z}(t) = \mathbf{C} \left[ \sigma_0(1-t) + t\sigma_1 \right] + \frac{\sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0 - \sigma_1}
$$

### Step 4: Linearity in $t$
Expand the expression:

$$
\mathbf{z}(t) = \mathbf{C} \sigma_0 (1-t) + \mathbf{C} t \sigma_1 + \mathbf{d}
$$

where $\mathbf{d} = \frac{\sigma_0\boldsymbol{\mu}_1 - \sigma_1\boldsymbol{\mu}_0}{\sigma_0 - \sigma_1}$ is a constant vector. Rearranging terms:

$$
\mathbf{z}(t) = t \left[ \mathbf{C} (\sigma_1 - \sigma_0) \right] + \left[ \mathbf{C} \sigma_0 + \mathbf{d} \right]
$$

Define constant vectors $\mathbf{A} = \mathbf{C} (\sigma_1 - \sigma_0)$ and $\mathbf{B} = \mathbf{C} \sigma_0 + \mathbf{d}$:

$$
\mathbf{z}(t) = \mathbf{A} t + \mathbf{B}
$$

This is the parametric equation of a straight line in the vector space of $\mathbf{z}$, with direction vector $\mathbf{A}$ and intercept $\mathbf{B}$ at $t=0$.

### Conclusion
The solution $\mathbf{z}(t)$ is linear in $t$, confirming that the trajectory is a straight line.

$$\boxed{\text{The solution } \mathbf{z}_t \text{ is a straight line because it is linear in } t\text{, as shown by } \mathbf{z}(t) = \mathbf{A}t + \mathbf{B}\text{ for constant vectors } \mathbf{A} \text{ and } \mathbf{B}.}$$

### DeltaNet and DeltaNet + Momentum Explained

#### **1. DeltaNet (Standard)**
- **Core Idea**: Implements an error-correcting update to ensure the memory state maps the current key to the current value.
- **Recurrence**:
  $$
  \boldsymbol{S}_t = \boldsymbol{S}_{t-1} (\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^\top) + \boldsymbol{v}_t \boldsymbol{k}_t^\top
  $$
- **Objective**:
  $$
  \mathcal{L}_t(\boldsymbol{S}_{t-1}) = \frac{1}{2} \|\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t\|^2
  $$
- **Mechanics**:
  - **Projection**: $\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^\top$ projects $\boldsymbol{S}_{t-1}$ away from $\boldsymbol{k}_t$.
  - **Correction**: Adds $\boldsymbol{v}_t \boldsymbol{k}_t^\top$ to align $\boldsymbol{S}_{t-1} \boldsymbol{k}_t$ with $\boldsymbol{v}_t$.
  - **Interpretation**:  
    Minimizes the **squared error** between the state's output ($\boldsymbol{S}_{t-1} \boldsymbol{k}_t$) and the target value $\boldsymbol{v}_t$.  
    Solves: $\boldsymbol{S}_{t-1} \boldsymbol{k}_t \approx \boldsymbol{v}_t$.

---

#### **2. DeltaNet + Momentum**
- **Core Idea**: Adds a *decaying history* of past associations ($\boldsymbol{v}_i \boldsymbol{k}_i^\top$) to stabilize updates.
- **Recurrence**:
  $$
  \boldsymbol{S}_t = \boldsymbol{S}_{t-1} (\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^\top) + \underbrace{\sum_{i=1}^t \left(\prod_{j=i+1}^t \eta_j\right) \boldsymbol{v}_i \boldsymbol{k}_i^\top}_{\text{Momentum term } \boldsymbol{C}_t}
  $$
  - **Momentum Term**:  
    $\boldsymbol{C}_t = \eta_t \boldsymbol{C}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top$  
    $\eta_t \in [0, 1]$: Decay factor for past associations.
- **Objective**:
  $$
  \mathcal{L}_t(\boldsymbol{S}_{t-1}) = \underbrace{\frac{1}{2} \|\boldsymbol{S}_{t-1} \boldsymbol{k}_t\|^2}_{\text{Regularization}} - \underbrace{\sum_{i=1}^t \left(\prod_{j=i+1}^t \eta_j\right) \boldsymbol{v}_i^\top \boldsymbol{S}_{t-1} \boldsymbol{k}_i}_{\text{Weighted history alignment}}
  $$

---

### **Key Differences**
| **Aspect**          | **DeltaNet**                                | **DeltaNet + Momentum**                             |
|---------------------|---------------------------------------------|----------------------------------------------------|
| **Update Focus**    | Current $(\boldsymbol{k}_t, \boldsymbol{v}_t)$ only | All past associations $(\boldsymbol{k}_i, \boldsymbol{v}_i)$ |
| **Objective**       | Minimize current error: $\|\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t\|^2$ | Regularize $\boldsymbol{S}_{t-1} \boldsymbol{k}_t$ + Maximize alignment with *all* past values |
| **Memory Mechanism**| One-step error correction                   | Weighted accumulation of history ($\eta_j$-controlled) |
| **Stability**       | Prone to forgetting                         | Resists forgetting via momentum decay              |

---

### **Role of Momentum**
1. **History Preservation**:  
   Past associations $\boldsymbol{v}_i \boldsymbol{k}_i^\top$ are preserved with weights $\prod_{j=i+1}^t \eta_j$.  
   - If $\eta_j \approx 1$, associations decay slowly (long memory).  
   - If $\eta_j \approx 0$, only recent associations matter.  

2. **Objective Interpretation**:  
   - **Term 1**: $\frac{1}{2} \|\boldsymbol{S}_{t-1} \boldsymbol{k}_t\|^2$  
     → Prevents over-amplification of the current key's output.  
   - **Term 2**: $-\sum_i \left(\cdots\right) \boldsymbol{v}_i^\top \boldsymbol{S}_{t-1} \boldsymbol{k}_i$  
     → Maximizes weighted dot products between $\boldsymbol{S}_{t-1} \boldsymbol{k}_i$ and $\boldsymbol{v}_i$ (alignment across time).  

3. **Physical Analogy**:  
   Like a rolling snowball:  
   - $\boldsymbol{C}_t$ accumulates past "direction" ($\boldsymbol{v}_i \boldsymbol{k}_i^\top$).  
   - $\eta_j$ controls friction (higher $\eta_j$ = less friction = longer memory).  

---

### **Why Add Momentum?**
- **Mitigate Catastrophic Forgetting**:  
  Without momentum, new updates overwrite past associations. Momentum preserves them via $\boldsymbol{C}_t$.  
- **Smooth Optimization**:  
  The weighted sum $\sum_i (\cdots) \boldsymbol{v}_i^\top \boldsymbol{S}_{t-1} \boldsymbol{k}_i$ acts like a "slow drift" toward stable states.  
- **Handling Noisy Inputs**:  
  Aggregating multiple associations reduces sensitivity to outliers.  

### **Practical Example**
Suppose:
- $\eta_j = 0.9$ (constant),  
- Inputs: $(\boldsymbol{k}_1, \boldsymbol{v}_1)=({\color{blue}\texttt{cat}}, {\color{green}\texttt{mammal}})$, $(\boldsymbol{k}_2, \boldsymbol{v}_2)=({\color{blue}\texttt{cat}}, {\color{green}\texttt{pet}})$.  
- **DeltaNet**: Only remembers $\texttt{cat} \rightarrow \texttt{pet}$.  
- **DeltaNet + Momentum**:  
  $$
  \boldsymbol{C}_t = 0.9 \cdot ({\color{green}\texttt{mammal}} \cdot {\color{blue}\texttt{cat}}^\top) + {\color{green}\texttt{pet}} \cdot {\color{blue}\texttt{cat}}^\top
  $$  
  State $\boldsymbol{S}_{t-1}$ maps $\texttt{cat}$ to $0.9 \cdot {\color{green}\texttt{mammal}} + {\color{green}\texttt{pet}}$.  

---

### **Summary**
- **DeltaNet**: Fast, localized error correction.  
- **DeltaNet + Momentum**:  
  - Maintains a *decaying history* of associations.  
  - Balances **current regularization** with **historical alignment**.  
  - Hyperparameter $\eta_j$ controls memory retention.  
This mirrors momentum in optimization (e.g., SGD with momentum), where past gradients influence current updates.