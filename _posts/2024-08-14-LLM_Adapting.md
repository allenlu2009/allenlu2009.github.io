---
title: RAG vs. Long Context vs. Fine-tuning
date: 2024-07-29 23:10:08
categories:
- AI
tags: [LLM, RAG, Fine-Tuning]
typora-root-url: ../../allenlu2009.github.io


---
  
## Introduction

Large language models (LLMs) have demonstrated exceptional abilities across a plethora of language tasks and natural language processing (NLP) [benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Product use cases based on these “generalized” models are on the rise. In this blog post, we’ll provide guidance for small AI product teams who want to adapt and integrate LLMs into their projects. Let’s start by clarifying the (often confusing) terminology surrounding LLMs, then briefly comparing the different methods of adaptation available, and finally recommending a step-by-step flowchart to identify the right approach for your use case.

|           | Pretrain       | Continuous Pretrain | Finetune | PEFT | RAG | In-context learning |
| --------- | -------------- | ------------------- | -------- | ---- | --- | ------------------- |
| parameter | 100%           | 100%                | 100%     | 5%   | 0%  | 0%                  |
| data      | 100%           | 5-10%               | 1%       | 1%   | 1%? | 0%                  |
| method    | SSL, SFT, RHLF | SSL only?           | SFT      | SFT  | x   | in-context          |
  

<img src="/media/image-20240814161459.png" alt="20240814161459" style="zoom:30%;" />

## Approaches to LLM adaptation

### Pre-training

Pre-training is the process of training an LLM from scratch using trillions of data tokens. The model is trained using a self-supervised algorithm. Most commonly, training happens by predicting the next token autoregressively (a.k.a. causal language modeling). Pre-training typically requires thousands of GPU hours spread across multiple GPUs. The output model from pre-training is known as a [foundation model](https://blogs.nvidia.com/blog/what-are-foundation-models/).

### Continued pre-training

Continued pre-training (a.k.a. second-stage pre-training) involves further training a foundation model with new, unseen domain data. The same self-supervised algorithm from the initial pre-training is used. All model weights are typically involved, and a fraction of the original data is mixed with the new data.

### Fine-tuning

Fine-tuning is the process of adapting a pre-trained language model using an annotated dataset in a supervised manner or using reinforcement learning-based techniques. There are two major differences compared to pre-training:

1. Supervised training on an annotated dataset—that contains the correct labels/answers/preferences—instead of self-supervised training

2. Requires fewer tokens (thousands or millions instead of the billions or trillions needed in pre-training) where the primary aim is to enhance abilities like instruction following, human alignment, task performance, etc.

There are two dimensions to understanding the current landscape of fine-tuning: percentage of parameters changed and new capabilities added as a result of the fine-tuning.

#### Percentage of parameters changed

Depending on the number of parameters changed, there are two categories of algorithms:

1. **Full fine-tuning:** As the name suggests, this encompasses changing all parameters of the model and includes legacy fine-tuning as done on smallish models like XLMR and BERT (100 – 300M parameters) as well as fine-tuning on large models like [Llama 2](https://ai.meta.com/blog/llama-2/), GPT3 (1B+ parameters), etc.

2. **Parameter-efficient fine-tuning (PEFT):** Instead offine-tuning all LLM weights, PEFT algorithms only fine-tune a small number of additional parametersorupdate a subset of the pre-trained parameters, typically 1 – 6% of the total parameters.

#### Capabilities added to a base model

Fine-tuning is carried out with the intention of adding capabilities to the pre-trained model—for example: instruction following, human alignment, etc. Chat-tuned Llama 2 is [an example](https://arxiv.org/abs/2307.09288) of a fine-tuned model with added instruction-following and alignment capabilities.

### Retrieval augmented generation (RAG)

Enterprises can also adapt LLMs by adding a domain-specific knowledge base. RAG is quintessentially “search-powered LLM text generation.” Introduced in 2020, RAG uses a dynamic prompt context that is retrieved using the user question and injected into the LLM prompt in order to steer it to use the retrieved content instead of its pre-trained—and possibly outdated—knowledge. [Chat LangChain](https://chat.langchain.com/) is a popular Q/A chatbot on LangChain documentation that’s powered by RAG.

### In-context learning (ICL)

With ICL, we adapt the LLM by placing prototype examples in the prompt. “Demonstration through examples” has been shown in multiple studies to be effective. The examples can contain different kinds of information:

- Input and output text only—that is, few-shot learning
- Reasoning traces: adding intermediate reasoning steps; see [Chain-of-Thought](https://arxiv.org/abs/2201.11903) (CoT) prompting
- Planning and reflection traces: adding information that teaches the LLM to plan and reflect on its problem solving strategy; see [ReACT](https://arxiv.org/abs/2210.03629)

Multiple other strategies to modify the prompts exist, and the [Prompt Engineering Guide](https://www.promptingguide.ai/) contains a comprehensive overview.

### Choosing the right adaptation method

To decide which of the above approaches are suitable for a particular application, you should consider various factors: the model capability required for the pursued task, cost of training, cost of inference, types of datasets, etc. The flowchart below summarizes our recommendations to assist you in choosing the right LLM adaptation method.

  
  

<img src="/media/image-20240811220657.png" alt="20240811220657" style="zoom:60%;" />

  

## 長文本模型、微調與增強式生成技術的比較

  

### 長文本模型（Long-Context LLMs, > 128K, e.g. 1M）

  

**優點：**

  

- **直接處理長輸入：** 能够直接處理較長的提示或文件，无需外部擷取。

- **潛在的推理能力提升：** 可能捕捉到輸入中的更多上下文和關係，進一步提升理解能力。

- **減少對外部知識源的依賴：** 在某些情況下，可以直接處理輸入信息，不需要額外的擷取。

  

**缺點：**

  

- **高計算成本：** 相較於傳統的 LLM，訓練和推理的計算成本更高。

- **上下文長度限制：** 雖然比傳統 LLM 更長，但仍存在可處理輸入長度的限制。

- **數據效率：** 可能需要更多的訓練數據來達到最佳性能。

  

### 微調（Fine-Tuning）

  

**優點：**

  

- **適應特定任務：** 可以針對特定任務或領域進行訓練，以提升性能。

- **性能提升：** 通常比直接使用預訓練模型效果更好。

- **潛在的小模型優化：** 有時可以在較小的模型上進行微調，降低計算需求。

  

**缺點：**

  

- **需要標註數據：** 微調通常需要帶有標籤的數據集。

- **過擬合風險：** 如果訓練數據不夠多樣化，可能導致過擬合。

- **計算成本：** 雖然比從頭訓練模型低，但仍需要一定的計算資源。

  

### 增強式生成（RAG）

  

**優點：**

  

- **接入外部知識：** 可以利用大量的外部信息。

- **提升準確度和相關性：** 通過結合外部知識，可以生成更準確、相關的回應。

- **靈活性：** 可以與不同類型的 LLM 和知識源結合使用。

  

**缺點：**

  

- **需要知識庫：** 建立和維護高品質的知識庫可能具有挑戰性。

- **擷取效率：** 從知識庫中高效擷取相關信息至關重要。

- **潛在幻覺問題：** 如果擷取到的信息不準確或誤導，生成的輸出也可能不正確。

  

### 選擇最佳方法

  

最佳方法取決於具體的應用場景、可用資源和期望的性能。

  

- **長文本模型** 適合直接處理長輸入的任務，例如摘要長篇文件或生成代碼。

- **微調** 適用於有大量標註數據且明確性能提升目標的任務。

- **增強式生成** 在需要接入外部知識且 LLM 上下文長度有限的情況下很有價值。

  

在許多情況下，可以結合使用這些技術來達到最佳效果。例如，RAG 系統可以使用長文本模型來處理增強後的輸入，以提升性能。

  
  
  

## RAG vs. Long Context

  

先説結論：

* LLM-4K + RAG > LLM-16K w/o RAG (LLM: Llama2-70B and GPT-43B)

* Llama2-70B-32K + RAG > GPT-3.5-(175B)-16K > Llama2-70B-32K w/o RAG

  
  
  

## Takeaway

  
  
  
  
  
  

| | LLM only (cloud or edge) | Cloud+Edge LLM<br>post arbitration | Cloud+Edge LLM<br/>pre arbitration | Cloud info+<br>edge LLM RAG | Cloud info+<br>edge AutoGPT |

| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- | --------------------------- | --------------------------- |

| Accuracy | 1. Based of pre-trained knowledge, <br>could be out-of-date. <br>2. Hallucination without credible source | 1. Based of pre-trained knowledge, <br/>could be out-of-date. <br/>2. Hallucination without credible source | Worst. <br>Edge LLM error<br>+ cloud LLM error | High | Med-High |

| Cost | Edge:low; Cloud:high | High | Medium | Low | Med |

| Latency | Edge: fast; Cloud: slow | Fast | Fast | Slow? | Slow |

| Self-improve | You don't know you don't know.<br>Ceiling: cloud LLM | Yes, use cloud LLM<br>fine-tune edge LLM | Maybe | Yes | Maybe |

  
  

## Source

  

* [Advanced RAG Techniques: an Illustrated Overview | by IVAN ILIN | Dec, 2023 | Towards AI](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)

  

* https://arxiv.org/html/2312.10997v5

  

  

### RAG

  

**Vanilla RAG case** in brief looks the following way: you split your texts into chunks, then you embed these chunks into vectors with some Transformer Encoder model, you put all those vectors into an index and finally you create a prompt for an LLM that tells the model to answers user’s query given the context we found on the search step.

In the runtime we vectorise user’s query with the same Encoder model and then execute search of this query vector against the index, find the top-k results, retrieve the corresponding text chunks from our database and feed them into the LLM prompt as context.

  
  
  

<img src="/media/image-20231222085752666.png" alt="image-20231222085752666" style="zoom:67%;" />

  

The prompt can look like:

  

```python

def question_answering(context, query):

prompt = f"""

Give the answer to the user query delimited by triple backticks ```{query}```\

using the information given in context delimited by triple backticks ```{context}```.\

If there is no relevant information in the provided context, try to answer yourself,

but tell user that you did not have any relevant context to base your answer on.

Be concise and output the answer of size less than 80 tokens.

"""

  

response = get_completion(instruction, prompt, model="gpt-3.5-turbo")

answer = response.choices[0].message["content"]

return answer

```

  
  
  

Advanced RAG

  

<img src="/media/image-20231222092039359.png" alt="image-20231222092039359" style="zoom:67%;" />

  
  

## Reference

  

Good introduction: https://ai.meta.com/blog/adapting-large-language-models-llms/