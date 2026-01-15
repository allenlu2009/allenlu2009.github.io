---
title: Llama3 70B Distributed Inference
date: 2024-10-18 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-02-18-Attn_All_U_Need_Visual]]"
categories:
  - LLM
tags:
  - Attention
  - Distributed_Inference
  - LLM
  - Llama
  - Transformer
---

[[2023-03-26-Transformer_LLM]]
[[2024-10-11-Linear_Attention]]
[[2024-10-10-Attention_Math]]
[[2023-10-21-LLM_Memory]]



## Follow-up

- MOE training, inferencing prefill and generation 的切法
- **Simulator of memory and communication requirement of Huggingface model, and the optimal partition schemes.** 

## Takeaways

一個 GPU 或 NPU 無法處理大模型 (70B, 200B, or > 1000B)，一定需要多個 GPUs 處理。因此原來的 model 以及 data (input/output/intermediate) 就需要被切分。

切分的時候要考慮下面幾件事
1. Training, Fine-tuning, or Inferencing
	1. Training 和 Fine-tuning 的考慮很像:  forward + backward,  fixed mini-batches
	2. Inferrencing 則不同: forward only, dynamic batches,  prefill mode and generation mode.
2. Memory footprint and BW
3. Communication:  high cost!  盡量 leverage 各種 parallelism.  最好是無相關 (batch), 或弱相關。
4. No repeate computation


給定使用場景 (training or inferencing), 最高 priority 是 memory footprint, 因爲如果連 model weights 或是 input/output/intermediate 都放不下，也就不用做了。

Memory footprint and BW 分成 
(1) static memory: weights (最簡單)  
(2) dynamic memory: 
	- training:  最大會是 $Q K^T$ 和 $(seq\_len)^2 \times l \times batch$ 成正比。對於 long context 基本正比于 context 的平方！ 
	- Inferencing:  最大的 pre-fill length 和 training 一樣。但是一般 prompt mode 的 input prompt 不會是 context length,  有機會偷一點。不過 worst case 就是 context 平方。再來的 generation 則是 KV cache $2 \times (seq\_len \times embed) \times l \times batch$.  是 context 的綫性成長。

接下來是 low communication between GPUs.  因此要 explore 各種 parallelism.    

## Transformer Series

<img src="/media/image-20241010115242.png" alt="20241010115242" style="zoom:60%;" />

+ Distributed Inference

<img src="/media/image-20241019083502.png" alt="20241019083502" style="zoom:60%;" />
## Transformer Model Size Vs.  内存 (HBM/SRAM)

大型的 transformer model 需要大量的内存。一般是 HBM 因爲 high bandwidth.  不過因爲價格昂貴，一般都只有 10's GB 内存如下表。

| **AI Chip**              | **Memory Type**     | **HBM Size**   | **Memory Bandwidth** |
| ------------------------ | ------------------- | -------------- | -------------------- |
| **NVIDIA A100**          | HBM2e               | 40/80 GB       | Up to 2 TB/s         |
| **NVIDIA H100**          | HBM3                | 80 GB          | Up to 3.35 TB/s      |
| **NVIDIA V100**          | HBM2                | 32 GB          | 1.13 TB/s            |
| **AMD MI250X**           | HBM2e               | 128 GB         | Up to 3.2 TB/s       |
| **AMD MI100**            | HBM2                | 32 GB          | Up to 1.23 TB/s      |
| **Graphcore IPU-M2000**  | In-Processor Memory | 900 MB per IPU | N/A                  |
| **Google TPU v4**        | HBM                 | 32 GB          | Estimated > 2 TB/s   |
| **Google TPU v3**        | HBM                 | 16 GB          | 600 GB/s             |
| **Cerebras CS-2**        | On-Chip SRAM        | 40 GB          | N/A                  |
| **Intel Habana Gaudi 2** | HBM2e               | 96 GB          | Up to 2.45 TB/s      |
| **Intel Xe**             | HBM2e               | Up to 128 GB   | Up to 4 TB/s         |


To calculate the **memory size needed for a transformer model**, including both the model weights and the **key-value (KV) cache** for attention, we can break it down into two components:

1. **Memory for Model Weights** (including parameters for multi-head attention, feed-forward layers, etc.)
2. **Memory for the KV Cache** (used during inference, especially in autoregressive models)


---

Example Calculation

| 模型名        | 参数量 | 层数, l | 隐藏维度, h | 注意力头数 a | Context |
| ---------- | --- | ----- | ------- | ------- | ------- |
| Llama2-7B  | 7B  | 32    | 4096    | 32      | 4K      |
| Llama2-13B | 13B | 40    | 5120    | 40      | 4K      |
| Llama2-33B | 33B | 60    | 6656    | 52      | 4K      |
| Llama2-70B | 70B | 80    | 8192    | 64      | 8K/128K |

Llama3-70B:
Weight: 140GB
KV Cache: 100GB?
Total: 240GB   

因此需要多個 GPU.   以 A100 40GB 爲例，需要 8 個 GPU (with 240GB memory for other system)
如何把一個 70B model 分到 4 or 8 個 models 就變成非常重要的工作。

<img src="/media/image-20241020001941.png" alt="20241020001941" style="zoom:60%;" />


## 如何切大語言模型?

如何切要考慮下面幾件事
1. Training, Fine-tuning, or Inferencing
2. Memory footprint and BW
3. Communication 
4. No repeate computation

### Training and Fine-Tuning

1. Forward and backward (gradient)
2. Mini-batch together!  一般沒有 batch = 1, 都是事先決定的 fixed batch size.
3. Only prompt mode (full attention), no generation mode:  Q, K, V all together compute, no KV cache
### Inference

1. Forward only
2. Dynamic batching:  batch = 1 (device) or dynamic batch size during inference
3. Prompt mode (full attention):  Q, K, V all together to build the KV cache  
4. Generation mode:  Q is 1 token at time, use KV cache

### A summary Table.

| **Partitioning Strategy**                                                                                                            | **Training/Inferecing**                                                                                                                                                                                           | **Memory**                                                                                             | Communication                                                                                                                                                                                   | Others                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Batch Parallelism** (e.g. 10 batch each GPU)                                                                                       | - Good for training with fixed batch<br>- NG for inferencing (batch=1, dynamic batch)                                                                                                                             | - Each GPU has complete weights, no memory BW saving                                                   | - Zero                                                                                                                                                                                          | - Often combined with methods                                                                     |
| **Token Parallelism** <br>(e.g. 10 tokens each GPU)                                                                                  | - NG for inference (token=1 in generation)<br>- NG for attention<br>- OK for MLP                                                                                                                                  | - Each GPU has complete MLP weights, no memory BW saving                                               | - A lot for attention<br>- Zero for MLP                                                                                                                                                         | - Not use for dense model<br>- **Use for MOE generation?**                                        |
| **Layer Parallelism** (e.g. 10 layers each GPU)                                                                                      | - Both training and inferencing<br>- Pipeline latency<br>- Can overlap computation and communication, increasing efficiency.<br>                                                                                  | - Each GPU only need to handle partial layers                                                          | - Limited, only at input/output                                                                                                                                                                 | - possible solution <br>- latency and pipeline bubbles (idle GPUs) concerns, both forward/backwar |
| **Tensor Parallelism**<br>(- Efficiently splits large weight matrices (e.g., attention, feedforward layers) across multiple GPUs.  ) | - both training and inferencing<br>                                                                                                                                                                               | - Reduces per-GPU memory usage for model parameters and kv cache  <br>- Allows handling larger models. | - **High communication overhead between GPUs for each layer, especially for large matrix operations.**  <br>- **Requires synchronization after every layer, leading to potential bottlenecks.** | **- Most important method**                                                                       |
| **Combination (Tensor + Layer Parallelism)**                                                                                         | **- Good for Inferencing**<br>- Reduces memory footprint for activations and weights.  <br>- Works well for models with many layers, like transformers.                                                           | - Tensor and Layer parallelism save memory BW<br>                                                      | - Increased complexity in managing communication and data flow between pipeline stages.<br>                                                                                                     | - Pipeline bubbles can still occur, especially with smaller batch sizes.                          |
| **Combination (Tensor + Layer + Batch Parallelism)**                                                                                 | **- Good for Training**<br>- Allows efficient handling of extremely large models by combining the benefits of all partitioning techniques.  <br>- Optimizes memory usage for weights, activations, and gradients. | - Tensor and Layer parallelism save memory BW and intermediate results<br>                             | - Very high implementation complexity.  <br>- Requires careful tuning of communication and synchronization to avoid bottlenecks.                                                                | - Difficult to manage and debug, especially with heterogeneous hardware.                          |



### Key Approaches for Partitioning:

#### 1. **Tensor Parallelism (Model Parallelism)**
   - Tensor parallelism splits the operations within individual layers (like matrix multiplications in attention and feedforward layers) across multiple GPUs.
   - For example, in **multi-head attention**, the computations for different heads can be split across GPUs. Similarly, the matrix multiplications for the feed-forward layers can be divided among GPUs.
   - If you're using 8 GPUs, the operations can be split such that each GPU performs a portion of the total matrix multiplication, reducing memory consumption and speeding up the computations.

   **How to partition**:
   - Partition large matrix multiplications (such as in attention layers) across the 8 GPUs.
   - Each GPU holds a shard of the weight matrices for the attention and feedforward layers.
   - Intermediate activations are computed locally, and GPUs communicate to exchange the necessary data to compute the final output.

#### 2. **Pipeline Parallelism**
   - In pipeline parallelism, the model is divided into sequential segments, and each segment is assigned to a different GPU.
   - For example, with 8 GPUs, you can split the model into 8 parts, with each GPU handling one segment of the model.
   - The forward and backward passes through the model are performed in stages, so GPU 1 computes the first few layers and passes the output to GPU 2, which computes the next few layers, and so on.
   - This reduces the memory load on each GPU, as they don’t need to store the full model.

   **How to partition**:
   - Divide the model layers across the 8 GPUs. For example, with 70 billion parameters, if the model has 80 layers, you could assign 10 layers to each GPU.
   - Synchronize the inputs and outputs between stages using pipeline synchronization techniques.
   - The GPUs work in parallel, processing mini-batches in a pipeline, where the first GPU starts processing the next mini-batch before the last GPU has finished processing the previous batch.

#### 3. **Data Parallelism**
   - Data parallelism involves copying the entire model to each GPU, but each GPU processes a different portion of the data (e.g., different mini-batches).
   - After each forward and backward pass, the gradients are averaged across all GPUs to ensure consistent updates.

   **How to partition**:
   - Use **data parallelism** in combination with tensor or pipeline parallelism. For instance, within each tensor-parallel or pipeline stage, split the data across GPUs and compute the loss and gradients locally.
   - Gradients are then synchronized between GPUs (using techniques like **gradient accumulation**) to ensure consistent weight updates across all GPUs.

#### 4. **Zero Redundancy Optimizer (ZeRO)**
   - ZeRO, a strategy used in **DeepSpeed**, helps reduce memory usage by partitioning optimizer states, gradients, and model parameters across GPUs.
   - This reduces the memory footprint per GPU and allows for training larger models.

   **How to partition**:
   - With **ZeRO Stage 2 or Stage 3**, partition the model states and gradients across the GPUs. Each GPU only needs to store a fraction of the total model parameters and gradients.
   - Synchronize the partitions during forward and backward passes.

---

### Summary of Partitioning Strategies:


A table summarizing the pros and cons of different partitioning strategies for running **LLaMA 70B** on 8 GPUs, considering **tensor parallelism**, **pipeline parallelism**, and **data parallelism**:

| **Partitioning Strategy**                              | **Pros**                                                                                                                                                                                                                                | **Cons**                                                                                                                                                                                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tensor Parallelism**                                 | - Efficiently splits large weight matrices (e.g., attention, feedforward layers) across multiple GPUs.  <br>- Reduces per-GPU memory usage for model parameters.  <br>- Allows handling larger models.                                  | - **High communication overhead between GPUs for each layer, especially for large matrix operations.**  <br>- **Requires synchronization after every layer, leading to potential bottlenecks.**                                                                    |
| **Pipeline Parallelism (10 layer for each GPU**        | - Reduces memory requirements for activations by splitting the model into sequential stages.  <br>- Allows deeper models to run on memory-limited GPUs.  <br>- Can overlap computation and communication, increasing efficiency.        | - Introduces pipeline bubbles (idle GPUs) during forward/backward passes, especially with smaller batch sizes.  <br>- More complex to implement, as it requires careful synchronization across pipeline stages.                                                    |
| **Data Parallelism**                                   | - Easy to implement and scale across multiple GPUs.  <br>- Minimizes inter-GPU communication during forward pass.  <br>- Works well with large batch sizes, improving efficiency.                                                       | **- Replicates the model on each GPU, increasing memory usage for weights (not feasible for very large models without tensor/pipeline parallelism).**  <br>- Requires all-reduce communication for gradient synchronization, which can be costly for large models. |
| **Combination (Tensor + Data Parallelism)**            | - Balances memory and computational load.  <br>- Tensor parallelism reduces memory footprint for weights, while data parallelism reduces the gradient synchronization overhead.  <br>- Scales well across many GPUs.                    | - Complex communication between GPUs, especially during training.  <br>- Communication overhead still present, particularly for large model sizes like LLaMA 70B.                                                                                                  |
| **Combination (Pipeline + Data Parallelism)**          | - Reduces memory footprint for activations and weights.  <br>- Works well for models with many layers, like transformers.  <br>- Good for handling large models with longer sequences.                                                  | - Increased complexity in managing communication and data flow between pipeline stages.  <br>- Pipeline bubbles can still occur, especially with smaller batch sizes.                                                                                              |
| **Combination (Tensor + Pipeline + Data Parallelism)** | - Allows efficient handling of extremely large models by combining the benefits of all partitioning techniques.  <br>- Optimizes memory usage for weights, activations, and gradients.  <br>- Can balance load effectively across GPUs. | - Very high implementation complexity.  <br>- Requires careful tuning of communication and synchronization to avoid bottlenecks.  <br>- Difficult to manage and debug, especially with heterogeneous hardware.                                                     |

### Summary of Partitioning Strategies:

1. **Tensor Parallelism**: Ideal for reducing memory load from large matrix operations but requires efficient GPU communication to minimize latency.
2. **Pipeline Parallelism**: Helps distribute the model layers across GPUs, reducing activation memory usage, but can introduce pipeline bubbles (idle time) between stages.
3. **Data Parallelism**: Simple to implement but duplicates model weights across GPUs, which can increase memory consumption.
4. **ZeRO Optimization**: Reduces memory overhead for optimizer states and gradients, enabling larger models or batch sizes, but introduces some communication complexity.
5. **Activation Checkpointing**: Reduces memory needed for activations but increases computation time, best used when memory is the primary bottleneck. 

Each strategy has its strengths and weaknesses, and in practice, **hybrid approaches** (e.g., combining tensor and pipeline parallelism with activation checkpointing) are often used to maximize memory and computational efficiency.



When running a large model like the **LLaMA 70B** on **8 GPUs**, you need to carefully partition the tasks to balance the workload and ensure efficient memory and compute usage. Common strategies to partition large models across multiple GPUs involve **model parallelism**, **data parallelism**, and **pipeline parallelism**. For a model as large as LLaMA 70B, a combination of these techniques is typically used, particularly **tensor parallelism** (model parallelism at the level of individual layers) and **pipeline parallelism**.
### Example: Partitioning LLaMA 70B Model Across 8 GPUs

#### Step 1: **Tensor Parallelism for Layer Computations**
   - Suppose LLaMA 70B has approximately **70 billion parameters** and each layer involves large matrix multiplications in the attention and feedforward layers.
   - The model’s large weight matrices (in both attention and feedforward layers) are split across 8 GPUs.
   - Each GPU performs only a portion of the computations for each layer, and the results are aggregated across the GPUs.

#### Step 2: **Pipeline Parallelism for Layer Distribution**
   - Assume LLaMA 70B has **80 transformer layers**.
   - You can split the model into **8 pipeline stages**, with each GPU handling 10 layers. Each mini-batch is processed sequentially by each GPU in the pipeline.
   - GPU 1 processes layers 1–10, GPU 2 processes layers 11–20, and so on until GPU 8 processes layers 71–80.

#### Step 3: **Data Parallelism Across Mini-batches**
   - After partitioning the model using tensor and pipeline parallelism, you can apply **data parallelism** across multiple GPUs, where each GPU processes a different mini-batch.
   - For example, during training, each GPU might work on a different part of the dataset while synchronizing gradients across GPUs after each forward and backward pass.

#### Step 4: **Memory Optimization with ZeRO**
   - By using **ZeRO Stage 2 or 3**, you can further reduce memory usage by partitioning the optimizer states and gradients across the GPUs.
   - Each GPU only stores a portion of the optimizer states and gradients, reducing memory requirements without sacrificing performance.

---

### Practical Considerations:
1. **Communication Overhead**: With tensor and pipeline parallelism, there will be communication overhead between GPUs, especially when synchronizing activations and gradients. Techniques like **NCCL** (NVIDIA Collective Communications Library) can help reduce this overhead.
2. **Batch Size**: With pipeline parallelism, ensure that your batch size is large enough to fully utilize all GPUs. Smaller batch sizes may result in idle GPUs, reducing efficiency.
3. **Memory and Compute Balance**: Ensure that the model partitioning balances both memory and compute workload across GPUs. Too much load on any single GPU can create bottlenecks.

---

### Frameworks to Assist:
- **DeepSpeed**: Provides support for ZeRO optimization and hybrid parallelism (tensor, pipeline, and data parallelism).
- **Megatron-LM**: Provides implementations for tensor parallelism and pipeline parallelism, especially for large language models like GPT and LLaMA.
- **PyTorch with Distributed Data Parallel (DDP)**: Useful for managing data parallelism across multiple GPUs.

By using a combination of **tensor parallelism**, **pipeline parallelism**, and **data parallelism**, and leveraging memory-efficient techniques like **ZeRO**, you can efficiently partition and run the LLaMA 70B model on 8 GPUs.

### Step-by-Step Partitioning of LLaMA 70B:

To partition the **LLaMA 70B** model tasks across **8 GPUs**, with the new **KV cache memory** requirements calculated (41.94 GB per GPU for a sequence length of 128,000), let's reconsider the entire partitioning strategy.

#### 1. **Memory Budget Per GPU**:

- **Total memory per GPU**: 80 GB (high-speed HBM).
- **KV cache per GPU**: 41.94 GB (as calculated previously for 128,000 sequence length and 80 layers).
- This leaves us with: Remaining memory per GPU=80 GB−41.94 GB=38.06 GB\text{Remaining memory per GPU} = 80 \, \text{GB} - 41.94 \, \text{GB} = 38.06 \, \text{GB}Remaining memory per GPU=80GB−41.94GB=38.06GB for weights, activations, and other overhead.

#### 2. **Memory for Model Weights**:

- The **LLaMA 70B** model has **70 billion parameters**.
- Using **FP16 precision** (2 bytes per parameter): Memory for model weights=70×109×2 bytes=140 GB\text{Memory for model weights} = 70 \times 10^9 \times 2 \, \text{bytes} = 140 \, \text{GB}Memory for model weights=70×109×2bytes=140GB
- Using **tensor parallelism** (splitting the weight matrices across 8 GPUs): Memory per GPU for weights=140 GB8=17.5 GB per GPU\text{Memory per GPU for weights} = \frac{140 \, \text{GB}}{8} = 17.5 \, \text{GB per GPU}Memory per GPU for weights=8140GB​=17.5GB per GPU
- **Remaining memory after weights**: Remaining memory=38.06 GB−17.5 GB=20.56 GB\text{Remaining memory} = 38.06 \, \text{GB} - 17.5 \, \text{GB} = 20.56 \, \text{GB}Remaining memory=38.06GB−17.5GB=20.56GB This remaining memory can be used for activations and overhead.

#### 3. **Memory for Activations and Overheads**:

- Each forward and backward pass generates activations that need to be stored temporarily during training or inference.
- With **20.56 GB** of memory remaining per GPU, this should be enough for activations, especially with **activation checkpointing** (a technique that reduces memory usage by recomputing activations during backpropagation).

#### 4. **Partition Strategy**:

To efficiently use the available memory and compute resources, we will use a combination of **tensor parallelism**, **pipeline parallelism**, and **data parallelism**:

- **Tensor Parallelism**:
    - Split large weight matrices within each layer across all 8 GPUs. This reduces the per-GPU memory load for model parameters to **17.5 GB**.
- **Pipeline Parallelism**:
    - The LLaMA 70B model has **80 transformer layers**. We can divide these layers into **8 pipeline stages**, with **10 layers per GPU**. Each GPU will be responsible for a sequential block of 10 layers.
- **Data Parallelism**:
    - To scale further, apply **data parallelism** across the 8 GPUs. Each GPU processes a different portion of the mini-batch during training, while gradient synchronization happens across GPUs.

#### 5. **Final Setup**:

- **Tensor Parallelism** ensures that large operations, like matrix multiplications, are split across GPUs.
- **Pipeline Parallelism** divides the 80 layers into **8 chunks** (10 layers per GPU), reducing the memory required for activations.
- **Data Parallelism** processes different parts of the mini-batch on each GPU, synchronized using techniques like **gradient accumulation**.

### Summary of Partition:

- **KV Cache per GPU**: 41.94 GB.
- **Model weights per GPU**: 17.5 GB (after tensor parallelism).
- **Remaining memory for activations and overhead**: 20.56 GB.
- **Layer partitioning**: 10 layers per GPU (pipeline parallelism).
- **Compute parallelism**: Tensor and data parallelism across the 8 GPUs.

This partitioning scheme allows the **LLaMA 70B** model to run efficiently on **8 GPUs**, each with **80 GB of HBM** memory and **FP16 precision**, with support for long sequence lengths (up to **128,000 tokens**) by appropriately managing the KV cache and model weights.

### Key Parameters:

- **LLaMA 70B**: 70 billion parameters.
- **8 GPUs**, each with **80 GB** HBM.
- **FP16 precision**: 2 bytes per parameter (since FP16 = 16 bits = 2 bytes).

### Memory Requirement for Weights:

1. **Model size**:
    
    - The model has **70 billion parameters**.
    - Memory per parameter in **FP16**: **2 bytes**.
    - Total memory for model weights: Memory for weights=70×109×2 bytes=140 GB
    $$
    \text{Memory for weights} = 70 \times 10^9 \times 2 \, \text{bytes} = 140 \, \text{GB}
    $$

1. **Memory per GPU (just for weights)**:
    
    - With **8 GPUs**, the total weight memory needs to be distributed across GPUs.
    - If we use **tensor parallelism** (splitting each layer’s operations across GPUs), the weights can be equally divided across the 8 GPUs.
    - Memory per GPU for weights: 140 GB8=17.5 GB per GPU
    -$$\frac{140 \, \text{GB}}{8} = 17.5 \, \text{GB per GPU}$$
    140GB​=17.5GB per GPU

Thus, only **17.5 GB** of memory per GPU is used to store weights, leaving ample memory for activations, the KV cache, and other overhead.

2. **Memory per layer for the KV cache:**

- $d_{\text{model}} = 8192$
- $\text{seq\_len} = 128,000$
- **FP16 precision**, so 
- $$\text{dtype}_{\text{KV cache}} = 2 \, \text{bytes}$$

The memory required for the **KV cache** per layer can be calculated as follows:

$$\text{Memory per layer for KV cache} = 2 \times \text{seq\_len} \times d_{\text{model}} \times \text{dtype}_{\text{KV cache}}$$​
Substitute the values:
$$
\text{Memory per layer for KV cache} = 2 \times 128,000 \times 8192 \times 2
$$
Now, calculate the result:

$$
\text{Memory per layer for KV cache} = 4,194,304,000 \, \text{bytes} = 4.194 \, \text{GB per layer}
$$

#### Memory for all 80 layers:

Now, multiply by the number of layers:

$$\text{Total Memory for KV cache} = 80 \times 4.194 \, \text{GB} = 335.52 \, \text{GB}
$$
Total Memory for KV cache=80×4.194GB=335.52GB


### Step-by-Step Partitioning of LLaMA 70B:

#### 1. **Memory Budget Per GPU**:

- **Total memory per GPU**: 80 GB (high-speed HBM).
- **KV cache per GPU**: 41.94 GB (as calculated previously for 128,000 sequence length and 80 layers).
- This leaves us with: 
$$\text{Remaining memory per GPU} = 80 \, \text{GB} - 41.94 \, \text{GB} = 38.06 \, \text{GB}$$for weights, activations, and other overhead.

#### 2. **Memory for Model Weights**:

- The **LLaMA 70B** model has **70 billion parameters**.
- Using **FP16 precision** (2 bytes per parameter): 
- $$\text{Memory for model weights} = 70 \times 10^9 \times 2 \, \text{bytes} = 140 \, \text{GB}$$
- Using **tensor parallelism** (splitting the weight matrices across 8 GPUs): 
- $$\text{Memory per GPU for weights} = \frac{140 \, \text{GB}}{8} = 17.5 \, \text{GB per GPU}$$
- **Remaining memory after weights**: 
- $$\text{Remaining memory} = 38.06 \, \text{GB} - 17.5 \, \text{GB} = 20.56 \, \text{GB}$$This remaining memory can be used for activations and overhead.

#### 3. **Memory for Activations and Overheads**:

- Each forward and backward pass generates activations that need to be stored temporarily during training or inference.
- With **20.56 GB** of memory remaining per GPU, this should be enough for activations, especially with **activation checkpointing** (a technique that reduces memory usage by recomputing activations during backpropagation).

#### 4. **Partition Strategy**:

To efficiently use the available memory and compute resources, we will use a combination of **tensor parallelism**, **pipeline parallelism**, and **data parallelism**:

- **Tensor Parallelism**:
    - Split large weight matrices within each layer across all 8 GPUs. This reduces the per-GPU memory load for model parameters to **17.5 GB**.
- **Pipeline Parallelism**:
    - The LLaMA 70B model has **80 transformer layers**. We can divide these layers into **8 pipeline stages**, with **10 layers per GPU**. Each GPU will be responsible for a sequential block of 10 layers.
- **Data Parallelism**:
    - To scale further, apply **data parallelism** across the 8 GPUs. Each GPU processes a different portion of the mini-batch during training, while gradient synchronization happens across GPUs.

#### 5. **Final Setup**:

- **Tensor Parallelism** ensures that large operations, like matrix multiplications, are split across GPUs.
- **Pipeline Parallelism** divides the 80 layers into **8 chunks** (10 layers per GPU), reducing the memory required for activations.
- **Data Parallelism** processes different parts of the mini-batch on each GPU, synchronized using techniques like **gradient accumulation**.

### Summary of Partition:

- **KV Cache per GPU**: 41.94 GB.
- **Model weights per GPU**: 17.5 GB (after tensor parallelism).
- **Remaining memory for activations and overhead**: 20.56 GB.
- **Layer partitioning**: 10 layers per GPU (pipeline parallelism).
- **Compute parallelism**: Tensor and data parallelism across the 8 GPUs.

This partitioning scheme allows the **LLaMA 70B** model to run efficiently on **8 GPUs**, each with **80 GB of HBM** memory and **FP16 precision**, with support for long sequence lengths (up to **128,000 tokens**) by appropriately managing the KV cache and model weights.


## Appendix

Let's take a look of 1-layer of transformer:

Attention 部分：
<img src="/media/image-20241019083903.png" alt="20241019083903" style="zoom:60%;" />


Here’s a more structured version of the explanation and formula breakdown for calculating the memory size needed for a transformer model, including both **weights** and **KV cache size**:



### Memory for Transformer Model (Weights and KV Cache)

#### 1. **Attention Layer Weights**

For the attention mechanism, we have the following parameters:
- $W_Q$(Query weights)
- $W_K$(Key weights)
- $W_V$(Value weights)
- $W_O$(Output weights)

Including bias for each, the total number of parameters for the attention layer is:

$$
ATTN_{\text{params}} = 4 \times d_{\text{model}}^2 + 4 \times d_{\text{model}}
$$

Where:
- $d_{\text{model}}$is the hidden size (embedding size).
- The factor of 4 accounts for the 3 weights for$Q$,$K$, and$V$, plus the output weight$W_O$, along with biases for each.

The memory required for attention weights, in bytes, is:

$$
\text{Memory for Attn Weights (in bytes)} = (4 \times d_{\text{model}}^2 + 4 \times d_{\text{model}}) \times \text{dtype}_{\text{weight}}
$$

Where:
- $\text{dtype}_{\text{weight}}$is the number of bytes per parameter (e.g., 4 bytes for FP32, 2 bytes for FP16).

---

#### 2. **KV Cache (Key and Value Cache)**

The **KV cache** is used during inference, storing the key and value vectors for each layer and maintaining these over the entire sequence.

For each layer, the number of parameters required for the KV cache is:

$$
\text{KV cache}_{\text{params}} = 2 \times d_{\text{model}} \times \text{context\_len}
$$

Where:
- The factor of 2 accounts for both **key** and **value**.
- $\text{context\_len}$ is the length of the sequence.

The memory required for the KV cache in bytes is:

$$
\text{Memory for KV Cache (in bytes)} = 2 \times d_{\text{model}} \times \text{context\_len} \times \text{dtype}_{\text{KV cache}}
$$

Where:
- $\text{dtype}_{\text{KV cache}}$ is the number of bytes per parameter (e.g., 4 bytes for FP32, 2 bytes for FP16).

---

#### 3. **Feedforward Layer Weights**

The feedforward layer in a transformer typically consists of two linear layers, where the hidden size of the intermediate layer is 4 times the embedding size.

The number of parameters for the feedforward network (ignoring biases) is:

$$
FFN_{\text{params}} = 8 \times d_{\text{model}}^2 + 5 \times d_{\text{model}}
$$

Where:
- $8 \times d_{\text{model}}^2$ comes from the two weight matrices in the feedforward layer.
- $5 \times d_{\text{model}}$ comes from the additional overhead of embedding dimensions.

---

#### 4. **Layer Normalization (Layer Norm)**

The transformer uses layer normalization with gain and bias for each layer. Each normalization layer requires 2 parameters for both gain and bias.

The total number of parameters for layer normalization is:

$$
LN_{\text{params}} = 4 \times d_{\text{model}}
$$

Where:
- The factor of 4 accounts for two layer norm layers with both gain and bias.

---

#### 5. **Total Parameters for One Transformer Layer**

The total number of parameters for one transformer layer, including attention, feedforward, and layer normalization, is:

$$
\text{Total Layer Params} = 12 \times d_{\text{model}}^2 + 13 \times d_{\text{model}}
$$

The total memory required for the weights (in bytes) is:

$$
\text{Memory for Weights (in bytes)} = (12 \times d_{\text{model}}^2 + 13 \times d_{\text{model}}) \times \text{dtype}_{\text{weight}}
$$

---

#### 6. **Total Memory for KV Cache**

The total memory required for the KV cache (in bytes) for each layer is:

$$
\text{Memory for KV Cache (in bytes)} = 2 \times d_{\text{model}} \times \text{context\_len} \times \text{dtype}_{\text{KV cache}}
$$

---

### 7. **Total Memory Formula (Weights + KV Cache)**

The total memory required for the transformer model, including weights and the KV cache, across all layers is:

$$
\text{Total Memory (in bytes)} = L \times \left( \text{Memory for Weights} + \text{Memory for KV Cache} \right)
$$

Where:
- $L$ is the number of transformer layers.

### 8. **Note on Different Data Types for Weights and KV Cache**:

In **memory-limited cases**, the data type for the **weights** may differ from that of the **KV cache**. For example:

- The weights might use **low-precision formats** such as **4-bit quantization** to save memory, where:
    
  $\text{dtype}_{\text{weight}} = 0.5$ bytes per parameter
  
- Meanwhile, the KV cache might use **higher precision**, such as **16-bit** or **8-bit precision**, with:
    
    $\text{dtype}_{\text{KV cache}} = 2 \, \text{bytes} \quad \text{(for FP16)}$
    
    or
    
    $\text{dtype}_{\text{KV cache}} = 1 \, \text{byte} \quad \text{(for INT8)}$.

This allows for memory optimization by using **lower precision** for the model weights while maintaining **higher precision** for the KV cache during inference, especially in scenarios where performance and memory constraints are critical.

---

### 




#### 1. **Memory for Weights**:
$$
\text{Memory for Weights} = (12 \times 768^2 + 13 \times 768) \times 4 \, \text{bytes}
$$

#### 2. **Memory for KV Cache**:
$$
\text{Memory for KV Cache} = 2 \times 768 \times 512 \times 4 \, \text{bytes}
$$




figure of GPU DRAM



### GPT/Llama 總參數量：

一層的 transformer block:

$$W_i^{Q,K,V},W_{1,2}, b_{1,2} = 4(d_{model})^2+4 d_{model} + 8(d_{model})^2 + 5 d_{model} + 4d_{model}  = 12(d_{model})^2 + 13 d_{model}$$

$$n_{layers}$$**多層以及加上$$W_e, W_p$$總參數量：**

$$n_{vocab}d_{model}+n_{ctx}d_{model}+n_{layers} \times (12 (d_{model})^2+ 13 d_{model})$$





## Source

*  Under the Hood of Llama 3.1 70B:  https://whatdhack.medium.com/under-the-hood-of-llama-3-1-70b-distributed-inference-8b3c03886f22
