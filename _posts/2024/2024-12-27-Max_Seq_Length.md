---
title: Max Sequence Length in LLM
date: 2024-12-27 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-03-26-Transformer_LLM]]"
categories:
  - Science
tags:
  - LLM
---



## 引言

訓練 LLM 會遇到很多參數設定，有些和 model 相關，例如 embedding dimension, layer number。有些和 tokenizer 相關 (tokenizer 和 model 目前還是兩個 files)，例如 vocab size，special token id。有些和訓練相關，例如 batch size, number of epoch。有些和 dataset 相關。

最常見也最混淆的是 max_seq_length 或是其他不同的名稱代表 temporal domain 的 boundary 或限制。原因是 tokenizer, model, training parameter 都會用到，而且有從屬關係。

Inferencing 比較簡單，只有 prompt context window

|            | Long text         | Structured Dataset           | Tokenizer  | Model            | Training length | Inf. length             |
| ---------- | ----------------- | ---------------------------- | ---------- | ---------------- | --------------- | ----------------------- |
| Stage      | Data prep         | Data prep                    | Train/inf. | Train/Inf.       | Train           | Inf.                    |
| Alias      | --                | --                           | max_length | max_pe           | max_length      | context_window          |
| Limitation | 切成 tokenizer size | 小等於 tokenizer, truncate 大於部分 | 小於 model   | fixed with model | 小等於 tokenizer   | 和 tokenizer 無關！小於 model |

**迷思**
1. LLM 和 input sequence 無關。input sequence 可以很短像是 "你好"，或是一個幾十萬或百萬字的長篇小説 "西游記"。 
	- **No,** input sequence 不能超過 model maximum positional embedings 和 tokenizer 的 max_seq_length.  不然會被 truncate.
2. LLM 和 output sequence 無關。可以產生短的 response, 或是非常長的 output 像是小説 
	- **No,** 一次輸出通常無法超過一定的長度？input + output < positional embeddings.  因此長的小説輸出一般需要分成多段，而且要自己維持一致性。 
3. 或是非常長的多輪對話
	- **Yes,** 多次輸出當然可以輸出非常長。context window 是 kv cache 的長度限制。不過 context window 的長度也會小於 potional embeddings.  超過 positional embeddings 的前文 LLM 無法記住。

**結論 :**
1. LLM 的 max position encode/embedding 是關鍵。需要夠長。不然就是 bottleneck
2. **只要設定 tokenizer 的 max_length** <= max position embedding.   tokenizer 的 max_length 由訓練的時間決定。不一定要和 model position encoder 一樣，不然訓練時間和會非常久，同時需要非常多的内存。



以下分成三個部分:  (1) Model Training/Fine-tuning;  (2) Inferencing; (3) Data preparation.  

## Model Training/Fine-Tuning
### Model (`max_position_embeddings`)

一個迷思是 transformer model 是和 input sequence 無關，有點類似 RNN。input sequence 可以很短像是 "你好"，或是一個幾十萬或百萬字的長篇小説 "西游記"。

雖然原則上如此，但是實務上有兩個因素限制 sequence 長度：
- position encoding/embedding.  這是所有 tokens 第一步要加上的位置資訊。一般有個固定的長度。例如 Huggingface 的 model 都會定義在 config.json 的 `max_position_embeddings` 參數。雖然 rotary position encode 可以把這個數字變成相對位置，並且延長。但還是有一個上限。
- 第二個問題是 transformer model 的計算量和内存都是和 sequence 的平方正比。如果太長會造成 GPU 和内存的問題。同樣也有 eviction, compression, 或是其他架構 (linear attention) address 這個問題。不過還是一個問題。

### `max_seq_length` of Tokenizer and SFTTrainer

**應該只要設定一個就 OK?  比較好是設定在 tokenizer,  可以用於 training or inferencing!**

The `max_seq_length` parameter in the tokenizer and the `max_seq_length` parameter in the `SFTTrainer` serve related but distinct purposes in the context of training and fine-tuning language models. Here's a breakdown of their roles and differences:

---

### **1. `max_seq_length` in the Tokenizer**

- **Purpose**: Defines the maximum sequence length (number of tokens) that the tokenizer can handle in a single input.
- **Usage**:
    - It **truncates or pads the input sequences** to ensure they have a consistent length suitable for the model.
    - The tokenizer ensures that no input sequence exceeds this limit by either:
        - Truncating sequences longer than `max_seq_length`.
        - Padding sequences shorter than `max_seq_length` (usually with a special padding token).
        
        ```python
        tokenized_input = tokenizer("Your input text here", max_length=max_seq_length, truncation=True, padding="max_length")
        ```
        
    - **Model Constraint**: The tokenizer's `max_seq_length` should not exceed the model's maximum positional embedding size (defined during model architecture design).

---

### **2. `max_seq_length` in the `SFTTrainer`**

- **Purpose**: Defines the maximum sequence length used during training within the `SFTTrainer`.
- **Usage**:
    - It determines how much of the tokenized data (**produced by the tokenizer**) is passed to the model during training.
    - When set, the trainer ensures that sequences exceeding this length are truncated, and shorter ones are padded (based on the tokenizer settings).

```python
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        #max_seq_length=custom_args.max_seq_length,  可以設定在 tokenizer 就 OK?
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        callbacks=[EfficiencyCallback()],
    )
```

### **Interaction Between the Two**

- **Tokenization First**: Inputs are tokenized based on the tokenizer's `max_seq_length`.
- **Trainer's Enforcement**: The `SFTTrainer` processes the tokenized inputs, ensuring they fit within its defined `max_seq_length`.

**Example Scenario**:

- Tokenizer `max_seq_length`: 1024 (can process up to 1024 tokens).
- Trainer `max_seq_length`: 512 (uses only the first 512 tokens for training).

**This setup ensures that tokenized inputs exceeding 512 tokens are truncated at the training stage, but the tokenizer can still process and generate longer sequences for other use cases (e.g., inference).**   雖然實務上很奇怪，因爲如果 model 沒有被長的 sequence 訓練過，卻在 inference 用更長的 sequence?  也許在 fine-tuning 是 OK?

**應該只要設定一個就 OK?  比較好是設定在 tokenizer,  可以用於 training or inferencing!**

### **Best Practices**

- **Alignment**:
    - Ensure `SFTTrainer.max_seq_length` ≤ `Tokenizer.max_seq_length`.
- **Efficiency**:
    - Adjust `SFTTrainer.max_seq_length` based on computational resources and the target application.
- **Consistency**:
    - Use the same `max_seq_length` for both tokenizer and trainer if the model needs to handle the same maximum context length during training and inference.

Let me know if you need further clarification or code examples!


| **Feature**           | **`max_seq_length` in Tokenizer**                                     | **`max_seq_length` in SFTTrainer**                                   | **`max_position_embeddings` in Model**                         |
| --------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Scope**             | Affects how input sequences are tokenized.                            | Affects how tokenized sequences are used in training.                | Determines the maximum sequence length the model can handle.   |
| **Enforcement Point** | During tokenization.                                                  | During dataset preparation for training.                             | During model computation (fixed by model architecture).        |
| **Responsibility**    | Prepares inputs up to a specified length.                             | Determines maximum sequence length for model training.               | Ensures the model can encode positional information correctly. |
| **Relationship**      | Typically aligns with or exceeds the trainer's `max_seq_length`.      | Should not exceed the tokenizer's `max_seq_length`.                  | Acts as an upper bound for both tokenizer and `SFTTrainer`.    |


---

### **Example with Specific Models**

| **Feature**           | **`max_seq_length` in Tokenizer** | **`max_seq_length` in SFTTrainer** | **`max_position_embeddings`** |
| --------------------- | --------------------------------- | ---------------------------------- | ----------------------------- |
| **GPT-3 (175B)**      | Up to 2048 tokens                 | ≤ tokenizer's `max_seq_length`     | 2048                          |
| **BERT (Base)**       | Up to 512 tokens                  | same above                         | 512                           |
| **T5 (11B)**          | Up to 1024 tokens                 | same above                         | 1024                          |
| **Meta LLaMA 2 (7B)** | Up to 4096 tokens                 | same above                         | 4096                          |

---

### **Explanation of `max_position_embeddings`**

- Refers to the maximum number of positional embeddings a model can handle.
- Defines the model's hard limit for input sequence length.

Let me know if you'd like any additional adjustments!


### Inferencing

Inferencing 也有 max_length.  

```python
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Include attention mask
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        do_sample=True  # Enable sampling for probabilistic generation
    )
```

The **`max_length`** parameter in the **`model.generate()`** function specifies the **maximum number of tokens** that the model can generate in its output sequence, including the input tokens.

### **Purpose of `max_length`**

1. **Controls Output Length**:
    
    - Ensures that the generated text does not exceed the specified maximum number of tokens.
    - This prevents the generation process from running indefinitely, especially for autoregressive models like GPT, which generate text one token at a time.
2. **Includes Input Tokens**:
    
    - The total length of the generated sequence includes the input tokens (from the `input_ids`) and the tokens generated by the model.
    - For example:
        - If the input has 100 tokens and `max_length=150`, the model can generate up to 50 additional tokens.
3. **Avoids Resource Overuse**:
    
    - Limits the computational and memory resources required for generation.
    - Useful when generating multiple sequences in batch mode or when working with large models.
4. **Aligns with Context Window**:
    
    - Must respect the model's **`max_position_embeddings`** (the maximum context window of the model).
    - For instance, if a model's context window is 2048 tokens, `max_length` should be ≤ 2048.

---

### **Impact of `max_length` During Generation**

1. **Stopping Generation**:
    
    - If the model reaches `max_length` during generation, it stops even if other stopping criteria (like an `<EOS>` token) are not met.
2. **Premature Truncation**:
    
    - If `max_length` is set too low, the output may get truncated before the model completes a coherent response.
3. **Interaction with Other Parameters**:
    
    - **`do_sample=True`**:
        - Enables stochastic sampling, meaning the model may explore diverse possibilities up to `max_length`.
    - **`num_return_sequences`**:
        - Generates multiple outputs, each adhering to `max_length`.
    - **`no_repeat_ngram_size`**:
        - Prevents repetitive patterns within the `max_length` span.
    - **`attention_mask`**:
        - Ensures only valid tokens from the input are attended to during generation, influencing how the model uses `max_length`.

---

### **Example Scenario**

#### **Input:**

```python
input_text = "Explain the concept of gravity."
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
```

#### **Generation with `max_length` = 50:**

```python
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    do_sample=True,
    temperature=0.7,
)
```

**Result:**

- The model generates text with a total of **50 tokens**, combining input tokens and generated tokens.

---

### **Tips for Setting `max_length`**

1. **Task-Dependent**:
    
    - For summarization: Keep `max_length` small.
    - For storytelling or long-form generation: Use a larger `max_length`.
2. **Model's Context Limit**:
    
    - Ensure `max_length` is within the model’s **`max_position_embeddings`** (e.g., 2048 for GPT models).
3. **Balance Coherence and Length**:
    
    - Large values for `max_length` may result in verbose or off-topic responses.
    - Smaller values may truncate coherent completions.

Let me know if you'd like more details on how `max_length` interacts with other generation parameters!

The **`max_length`** in the **`model.generate()`** function acts as a **one-time output limitation** for each call to `generate()`. It defines the total length of tokens (including input tokens) that can be generated in a single run.

**Is this a one-time output limitation? If doing multi-run conversation, will the model takes the previous conversation from the kv cache to generate the output? or it needs the user to take care of it?**

For multi-turn conversations or extended interactions, handling the conversation context depends on the **model's capabilities** and **how you manage the input tokens**. Here's a detailed breakdown:

---

### **1. Does the Model Automatically Use Previous Context (KV Cache)?**

- **Autoregressive Models (e.g., GPT, LLaMA):**
    - The model itself does **not retain context** across multiple `generate()` calls unless you explicitly include the previous conversation in the input.
    - The **KV cache** (key-value cache) is used to speed up generation during a single call to `generate()`. It helps the model avoid recomputing attention for tokens that have already been processed.
    - However, the KV cache is **not persistent across multiple calls** to `generate()`. You, as the user, need to manage the conversation history.

---

### **2. How to Handle Multi-Turn Conversations?**

For multi-turn conversations:

1. **User Responsibility**:
    
    - You must append previous user and model responses to the input sequence to provide the model with the conversation history.
    - Example:
        
        ```python
        # Maintaining conversation history
        conversation_history = "User: Hello!\nAI: Hi there! How can I assist you today?\n"
        user_input = "What is the weather like today?"
        prompt = conversation_history + f"User: {user_input}\nAI:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate a response
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            do_sample=True,
        )
        # Decode and update conversation history
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        conversation_history += f"User: {user_input}\nAI: {response}\n"
        ```
        
2. **Managing Context Window**:
    
    - As the conversation grows, the total token length of the history might exceed the model’s **context window** (`max_position_embeddings`).
    - To handle this:
        - **Truncate** older parts of the conversation:
            
            ```python
            conversation_history = conversation_history[-model.config.max_position_embeddings:]
            ```
            
        - Retain the most relevant or recent exchanges.

---

### **3. Key Points About KV Cache**

- **Purpose of KV Cache**:
    
    - During a single `generate()` call, the KV cache stores intermediate results (keys and values from attention layers) for tokens that have already been processed. This speeds up generation by avoiding redundant computations.
- **Scope of KV Cache**:
    
    - The KV cache is **reset between calls** to `generate()`. If you want the model to retain context across multiple turns, you must explicitly include the previous context in the input tokens.

---

### **4. Benefits of User-Managed Context**

- **Fine-Grained Control**:
    - You can decide how much of the conversation history to include, truncate, or prioritize.
- **Flexibility**:
    - Tailor the history management to the application (e.g., keeping the full history for chatbots or focusing on the most recent exchanges for task-oriented systems).

---

### **5. Summary**

- **`max_length`**: A one-time limit for a single call to `generate()`.
- **KV Cache**:
    - Speeds up generation within a single call but is not persistent across multiple calls.
    - You must manage conversation history explicitly for multi-turn interactions.
- **User Responsibility**:
    - Include past conversation turns in the input tokens to simulate context retention.
    - Manage token limits to stay within the model’s `max_position_embeddings`.

Would you like an example implementation for a conversation manager?


**Prompt caching** and the **key-value (KV) cache** are related concepts, but they are distinct in purpose and implementation. Let’s break down each concept and clarify the confusion:

---
**What is prompt caching? How it differs from kv cache tokens?**

注意:  各大 AI 公司推出的 promp caching 只是 store prompt tokens,  和 KV cache store 的 K, V 完全沒關係。


### **1. What Is Prompt Caching?**

**Prompt caching** refers to **storing preprocessed (tokenized) versions of prompts or conversation history** for reuse in subsequent processing. It avoids the need to re-tokenize the same input repeatedly, which can save time and computational resources.

- **What It Stores**:
    
    - The **tokenized input IDs** for the conversation history (or prompts).
    - This may also include associated metadata, such as attention masks or truncated tokens.
- **Purpose**:
    
    - To avoid re-tokenizing parts of the input that remain unchanged across multiple turns of a conversation.
    - Useful for multi-turn conversations or when frequently reusing specific prompts.
- **How It Works**:
    
    - Tokenized prompts are cached in memory or disk and reused directly when needed.
    - New input is appended to the cached tokenized sequence, rather than starting from raw text.
- **Example**:
    
    ```python
    # Cache tokenized history
    tokenized_history = tokenizer("User: Hello!\nAI: Hi there!", return_tensors="pt")
    
    # Tokenize new input and append to cached history
    new_input = tokenizer("User: What's the weather like today?", return_tensors="pt")
    cached_input_ids = torch.cat([tokenized_history["input_ids"], new_input["input_ids"]], dim=1)
    ```
    

---

### **2. What Is KV Cache?**

The **KV cache** (key-value cache) stores **intermediate attention representations** during a single forward pass of the model. It is specific to transformer models and is used to speed up generation by avoiding redundant computations.

- **What It Stores**:
    
    - The keys and values computed in the self-attention mechanism for already-processed tokens.
    - These representations are specific to the current session or `generate()` call.
- **Purpose**:
    
    - To accelerate autoregressive generation by reusing attention values for past tokens without recomputing them.
- **Scope**:
    
    - The KV cache only exists during a single generation run. It is discarded once the generation is complete and does not persist across multiple calls to `generate()`.

---

### **Key Differences Between Prompt Caching and KV Cache**

|Feature|Prompt Caching|KV Cache|
|---|---|---|
|**What It Stores**|Preprocessed tokenized inputs (e.g., `input_ids`, attention masks).|Keys and values from the self-attention layers.|
|**Purpose**|Avoid re-tokenizing unchanged inputs across sessions.|Speed up token generation during a single session.|
|**Scope**|Persistent across multiple sessions or conversations.|Temporary, only exists during one `generate()` call.|
|**Reuse**|Allows appending new inputs to cached tokenized history.|Cannot append new inputs; recalculates attention for updated sequences.|
|**Efficiency**|Saves time during tokenization.|Saves time during autoregressive generation.|

---

### **3. How Do Prompt Caching and KV Cache Work Together?**

They address different bottlenecks in the pipeline:

1. **Prompt Caching**:
    
    - Reduces preprocessing overhead by reusing tokenized inputs.
    - Helps manage multi-turn conversations efficiently by appending new tokens to cached history.
2. **KV Cache**:
    
    - Speeds up token-by-token generation within a single call to `generate()` by reusing attention representations.
    - Works in tandem with prompt caching but is session-specific.

---

### **4. Why Prompt Caching Does Not Store KV Cache?**

The KV cache is tied to the internal states of the model for the specific tokens processed during a session. It cannot be reused across sessions because:

- **Dynamic Nature**:
    - KV cache representations are dependent on the tokens processed in a specific context. Appending new tokens would require recomputing attention values to account for the new context.
- **Stateless Transformer Design**:
    - Transformers are designed to be stateless between calls to `generate()`. The input tokens (and prompt cache) must provide all the context needed for the model to compute the KV cache anew.

---

### **5. How to Implement Prompt Caching?**

For multi-turn conversations, you can use prompt caching like this:

#### **Example Implementation**

```python
# Initialize cached tokenized history
cached_history = None

# Function to append new input to cached history
def update_cached_history(cached_history, new_input, tokenizer, max_position_embeddings):
    # Tokenize new input
    new_input_tokens = tokenizer(new_input, return_tensors="pt")["input_ids"]

    if cached_history is None:
        # First turn: Initialize history
        cached_history = new_input_tokens
    else:
        # Append to cached history
        cached_history = torch.cat([cached_history, new_input_tokens], dim=1)

    # Truncate if history exceeds max context length
    cached_history = cached_history[:, -max_position_embeddings:]
    return cached_history

# Usage
cached_history = update_cached_history(cached_history, "User: Hello!", tokenizer, max_position_embeddings=2048)
cached_history = update_cached_history(cached_history, "AI: Hi there!", tokenizer, max_position_embeddings=2048)
cached_history = update_cached_history(cached_history, "User: What's the weather like today?", tokenizer, max_position_embeddings=2048)

# Generate response
outputs = model.generate(
    input_ids=cached_history,
    max_length=100,
    do_sample=True,
    temperature=0.7,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### **6. Summary**

- **Prompt caching** stores pre-tokenized inputs (e.g., conversation history) for reuse in subsequent turns.
- **KV cache** speeds up token generation within a single call to `generate()` but cannot persist across sessions.
- For multi-turn conversations, **prompt caching** allows you to efficiently manage and reuse context without needing to re-tokenize everything.
- However, you cannot store or reuse the KV cache between different calls.

Let me know if you'd like further clarification or examples!


## Data Preparation


### 1. Strucutured Data/Dataset (例如 Alpaca 或是 Wikitext2 Dataset)

一般是已經 multiple data records.  使用時 Tokenizer 會根據設定自動 truncate and pad tokenized data record based on tokenizer max length. 

---

### **1. Dataset: `alpaca-cleaned`**

- **Structure**:
    
    - The `alpaca-cleaned` dataset contains instruction-response pairs formatted for instruction-tuned language models. Each entry typically has:
        - **Instruction**: The task description.
        - **Input**: (Optional) Additional context or input.
        - **Output**: The expected response.
    - These fields are often concatenated into a single sequence for training.
- **Problem**:
    
    - If the combined text length of `Instruction + Input + Output` exceeds `max_seq_length`, it would lead to truncation unless splitting or other preprocessing is applied.

---

### **2. What Happens in the Code?**

In the provided code:

```python
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

- **`formatting_prompts_func`**:
    - This function is where preprocessing (like concatenation, truncation, or splitting) might happen.
    - If `formatting_prompts_func` does not handle splitting, then the entire sequence (instruction + input + output) is passed as a single entry to the `SFTTrainer`.

---

### **3. Role of `SFTTrainer`**

When the dataset is passed to the `SFTTrainer`:

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    ...
)
```

#### **How Does `SFTTrainer` Handle Long Examples?**

1. **Tokenizer’s Role**:
    
    - Inside the `SFTTrainer`, the `tokenizer` processes the text from the `dataset_text_field` (`"text"` in this case).
    - The tokenizer applies **truncation** if the sequence exceeds `max_seq_length`.
    - Example behavior:
        
        ```python
        tokenized = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length"
        )
        ```
        
    - **Effect**: The **end** of the sequence (tokens beyond `max_seq_length`) is discarded.
2. **No Automatic Splitting**:
    
    - The `SFTTrainer` and tokenizer do not split long sequences into multiple chunks.
    - Instead, they **truncate the input** to fit within `max_seq_length`.

---

### **4. What Happens Without Splitting?**

If no explicit splitting is performed in `formatting_prompts_func`:

1. **Loss of Information**:
    
    - If the combined text (instruction + input + output) exceeds `max_seq_length`, the tokenizer truncates the excess tokens. This can result in losing parts of the instruction, input, or output.
2. **Training Still Works**:
    
    - The `SFTTrainer` will train the model on truncated sequences.
    - However, the model may not learn effectively because critical parts of the sequence (e.g., output) might be missing.

---

### **5. Who Should Handle Splitting?**

If splitting is necessary (to preserve all parts of the dataset):

- **Responsibility**: It is up to the user to implement this in the `formatting_prompts_func`.
- **Why**: The `SFTTrainer` assumes the dataset has been preprocessed to fit `max_seq_length`.

#### **Example: Add Splitting Logic**

You can modify `formatting_prompts_func` to handle splitting for long examples:

```python
def formatting_prompts_func(examples):
    processed = []
    for example in examples["text"]:
        tokens = tokenizer(example)["input_ids"]
        # Split long examples into chunks
        chunks = [
            tokenizer.decode(tokens[i:i + max_seq_length])
            for i in range(0, len(tokens), max_seq_length)
        ]
        for chunk in chunks:
            processed.append({"text": chunk})
    return processed
```

---

### **6. Why Does It Work Without Splitting?**

Even without explicit splitting, the training does not fail because:

1. **Truncation**:
    - The tokenizer ensures that sequences fit within `max_seq_length`.
2. **Implicit Assumptions**:
    - If critical parts of the sequence (e.g., output) are truncated, the model may underperform, but training proceeds without error.

---

### **7. Best Practices**

If the `alpaca-cleaned` dataset has long sequences:

- Inspect the examples to check if truncation affects important parts (like the output).
- Implement splitting logic if you need to preserve all parts of the text.
- Alternatively, increase `max_seq_length` if your hardware supports longer sequences and the model's `max_position_embeddings` allows it.

---

### **Key Takeaways**

- **Who Splits?**
    - If `formatting_prompts_func` includes splitting logic, it handles this task. Otherwise, no splitting occurs.
- **What Happens Without Splitting?**
    - Sequences are truncated to `max_seq_length` by the tokenizer inside the `SFTTrainer`.
- **Recommendation**:
    - Add splitting logic if truncation leads to the loss of critical information.

Let me know if you'd like further clarification or assistance in adding splitting logic!

### 2. Unstructured Data (例如 Shakespeare 文章)

**基本就是自己要做分句 (基於 tokenizer 的 max length) 成爲 multiple data 和 tokenization.**

In scenarios like the **Shakespeare input.txt** used in projects like **NanoGPT**, the long text sequence is typically converted into multiple smaller data samples to fit within the model's `max_seq_length`. This preprocessing is not done based solely on lines but instead involves strategies like splitting into fixed-length chunks or token-level sliding windows. Here's how it works in detail:

---

### **1. Loading the Long Text**

- The **Shakespeare input.txt** file is loaded as a single, long string of text.
- Example:
    
    ```python
    with open("shakespeare.txt", "r") as f:
        text = f.read()
    ```
    

---

### **2. Tokenization**

- The loaded text is passed through a tokenizer to convert it into numerical tokens (IDs).
- The tokenizer maps each character, word, or subword (depending on the tokenizer type) to a corresponding ID.
- Example:
    
    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(text)["input_ids"]
    ```
    
- This results in a single, long sequence of token IDs.

---

### **3. Splitting the Long Sequence**

Since the entire sequence cannot be fed into the model due to the `max_seq_length` constraint, it is split into smaller chunks:

#### **A. Fixed-Length Splitting**

- The long sequence is split into non-overlapping chunks of size `max_seq_length`:
    
    ```python
    max_seq_length = 512
    chunks = [
        tokens[i : i + max_seq_length]
        for i in range(0, len(tokens), max_seq_length)
    ]
    ```
    
- **Output**:
    - Each chunk is a sequence of exactly `max_seq_length` tokens (except possibly the last chunk, which may be shorter).

#### **B. Sliding Window Splitting**

- If context preservation is important, overlapping chunks are created using a **stride**:
    
    ```python
    stride = 256
    chunks = [
        tokens[i : i + max_seq_length]
        for i in range(0, len(tokens), max_seq_length - stride)
    ]
    ```
    
- **Output**:
    - Chunks overlap by `stride` tokens, retaining some context between consecutive chunks.

#### **C. Line-Based Splitting (Optional)**

- For datasets with structured data (e.g., transcripts, dialogues), splitting by lines or paragraphs might make sense:
    
    ```python
    lines = text.splitlines()
    tokenized_lines = [tokenizer(line)["input_ids"] for line in lines if line.strip()]
    ```
    
- **Output**:
    - Each line becomes a separate data sample. This method is less common for continuous text like Shakespeare because it breaks the natural flow of the text.

---

### **4. Preparing Input-Output Pairs (Optional)**

For tasks requiring input-output pairs (e.g., causal language modeling):

- The model predicts the next token for each token in the sequence.
- Example:
    
    ```python
    inputs = chunks[:-1]  # All tokens except the last one in the chunk
    labels = chunks[1:]   # All tokens except the first one in the chunk
    ```
    
- This setup ensures that the model learns to predict each token in the sequence based on its preceding tokens.

---

### **5. Padding (If Needed)**

- If the final chunk is shorter than `max_seq_length`, padding tokens are added to make it uniform in length:
    
    ```python
    from transformers import DataCollatorForLanguageModeling
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    padded_chunks = collator(chunks)
    ```
    

---

### **6. Batch Creation**

- After splitting and padding, the chunks are grouped into batches for training:
    
    ```python
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(torch.tensor(chunks))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    ```
    

---

### **7. Summary of Process**

1. **Load the Text**: The long text file is loaded into memory.
2. **Tokenize**: The text is converted into a sequence of token IDs using a tokenizer.
3. **Split into Chunks**: The sequence is divided into smaller chunks of size `max_seq_length`, optionally with a sliding window.
4. **Prepare for Training**:
    - For causal language modeling, input-output pairs are created.
    - For fine-tuning, padding is added to shorter sequences.
5. **Batching**: The chunks are grouped into batches for training.

---

### **Why Not Split by Line?**

- Splitting by lines works for structured data but is not ideal for continuous text like Shakespeare:
    - It breaks the natural flow of text and context across lines.
    - Lines can vary greatly in length, making it harder to create uniform data samples.

---

### **Recommended Splitting Strategy for Shakespeare**

For continuous text like Shakespeare, **fixed-length splitting** or **sliding window splitting** is preferred:

- Fixed-length splitting works well for straightforward token prediction tasks.
- Sliding window splitting is useful if you want to retain more context between chunks.

Would you like code examples for a specific splitting strategy?
