VectorRAG vs. GraphRAG
VectorRAG: correlation;  GraphRAG: causality


Vector RAG: similarity.
Graph RAG: relationship   


## Vector RAG v. Graph RAG

Here’s a comparison table that highlights the key differences between Graph RAG [[2024-07-28-Graph_RAG|blog]] and Vector RAG [[2023-12-22-LLM_RAG|blog]] strengths and weaknesses.

| Feature                  | Graph RAG                                   | Vector RAG                                  |
| ------------------------ | ------------------------------------------- | ------------------------------------------- |
| **Data Structure**       | Utilizes knowledge graphs                   | Utilizes vector representations             |
| **Focus**                | **Entity relationships and semantics**      | **Semantic similarity**                     |
| **Retrieval**            | Correlation                                 | Causality                                   |
| **Explainability**       | High; relationships are clear and traceable | Lower; based on proximity in vector space   |
| **Complex Queries**      | Well-suited for complex, relational queries | Effective for simple, thematic queries      |
| **Scalability**          | Can be less scalable with very large graphs | Highly scalable for large unstructured data |
| **Use Cases**            | Healthcare, finance, scientific research    | Document search, content recommendations    |
| **Inference Capability** | Strong; can reason through relationships    | Limited; focuses on similarity              |
| **Data Types**           | Structured and semi-structured data         | Primarily unstructured data                 |

## Difference Between Correlation and Causality

Correlation and causality are related but distinct concepts in statistics and research. Correlation measures the strength and direction of the linear relationship between two variables, while causality refers to the cause-and-effect relationship between variables.

**Correlation** implies that two variables are associated with each other, but it does not necessarily mean that one variable causes the other. A high correlation coefficient indicates a strong relationship, but it does not specify the direction of the relationship or rule out the possibility of a third variable influencing both variables.

**Causality**, on the other hand, implies that one variable directly causes a change in another variable. For causality to be established, certain criteria must be met, such as temporal precedence (the cause must precede the effect), covariation (changes in the cause must be associated with changes in the effect, i.e. explicit relationship), and the absence of plausible alternative explanations.

### Example

Correltaed: articles group based on the same tags: LLM, AI.
Causal: articles group based on the temporal precedence or covariation 

## Mapping to Vector RAG and Graph RAG

The concepts of correlation and causality can be mapped to the differences between **Vector RAG (Retrieval-Augmented Generation)** and **Graph RAG (Retrieval-Augmented Generation)**.

**Vector RAG** primarily relies on vector embeddings and similarity measures to retrieve relevant information and generate outputs. It focuses on the correlation between input queries and retrieved information, aiming to find the most relevant content based on semantic similarity. However, **Vector RAG** may struggle to capture the complex relationships and causal connections between different pieces of information.

On the other hand, **Graph RAG** utilizes graph structures to represent and retrieve information. Graphs allow for the modeling of complex relationships, including causal connections, between entities and concepts. By leveraging the inherent structure of graphs, **Graph RAG** can better capture and reason about the causal relationships between different elements of information.

In summary, while **Vector RAG** is effective in finding correlated information based on semantic similarity, **Graph RAG** has an advantage in capturing and reasoning about causal relationships between entities by leveraging the structure and interconnections within graphs.



## HybridRAG (Simple Concatenation)

Can we combine the strength of both VectorRAG and GraphRAG.   Hybrid RAG provides an example.

### Procedure
1. **Knowledge Graph (KG) Construction (Indexing Phase)**: First, extract entities, relationships, and attributes from financial documents and construct this information into a knowledge graph. The knowledge graph provides a structured way to represent the entities in the documents and their interrelationships.

2. **VectorRAG (Encoding Phase)**: Use a vector database to store the vectorized representations of the documents and retrieve the most relevant document segments based on queries, providing contextual information for the language model.

3. **GraphRAG (Query Phase)**: Utilize the constructed knowledge graph to retrieve relevant nodes (entities) and edges (relationships) based on user queries, and extract subgraphs from the knowledge graph to provide contextual information.

4. **HybridRAG Technique**: Integrate the contextual information retrieved from both VectorRAG and GraphRAG to form a unified context, which is then input into the large language model (LLM) to generate the final answer. Hybrid RAG concatenates these contexts to form a unified context.

HybridRAG has better faithfulness, answer relevance, and context recall, but the context precision is worst compared to GraphRAG and VectorRAG.   It's up to the LLM to generate precision answer.

<img src="/media/image-20240820135651.png" alt="20240820135651" style="zoom:60%;" />




這裡有一個比較表，突顯了圖形 RAG 和向量 RAG 的優勢和劣勢之間的主要差異。

| 特徵       | 圖形 RAG           | 向量 RAG         |
| -------- | ---------------- | -------------- |
| **數據結構** | 利用知識圖譜           | 利用向量表示         |
| **重點**   | 實體關係和語義          | 數據的語義相似性       |
| **檢索**   | 相關性              | 因果關係           |
| **可解釋性** | 高；關係清晰且可追溯       | 較低；基於向量空間中的接近性 |
| **複雜查詢** | 適合複雜的關聯查詢        | 對簡單的主題查詢有效     |
| **可擴展性** | 對於非常大的圖形可能較不具擴展性 | 對大型非結構化數據高度可擴展 |
| **使用案例** | 醫療保健、金融、科學研究     | 文檔搜索、內容推薦      |
| **推理能力** | 強；能夠通過關係進行推理     | 有限；專注於相似性      |
| **數據類型** | 結構化和半結構化數據       | 主要是非結構化數據      |


## 相關性與因果關係的區別

相關性和因果關係是統計和研究中的相關但不同的概念。相關性測量兩個變數之間線性關係的強度和方向，而因果關係則指變數之間的因果關係。

**相關性**意味著兩個變數彼此有關聯，但這並不一定意味著一個變數會導致另一個變數。高相關係數表明強烈的關係，但它並未具體說明這種關係的方向，也無法排除第三個變數影響這兩個變數的可能性。

另一方面，**因果關係**意味著一個變數直接導致另一個變數的變化。要確立因果關係，必須滿足某些標準，例如時間優先（原因必須在結果之前）、共變（原因的變化必須與結果的變化相關）以及不存在可信的替代表述。

## 與向量 RAG 和圖形 RAG 的對應

相關性和因果關係的概念可以映射到 **向量 RAG（檢索增強生成）** 和 **圖形 RAG（檢索增強生成）** 之間的差異。

**向量 RAG** 主要依賴於向量嵌入和相似度測量來檢索相關信息並生成輸出。它專注於輸入查詢和檢索信息之間的相關性，旨在根據語義相似性找到最相關的內容。然而，**向量 RAG** 可能難以捕捉不同信息片段之間複雜的關係和因果聯繫。

另一方面，**圖形 RAG** 利用圖形結構來表示和檢索信息。圖形允許對實體和概念之間複雜關係，包括因果聯繫進行建模。通過利用圖形內在結構，**圖形 RAG** 可以更好地捕捉和推理實體的因果聯係。

總結來說，雖然 **向量檢索增強（Vector RAG）** 在基於語義相似性尋找相關信息方面非常有效，但 **圖形檢索增強（Graph RAG）** 在捕捉和推理實體之間的因果關係方面具有優勢，因為它利用了圖形中的結構和相互連接。

## Hybrid RAG (Simple Concatenation)

Can we combine the strength of both VectorRAG and GraphRAG.   Hybrid RAG provides an example.

### Procedure
1. **知识图谱（KG）构建 (indexing phase)**：首先，从金融文档中提取实体、关系和属性，并将这些信息构建成知识图谱。知识图谱提供了一种结构化的方式来表示文档中的实体及其相互关系。
    
2. **VectorRAG (encoding phase)**：使用向量数据库来存储文档的向量化表示，并根据查询检索最相关的文档片段，为语言模型提供上下文信息。
    
3. **GraphRAG (query phase)**：利用构建好的知识图谱，根据用户查询检索相关的节点（实体）和边（关系），并从知识图谱中提取子图来提供上下文信息。
    
4. **HybridRAG技术**：将VectorRAG和GraphRAG检索到的上下文信息整合起来，形成统一的上下文，然后输入到大型语言模型（LLM）中生成最终的回答。Hybrid RAG concatenate these contexts to form a unified context

HybridRAG has better faithfulness, answer relevance, and context recall, but the context precision is worst compared to GraphRAG and VectorRAG.   It's up to the LLM to generate precision answer.

<img src="/media/image-20240820135651.png" alt="20240820135651" style="zoom:60%;" />



Citations:
[1] https://www.semanticscholar.org/paper/948c74af967d1049150fd6b941618671c3236d0d
[2] https://www.semanticscholar.org/paper/a511c7daa1b961fc8a2e6c9170dddcb2b0f6dcee
[3] https://www.semanticscholar.org/paper/d145aa5b88e00efc119c0e852106f8664e073911
[4] https://pubmed.ncbi.nlm.nih.gov/33044752/
[5] https://www.semanticscholar.org/paper/4df2f3b2d9be8b34ae218409040c26ec019350b2
[6] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9203735/
[7] https://arxiv.org/abs/1907.05990
[8] https://www.semanticscholar.org/paper/7cb7663145818d4c5342a30d3f9a305932fb1e0a

Hybrid RAG:   https://arxiv.org/abs/2408.04948