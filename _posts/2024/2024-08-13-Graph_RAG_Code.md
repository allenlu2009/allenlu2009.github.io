---
title: Graph RAG Coding
date: 2024-08-13 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Graph
  - LLM
  - RAG
  - Search
---


## 介紹

Graph RAG 包含幾個階段，如下所示。[[2024-07-28-Graph_RAG|Blog]]

**索引階段**

- 前兩個步驟 (source documents, text chunking) 與 RAG 相同
- 實體 (Entity)： (人、地點) 節點 (node)，關係 (relationship)：邊

<img src="/media/image-20240811203003.png" alt="20240811203003" style="zoom:60%;" />

**查詢階段** <img src="/media/image-20240811203300.png" alt="20240811203300" style="zoom:60%;" />

## 微軟 Graph RAG

微軟開源了一個 Graph RAG 並提供了一個流程。我們可以按照 YouTube 影片進行解析，並在 **由狄更斯撰寫的《聖誕頌歌》，羅貫中的《三國演義》，和吳承恩的《西遊記》中的古騰堡數據庫** 中使用它。使用 CPT-4o-mini 模型進行索引的成本約為 $0.13（上下文/輸入標記：520K，生成/輸出標記：80K）。


### 聖誕頌歌

> $ conda create --name graphrag --clone base 
> $ conda activate graphrag 
> $ pip install gragphrag 
> ## Download Christmas Carole 
> $ mkdir -p ./xmas_carole/input
> $ curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./xmas_carole/input/book.txt
> ## Download 三國演義
>  $ mkdir -p ./xmas_carole/input
> $ curl https://www.gutenberg.org/cache/epub/23950/pg23950.txt > ./three_countries/input/book.txt
> > ## Download 西遊記
>  $ mkdir -p ./xiyouji/input
> $ curl https://www.gutenberg.org/cache/epub/23962/pg23962.txt > ./xiyouji/input/book.txt



### 索引階段

> $ python -m graphrag.index --init --root ./xmas_carole 
> $ ## 編輯 ./xmas_carole/.env 並添加 GRAPHRAG_API_KEY=<openai api key\> 
> $ ## 編輯 ./xmas_carole/settings.yaml (1) 設定 llm 模型：gpt-4o-mini； (2) 設定嵌入模型：text-embedding-3-small； (3) 設定 graphml：false 
> $ python -m graphrag.index --root ./xmas_carole

GPT-4o-mini 價格： $0.15 / 1M 輸入標記； $0.6 / 1M 輸出標記。 
聖誕頌歌 (36K words):  0.15 x 0.52 + 0.6 x 0.08 = $0.13
三國演義 (700K words)：0.15 x 2.67 + 0.6 x 0.62 = $0.8. 
西遊記 (800K words)  ：0.15 x 7.04 + 0.6 x 1.36 = $1.9. 
**中文書的 tokens 約為英文的 5-10 倍，1. 是文章長度，2 是中文的 token 是英文的 2-3 倍。選擇更好的 tokenizer 可以減少 2.** 


索引階段將使用來自 OpenAI 的兩個模型。一個用於自然語言處理以理解和分類數據，另一個用於生成嵌入，以幫助有效檢索相關信息。

#### 文本嵌入

- 使用 openai text-embedding-3-small

#### 使用 gpt-4o-mini 解析圖形。

- 提取實體
- 提取關係
- 創建社區摘要

<img src="/media/image-20240811212458.png" alt="20240811212458" style="zoom:60%;" />

## Parquet 內容

使用 VS Code + parquet-viewer 作為插件來查看文件。

最簡單的 parquet：**社區**：僅標題和級別，以及 **關係** <img src="/media/image-20240813224922.png" alt="20240813224922" style="zoom:60%;" /> 

不同的 **community_reports** 是文本的分組 <img src="/media/image-20240813225047.png" alt="20240813225047" style="zoom:60%;" />

**Node parquet：節點包括：組織、人物、事件、地理（位置）** ![[Pasted image 20240814223013.png]]

**Entity parquet：和 node 相同，只是多了 id** 
![[Pasted image 20240814223454.png]]

**Relationship parquet：節點之間的關係（weight, description）（source, target）。** ![[Pasted image 20240814222809.png]]

Text unit parquet：**文本單元**：給文本單元一個 ID。 ![[Pasted image 20240814223913.png]]


## Global Vs. Local Query

In a nutshell: 
- Globel query uses community
- Local query uses node/entity and relationship

**Caveate: no special character (space or Chinese character) in the path!!**
I move the ./xmas_carole and ./lancedb folders from /mnt/g/My Drive/... to /mnt/c/Users/allen/ to solve the problem!

### Query requres global content:  -->  Vector RAG would fail!

> $ python -m graphrag.query --root graphrag/three_countries --method global  
"三國演義的主旨是什麼"

SUCCESS: Global Search Response: 
## 《三國演義》的主旨分析

《三國演義》是一部深刻探討權力鬥爭、忠誠與背叛的歷史小說。其主旨圍繞著三國時期的政治動盪和軍事衝突，通過描繪劉備、曹操、孫權等主要人物的複雜關係，反映了人性中的光明與黑暗面。這些主題在角色的行為和決策中得以體現，特別是在主要人物之間的互動中，展示了個人與集體之間的矛盾和衝突 [Data: Reports (1, 3, 4, 5, 9)]。

### 忠誠與背叛

小說中強調了忠誠與背叛的主題，許多角色的行為和選擇揭示了在權力鬥爭中，個人忠誠的脆弱性和背叛的後果。這些情節不僅增強了故事的戲劇性，也反映了人性中的複雜性 [Data: Reports (2, 3, 12, 19)]。劉備與關羽、張飛之間的兄弟情誼，成為了他們行動的動力，並在動盪的時代中展現了忠義的重要性 [Data: Reports (11, 18)]。

### 智謀與策略

此外，《三國演義》也強調了智謀和策略在戰爭中的關鍵作用。角色如諸葛亮和曹操的計謀和軍事策略對於戰爭的結果起到了關鍵作用，這反映了在動盪時期，智慧和計劃的價值 [Data: Reports (1, 4, 5, 8, 20)]。這些策略不僅影響了軍事行動，也在政治鬥爭中發揮了重要作用，展示了智者如何在困難的環境中尋求生存和勝利的方式 [Data: Reports (6, 14)]。

### 歷史的無常與人性的脆弱

《三國演義》同時反映了歷史的無常和人性的脆弱。許多角色的命運因為政治鬥爭而改變，展現了歷史的殘酷與無常 [Data: Reports (15, 16)]。這一點引發了對於英雄主義的反思，無論是勝利者還是失敗者，最終都難逃歷史的洪流 [Data: Reports (12, 16)]。

### 總結

總的來說，《三國演義》不僅是一部歷史小說，更是一部關於人性、道德和政治的深刻反思。它探討了忠誠與背叛、智謀與策略、以及歷史的無常等主題，對後世的文學和文化產生了深遠的影響 [Data: Reports (1, 19)].

> $ python -m graphrag.query --root graphrag/three_countries --method local " 請說明草船借箭"

SUCCESS: Local Search Response: 
## 草船借箭

草船借箭是三國演義中的一個著名故事，主要講述了蜀漢軍師諸葛亮利用智慧和計謀，成功地從敵軍那裡借取箭矢的過程。這一事件不僅展示了諸葛亮的智謀，也反映了當時戰爭中的策略運用。

### 故事背景

在故事中，諸葛亮面對曹軍的威脅，發現蜀軍缺乏箭矢，無法進行有效的軍事行動。為了應對這一困境，諸葛亮決定採取一個巧妙的計策。他利用草船，假裝蜀軍即將進攻，從而引誘曹軍射箭。

### 計謀的實施

諸葛亮命令製作了許多草人，並將這些草人放置在船上，然後在濃霧中派出這些草船接近曹軍的水域。曹軍見到草船，以為蜀軍來襲，便開始向草船射箭。由於草船的數量眾多，曹軍的箭矢很快就被射光，草船上也收集了大量的箭矢。

### 成功的結果

最終，諸葛亮成功地借到了足夠的箭矢，解決了蜀軍的燃眉之急。這一事件不僅展示了諸葛亮的智慧和果斷，也成為了後世傳頌的經典故事，象徵著以智取勝的軍事策略。

### 文化影響

草船借箭的故事在中國文化中具有深遠的影響，常被用來比喻巧妙地利用環境和敵人的弱點來達成自己的目的。這一故事也被多次改編成戲劇、電影和電視劇，成為三國演義中最具代表性的情節之一。

這一故事的成功，反映了在戰爭中，智慧和策略往往比單純的武力更為重要，並且在面對困難時，靈活的思維和創新的方法能夠帶來意想不到的成功。




# Appendix A: English Version
## Introduction

Graph RAG includes several stages as below.  

**Indexing Phase**
- The first two steps are the same as RAG
- The entity : (people, place) node,  the relationship: edge

<img src="/media/image-20240811203003.png" alt="20240811203003" style="zoom:60%;" />

**Query Phase**
<img src="/media/image-20240811203300.png" alt="20240811203300" style="zoom:60%;" />


## Microsoft GRAPH RAG

Microsoft open source a graph RAG and provide a flow.  We can follow the Youtube video to parse it and use it in **Christmas Carole written by Dickens from Gutenberg Database**.  It's about $0.13 to do the indexing (contextual/input tokens: 520K,  generated/output tokens: 80K) using CPT-4o-mini model.

GPT-4o-mini price:  $0.15 / 1M input tokens;  $0.6 / 1M output tokens.
0.15 x 0.52 + 0.6 x 0.08 = $0.13

>  $ conda create --name graphrag --clone base
>  $ conda activate graphrag
>  $ pip install gragphrag
>  $ mkdir -p ./xmas_carole/input
>  $ curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./xmas_carole/input/book.txt


### Indexing Phase
> $ python -m graphrag.index --init --root ./xmas_carole
> $ ## edit ./xmas_carole/.env  and add GRAPHRAG_API_KEY=\<openai api key\>
> $ ## edit ./xmas_carole/settings.yaml  (1) set llm model: gpt-4o-mini; (2) set embedding model: text-embedding-3-small; (3) set graphml: false
> $ python -m graphrag.index --root ./xmas_carole


The indexing phase will use two models from openai. The indexing phase will use two models from OpenAI: one for natural language processing to understand and categorize the data, and another for generating embeddings that will help in efficiently retrieving relevant information.
#### Text Embedding
- use openai text-embedding-3-small
#### Parse the graph using gpt-4o-mini.
- Extracted entities
- Extracted relationship
- Create community summary

<img src="/media/image-20240811212458.png" alt="20240811212458" style="zoom:60%;" />

## Parquet Content

Use VS Code + parquet-viewer as plugin to view the files.

The much easiest parquet: **communities**: only titles and levels, and **relationship**
<img src="/media/image-20240813224922.png" alt="20240813224922" style="zoom:60%;" />

Different **community_reports** is a grouping of texts
<img src="/media/image-20240813225047.png" alt="20240813225047" style="zoom:60%;" />

**The easiest parquet:  nodes includes: organization, person, event, geo (location), ....  人事時地物**
<img src="/media/image-20240813223545.png" alt="20240813223545" style="zoom:60%;" />

**The 2nd easiest parquet: relationships (weight, description) between nodes (source, target).**
<img src="/media/image-20240813223905.png" alt="20240813223905" style="zoom:60%;" />

The 3rd easist parquet: **text units**: give text units an id.
<img src="/media/image-20240813224350.png" alt="20240813224350" style="zoom:60%;" />

**entities:** it looks like text embedding, more complicated than node.
<img src="/media/image-20240813224603.png" alt="20240813224603" style="zoom:60%;" />

## Global Vs. Local Query

In a nutshell: 
- Globel query uses community
- Local query uses node/entity and relationship

**Caveate: no special character (space or Chinese character) in the path!!**
I move the ./xmas_carole and ./lancedb folders from /mnt/g/My Drive/... to /mnt/c/Users/allen/ to solve the problem!

### Query requres global content:  -->  Vector RAG would fail!

> $ python -m graphrag.query --root ./xmas_carole --method global "What are the top themes in this story?"

SUCCESS: Global Search Response: 

The story prominently features several key themes that intertwine to convey its moral and emotional messages. Below are the most significant themes identified:

#### Transformation
The theme of transformation is central to the narrative, particularly through the character of Ebenezer Scrooge. His evolution from a miserly, isolated individual to a generous and compassionate person is catalyzed by visits from the Ghosts. These encounters prompt him to reflect on his past, present, and future, emphasizing the potential for change in everyone [Data: Reports (4, 39, 40, 43, 28)].

#### Generosity and Compassion
Generosity and compassion are significant themes illustrated through Scrooge's relationships, especially with the Cratchit family. His eventual kindness towards them, particularly in providing for Tiny Tim, highlights the importance of caring for others and the impact of individual actions on the community. This theme is further reinforced by the backdrop of Christmas, which fosters human connection and the spirit of giving [Data: Reports (39, 40, 4)].

#### Wealth and Poverty
The contrast between wealth and poverty is a recurring theme, illustrated through Scrooge's initial indifference to the struggles of the poor, particularly the Cratchit family. This theme critiques societal attitudes towards the impoverished and emphasizes the need for empathy and social responsibility [Data: Reports (30, 19, 14)].

#### Family and Togetherness
Family and togetherness are prevalent themes, especially within the Cratchit household, where love and support are emphasized despite financial struggles. Their Christmas celebration serves as a poignant reminder of the joy and hope that family brings, reinforcing the narrative's message about the significance of familial bonds during the holiday season [Data: Reports (39, 40)].

#### Redemption
The theme of redemption is woven throughout the narrative, particularly through Scrooge's interactions with the Ghosts and his reflections on his life choices. The story illustrates that it is never too late to change and make amends, emphasizing the power of self-reflection and the possibility of a second chance [Data: Reports (4, 43)].

#### Mortality and Legacy
Mortality and the legacy one leaves behind are explored through Scrooge's encounters with the Ghost of Christmas Future. These encounters prompt reflections on the consequences of one's actions and the importance of living a compassionate life [Data: Reports (41, 24)].

#### Social Responsibility
The concept of social responsibility is explored through Scrooge's initial neglect of the poor and his eventual realization of the importance of helping those in need. The story critiques societal attitudes towards the less fortunate, urging readers to consider their moral obligations to support and uplift others [Data: Reports (4, 39)].

In summary, the story encapsulates a rich tapestry of themes that collectively advocate for compassion, transformation, and the importance of human connections, particularly during the festive season. These themes resonate deeply, encouraging readers to reflect on their own lives and the impact of their actions on others.

### Query local content:

> $ python -m graphrag.query --root ./xmas_carole --method local "Who is Scrooge, and what are his main relationships?"

SUCCESS: Local Search Response: 
#### Who is Scrooge?

Ebenezer Scrooge is a central character in Charles Dickens' classic novella "A Christmas Carol." Initially depicted as a cold-hearted miser, Scrooge embodies the negative traits of greed and cynicism, particularly evident in his disdain for Christmas and lack of compassion towards others. His character serves as a representation of the consequences of a life devoid of empathy and generosity. Scrooge's 
transformation is catalyzed by supernatural visitations from various spirits, including the Ghost of Christmas Past, the Ghost of Christmas Present, and the Ghost of Christmas Future, who guide him through a journey of self-reflection and redemption [Data: Entities (26, 10); Relationships (35, 84)].

#### Main Relationships

##### 1. **Bob Cratchit**
Scrooge's relationship with Bob Cratchit, his underpaid clerk, is pivotal in illustrating the contrast between wealth and the struggles of the working class. Initially, Scrooge is indifferent to Bob's hardships, but as the story progresses, he becomes increasingly concerned for Bob's family, particularly for Tiny Tim, Bob's ill son. This evolving relationship signifies Scrooge's shift from a self-centered existence to one that values human connection and compassion, culminating in Scrooge raising Bob's salary and supporting his family [Data: Entities (5, 168); Relationships (35, 168)].

##### 2. **Jacob Marley**
Jacob Marley, Scrooge's deceased business partner, plays a crucial role in Scrooge's transformation. Marley appears as a ghost to warn 
Scrooge about the dire consequences of his miserly life choices. His haunting visit serves as a critical turning point, prompting Scrooge to reflect on his past and consider the implications of his actions for his future. Marley's message emphasizes the importance of compassion and generosity, setting the stage for Scrooge's eventual redemption [Data: Entities (177, 27); Relationships (128, 96)].      

##### 3. **Fred (Scrooge's Nephew)**
Fred, Scrooge's cheerful nephew, embodies the spirit of Christmas and familial love. He consistently invites Scrooge to join in Christmas celebrations, despite Scrooge's reluctance and dismissive attitude towards the holiday. Fred's unwavering optimism and efforts to include Scrooge highlight the stark contrast between their characters, ultimately serving as a reminder of the joy and warmth that family connections can bring [Data: Entities (30, 52); Relationships (107, 52)].

##### 4. **Tiny Tim**
Tiny Tim, the youngest son of Bob Cratchit, becomes a significant figure in Scrooge's transformation. Initially indifferent to the Cratchit family's struggles, Scrooge's concern for Tiny Tim's well-being prompts him to reconsider his views on generosity. The bond that forms between Scrooge and Tiny Tim symbolizes Scrooge's shift towards valuing compassion and human connection, ultimately influencing his journey towards redemption [Data: Entities (205, 168); Relationships (168)].

##### 5. **The Spirits**
The various spirits that visit Scrooge—most notably the Ghost of Christmas Past, the Ghost of Christmas Present, and the Ghost of Christmas Yet to Come—serve as guides in his transformative journey. Each spirit reveals different aspects of Scrooge's life, compelling him to confront his past, understand the joy of the present, and recognize the potential consequences of his future choices. Their influence is instrumental in teaching Scrooge valuable lessons about kindness and the importance of change [Data: Entities (10, 60, 69); Relationships (147, 57, 69)].

In summary, Scrooge's character arc from a miser to a generous individual is intricately tied to his relationships with others, particularly Bob Cratchit, Jacob Marley, Fred, Tiny Tim, and the spirits. These connections highlight the themes of compassion, redemption, and the true spirit of Christmas that Dickens emphasizes throughout the narrative.


> $ python -m graphrag.query --root ./xmas_carole --method **local** "What is Christmas Carole?"

SUCCESS: Local Search Response: 
### Understanding "A Christmas Carol"

"A Christmas Carol" is a novella written by Charles Dickens, first published in 1843. It tells the story of Ebenezer Scrooge, a miserly old man who despises Christmas and all things joyful. The narrative follows Scrooge's transformation after he is visited by the spirits of Christmas Past, Present, and Yet to Come on Christmas Eve. These supernatural encounters lead him to reflect on his life, the impact of his actions on others, and ultimately, to embrace the spirit of Christmas.

### Themes and Significance

The novella is rich with themes such as redemption, compassion, and the importance of community. It emphasizes the values of generosity and kindness, contrasting Scrooge's initial disdain for Christmas with the joy and warmth that the holiday brings to others, particularly the Cratchit family. The story serves as a moral lesson about the consequences of selfishness and the potential for personal change.

### Cultural Impact

"A Christmas Carol" has had a profound influence on Christmas traditions and celebrations. It popularized the idea of Christmas as a time for family gatherings, generosity, and goodwill. The character of Scrooge has become synonymous with the archetype of a miser who undergoes a significant transformation, and the novella has inspired countless adaptations in various forms of media, including films, plays, and musicals.

In summary, "A Christmas Carol" is not just a story about Christmas; it is a timeless tale that explores the human capacity for change 
and the importance of compassion in our lives. Its enduring popularity reflects its universal themes and the way it resonates with the 
spirit of the holiday season.


> $ python -m graphrag.query --root ./xmas_carole --method **global** "What is Christmas Carole?"

SUCCESS: Global Search Response: 
### Overview of "A Christmas Carol"

"A Christmas Carol" is a novella written by Charles Dickens that narrates the transformative journey of Ebenezer Scrooge, a miserly old man who experiences a profound change after being visited by three spirits on Christmas Eve. The story emphasizes themes of redemption, compassion, and the importance of community and generosity during the holiday season [Data: Reports (21, 24, 30, 19, 8)].

### Key Characters and Themes

The novella features several key characters, including Scrooge, the Ghost of Jacob Marley, Bob Cratchit, and Tiny Tim. Each character represents different societal aspects and highlights the impact of one's actions on others. Tiny Tim, in particular, symbolizes hope and innocence amidst poverty, illustrating the emotional depth and societal implications of the narrative [Data: Reports (19, 21, 30, 24, 
8)].

The transformation of Scrooge from a greedy, isolated figure to a generous and caring individual is central to the moral lessons of the story. The narrative critiques social attitudes towards the poor and emphasizes the moral obligation to help those in need, particularly during the festive season [Data: Reports (39, 43, 40)].

### Cultural Impact

"A Christmas Carol" has become a cultural touchstone, inspiring numerous adaptations across various media, including film, theater, and literature. Its themes resonate across generations, making it a vital part of the Christmas tradition in many cultures [Data: Reports 
(21)]. The novella serves as a reminder of the significance of Christmas as a time for generosity and reflection on one's actions, reinforcing the importance of family, love, and community gatherings [Data: Reports (29, 39, 40)].

In summary, "A Christmas Carol" is not only a story about personal transformation but also a critique of societal values, urging readers to embrace compassion and community spirit during the holiday season [Data: Reports (4, 39, 40)].


## Reference

Microsoft example: https://microsoft.github.io/graphrag/posts/get_started/

Good youtube video:  https://www.youtube.com/watch?v=vX3A96_F3FU

Llamaindex knowledge Graph RAG (not very useful) : https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/#graphstore-backed-rag-vs-vectorstore-rag

Medium:  https://medium.com/@vkmauryavk/understanding-retrieval-augmented-generation-rag-vector-based-vs-graph-based-3fe6b90cc92a



