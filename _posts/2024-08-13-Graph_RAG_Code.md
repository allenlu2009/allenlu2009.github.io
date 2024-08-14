---
title: Graph RAG
date: 2024-07-28 23:10:08
categories:
  - Language
tags:
  - LLM
  - RAG
  - Graph
typora-root-url: ../../allenlu2009.github.io
---

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
>  $ mkdir -p ./ragtest/input
>  $ curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./ragtestg/input/book.txt


### Indexing Phase
> $ python -m graphrag.index --init --root ./ragtest
> $ ## edit ./ragtest/.env  and add GRAPHRAG_API_KEY=\<openai api key\>
> $ ## edit ./ragtest/settings.yaml  (1) set llm model: gpt-4o-mini; (2) set embedding model: text-embedding-3-small; (3) set graphml: false
> $ python -m graphrag.index --root ./ragtest


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

The much eaist parquet: **communities**: only titles and levels, and **relationship**
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
I move the ./ragtest and ./lancedb folders from /mnt/g/My Drive/... to /mnt/c/Users/allen/ to solve the problem!

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

> $ python -m graphrag.query --root ./ragtest --method local "Who is Scrooge, and what are his main relationships?"

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


> $ python -m graphrag.query --root ./ragtest --method **local** "What is Christmas Carole?"

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


> $ python -m graphrag.query --root ./ragtest --method **global** "What is Christmas Carole?"

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



