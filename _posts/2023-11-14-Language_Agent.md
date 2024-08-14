---
title: LLM Agent
date: 2023-11-12 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* CoALA (Princeton) [(49) I-X Seminar: Formulating and Evaluating Language Agents with Shunyu Yao - YouTube](https://www.youtube.com/watch?v=qmGu9okiICU&ab_channel=I-X)


## Takeaway

* **LLM -> prompt engineering -> Agent (very hard!) to interact with environment using LLM**



### What is LLM Agent?

- Agent can interact via action with environment.
- LLM agent: Use LLM to interact with environment
- Typical tie to robot



Question? What's the relationship between LLM agent vs. RL agent?



#### Application



#### Key

1. Formulation
2. Evaluation



### Formulation

Use CPU analogy to LLM

<img src="/media/image-20231113161327612.png" alt="image-20231113161327612" style="zoom:33%;" />

#### RL Agent

<img src="/media/image-20231113161441886.png" alt="image-20231113161441886" style="zoom:25%;" />

<img src="/media/image-20231113161456904.png" alt="image-20231113161456904" style="zoom:33%;" />

- Memory: because LLM is "stateless", need memory;  different from RL

   * short-term working memory
   * long-term memory: Episodic (experience);  semantic (knowledge); procedural (LLM, code)

- Action (space):  RL is defined by action space. 

   ＊External actions interact with external environments (grounding) - same as RL agent

   ＊Internal actions interact with internal memories - different from RL agent 

   * Reasoning: read & write working memory (via LLMs)
   * Retrieval: read long-term memory
   * Learning: write long-term memory (or weights)

   <img src="/media/image-20231113162758881.png" alt="image-20231113162758881" style="zoom:33%;" />

- Decision making

  - A language agent chooses actions via decision procedures
    - Split taken actions into decision cycles


<img src="/media/image-20231113163025656.png" alt="image-20231113163025656" style="zoom:33%;" />



#### Framework Application

<img src="/media/image-20231113163539003.png" alt="image-20231113163539003" style="zoom:50%;" />



### Evaluation

1. Interact with physical world / humans: hard to scale
2. Interact with digital simulation (Atari, Mujuco): hard to be practical
3. Interact with digital applications: ?
   1. **Interact with web** (smartphone)
   2. Interact with code (PC, ..)  productivity



#### Webshop: query, modification, and web purchase

<img src="/media/image-20231113165023107.png" alt="image-20231113165023107" style="zoom: 50%;" />

<img src="/media/image-20231113165501828.png" alt="image-20231113165501828" style="zoom:50%;" />

#### Code Interaction (interact with computer)

LLM - auto-regressive way

Human code - interactive way

Lang-Chain is one example?



<img src="/media/image-20231113165824797.png" alt="image-20231113165824797" style="zoom:50%;" />



<img src="/media/image-20231113170921650.png" alt="image-20231113170921650" style="zoom:50%;" />



## Appendix

