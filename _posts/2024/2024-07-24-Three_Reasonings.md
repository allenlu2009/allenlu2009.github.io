---
title: Math AI - 演繹推理和合情推理
date: 2024-07-24 09:28:08
typora-root-url: ../../allenlu2009.github.io
description: Deductive Reasoning, Inductive Reasoning, and Plausible Reasoning
categories:
  - Science
tags:
  - Bayesian
  - ML
---


## 從亞里士多德到機器學習的推理演變

### 演繹推理的歷史 (Deductive Reasoning), base of Math

演繹推理的根源可以追溯到古希臘，亞里士多德在他的著作中奠定了這種邏輯方法的基礎。亞里士多德對邏輯學的貢獻，尤其是他發展的三段論，建立了演繹推理的框架。在三段論中，結論是從兩個給定或假設的前提（大前提和小前提）中得出的。例如，如果所有人都是凡人（大前提），蘇格拉底是人（小前提），那麼蘇格拉底是凡人（結論）。這種推理形式以其邏輯結構為特點，確保如果前提為真，結論必然為真。

若 A 為真 -> 則 B 為真。     若 B 為假，則 A 為假。

### 歸納推理的歷史 (Inductive Reasoning), base of Physics

歸納推理，不同於其演繹對應物，其作為一種正式化的方法出現得要晚得多。其歷史發展可以追溯到17世紀初弗朗西斯·培根的作品。培根強調經驗觀察和系統實驗，奠定了歸納推理的基礎，這涉及從具體觀察中得出一般結論。例如，觀察到太陽每天從東方升起，可以得出太陽總是從東方升起的普遍結論。因此，歸納推理從具體到一般，但它本質上涉及一定程度的不確定性和概率。

若 A 為真 -> 則 B 為真。     若 B 為真，則 A 為真。  邏輯上並不成立

### 合情推理 (Plausible Reasoning) 的歷史與貝葉斯推理, based on AI/ML  

喬治·波利亞（George Pólya，1887–1985）是一位匈牙利數學家，以其在問題解決、數學教育和啟發式方法方面的工作而聞名。他對推理的貢獻在於人們如何接近和解決問題的背景中尤為相關。他強調了啟發式方法的重要性——這些是指導問題解決過程的經驗法則或策略。波利亞的工作突顯了推理往往不是純粹的演繹，而涉及**合乎邏輯的推理**，其中直覺和經驗扮演著重要角色。

愛德溫·詹斯（Edwin Jaynes，1922–1998）是一位物理學家和統計學家，以其對概率論及其哲學意涵的研究而聞名。他的貢獻顯著推進了對不確定情況下推理的理解，特別是通過貝葉斯推斷的視角。在他具影響力的作品中，尤其是書籍《概率論：科學邏輯》（2003），詹斯主張**概率應被視為邏輯的一個延伸。他認為，概率代表了對命題真實性的信念或合乎邏輯性的程度，而不僅僅是事件發生頻率。

**貝葉斯推斷**：詹斯強烈倡導貝葉斯方法，這種方法允許根據新證據更新信念。他強調合乎邏輯的推理涉及使用先前知識和證據來對不確定情況做出推斷。這種方法與人類在現實世界中的推理方式密切相關，在那裡信息是不完整且不確定的。


## 推理原則
### 演繹推理的原則

演繹推理基於結論邏輯地從前提中得出的原則。它在一個框架內運作，其中前提與結論之間的關係是必然的。如果前提為真，則得出的結論必然為真，這提供了一種100％準確的推理形式。這種確定性使演繹推理成為數學和形式邏輯等領域中的一種強大工具，在這些領域中，論證的有效性至關重要。

### 歸納推理的原則

歸納推理雖然強大，但不提供與演繹推理相同的確定性。它涉及基於具體觀察和經驗進行推廣。歸納推理的準確性是概率性的，而非確定性的。例如，在觀察某地區的天鵝都是白色之後，可以得出所有天鵝都是白色的結論。然而，這一結論可能會被發現一隻黑天鵝所推翻。因此，歸納推理是可修正的，因為它基於的觀察模式可能隨著新證據的出現而改變。

### 合情推理 (Plausible Reasoning) 的原則  

合情推理，旨在量化不確定性和管理概率，其歷史根源可追溯到18世紀統計學家和牧師托馬斯·貝葉斯的工作。貝葉斯開發了一種基於新證據更新假設概率的方法，今天稱為貝葉斯推理。這種方法結合了先驗知識和新數據來計算不同結果的可能性。貝葉斯推理成為現代統計學和機器學習的基石，提供了一個在新證據下更新信念的正式框架。

若 A (prior) 為真 -> 則 B (evidence) 為真。     若 B 為真，則 A 更可能為真。  


### 小結論
#### **演繹推理**：

- **定義**：演繹推理是從一般原則或前提出發，得出特定結論的過程。如果前提為真，則結論必須也為真。
- **範例**：所有人類都是必死的。蘇格拉底是一個人類。因此，蘇格拉底是必死的。
- **性質**：這是一種自上而下的方法，在前提正確的情況下可以保證結論的確定性。

#### **歸納推理**：

- **定義**：歸納推理是從具體例子或觀察中得出一般結論的過程。這些結論是可能的，但並不確定。
- **範例**：我見過的每一隻天鵝都是白色的。因此，所有天鵝可能都是白色的。
- **性質**：這是一種自下而上的方法，暗示可能性，而非確定性。

#### **合情推理**：

- **定義**：合情推理是指看起來似乎真實但未必得到證明的推理。這是一種我們經常在缺乏所有事實時仍需要做出決策或假設時使用的推理方式。
- **範例**：如果你知道大多數冬天有咳嗽的人都有感冒，而某人在十二月咳嗽，你可能會合理地推斷他們有感冒。
- **性質**：這種推理更靈活和創造性，涉及猜測、類比和臆測。它基於看起來可能或合乎邏輯的事物，即使它並不在邏輯上得到保證。

### **比較**：

- **確定性**：演繹推理提供確定性，歸納推理提供可能結論，而合情推理更關注於即使面對不確定性時，看起來合理或可能的事物。
- **使用場景**：演繹和歸納推理更為剛性和正式，而合情推理由於日常問題解決、科學發現和假設形成中通常使用，其中並非所有信息都可用。

演繹推理與歸納推理之間的區別很明顯。而歸納推理與合情推理之間的區別則不是很清楚。以下是更多解釋。

## 歸納推理與合情推理之間的區別

一句話：合情推理承認不確定性並利用概率表述。而歸納推理期望找出一般規則（例如牛頓定律，相對論），即使那些規則是從有限證據中得出的。

我們實際上可以將**歸納推理**視為一種更特定類型的合情推理。原因如下：

- **合情推理**：使用合乎邏輯的機率，承認結論不是百分之百確定。它涉及基於先前知識、直覺或部分信息進行再現，通常沒有嚴謹證據。
    
- **歸納推理**：這是一個更正式的過程，人們觀察具體實例並試圖將其概括為一條規則或理論。儘管該規則永遠無法百分之百確定（因為它基於觀察而非純邏輯），但目標是盡量接近可靠結論。

從這個意義上說，歸納推理解釋可以被視為一種合理性的子集，其可行性基於所觀察到模式，其目標是尋找可普遍化規則，即使在某些情況下未必能保證成立。

另一方面，合情推理解釋涵蓋了更廣泛的一系列思維過程，包括那些你可能沒有足夠數據形成一般規則，但仍需要根據看起來最有可能的信息做出決策。


### 合情推理及其在機器學習中的作用

合情推理，特別是通過貝葉斯推理的角度，通過引入概率方法填補了演繹和歸納推理之間的空隙。**在機器學習領域，這種概率推理允許隨著新數據的出現不斷更新和完善模型。它增強了機器學習算法的穩健性，使其能夠從數據中學習，進行預測，並適應新信息。**這一方法構成了許多現代機器學習技術的基礎，這些技術依賴概率模型來管理不確定性並改進決策。

總之，推理的演變從亞里士多德的演繹邏輯到培根的歸納方法，最終到貝葉斯的合情推理，展示了向越來越複雜的方法理解和解釋世界的軌跡。這一歷史進程中的每一步都為現代科學探究和技術進步提供了強大的工具，特別是在機器學習領域。通過結合演繹推理的確定性、歸納推理的經驗豐富性和合情推理的靈活性，我們能夠以更大的信心和精確度駕馭複雜的數據豐富環境。





## Appendix


## The Evolution and Principles of Reasoning: From Aristotle to Machine Learning

### History of Deductive Reasoning

The roots of deductive reasoning trace back to ancient Greece, where Aristotle laid the foundation of this logical approach in his works. Aristotle's contributions to logic, particularly through his development of syllogism, established the framework for deductive reasoning. In syllogism, a conclusion is drawn from two given or assumed propositions (premises). For example, if all humans are mortal (major premise), and Socrates is human (minor premise), then Socrates is mortal (conclusion). This form of reasoning is characterized by its logical structure, ensuring that if the premises are true, the conclusion must also be true.

### History of Inductive Reasoning

Inductive reasoning, unlike its deductive counterpart, emerged as a formalized method much later. Its historical development can be traced back to the works of Francis Bacon in the early 17th century. Bacon's emphasis on empirical observation and systematic experimentation laid the groundwork for inductive reasoning, which involves drawing general conclusions from specific observations. For instance, observing that the sun rises in the east every morning leads to the general conclusion that the sun always rises in the east. Inductive reasoning, therefore, moves from the specific to the general, but it inherently involves a degree of uncertainty and probability.

### History of Plausible Reasoning and Bayesian Inference

George Pólya (1887–1985) was a Hungarian mathematician known for his work in problem-solving, mathematics education, and heuristics. His contributions to reasoning are particularly relevant in the context of how people approach and solve problems.  He emphasized the importance of heuristics—rules of thumb or strategies that guide problem-solving processes. Pólya's work highlighted that reasoning is often not purely deductive but involves **plausible reasoning** where intuition and experience play significant roles.

Edwin Jaynes (1922–1998) was an physicist and statistician known for his work on probability theory and its philosophical implications. His contributions significantly advanced the understanding of reasoning under uncertainty, particularly through the lens of Bayesian inference.  In his influential work, particularly the book "Probability Theory: The Logic of Science" (2003), Jaynes argued that **probability should be viewed as an extension of logic.** He posited that probabilities represent a degree of belief or plausibility about the truth of propositions, rather than merely frequencies of events.
    
**Bayesian Inference**: Jaynes was a strong advocate for Bayesian methods, which allow for updating beliefs based on new evidence. He emphasized that plausible reasoning involves using prior knowledge and evidence to make inferences about uncertain situations. This approach aligns closely with how humans often reason in real-world contexts, where information is incomplete and uncertain.

### Principles of Deductive Reasoning

Deductive reasoning is based on the principle that conclusions logically follow from premises. It operates within a framework where the relationship between premises and conclusion is one of necessity. If the premises are true, the conclusion derived must be true, providing a form of reasoning that is 100% accurate. This certainty is what makes deductive reasoning a powerful tool in fields like mathematics and formal logic, where the validity of arguments is paramount.

### Principles of Inductive Reasoning

Inductive reasoning, while powerful, does not offer the same level of certainty as deductive reasoning. It involves making generalizations based on specific observations and experiences. The accuracy of inductive reasoning is probabilistic rather than certain. For example, after observing that swans in a particular region are white, one might conclude that all swans are white. However, this conclusion can be overturned by the discovery of a single black swan. Thus, inductive reasoning is subject to revision and is less certain because it is based on observed patterns that may change with new evidence.

### Principles of Plausible Reasoning

Plausible reasoning, which seeks to quantify uncertainty and manage probabilities, finds its historical roots in the work of Thomas Bayes, an 18th-century statistician and clergyman. Bayes developed a method to update the probability of a hypothesis based on new evidence, known today as Bayesian inference. This approach uses prior knowledge combined with new data to calculate the likelihood of different outcomes. Bayesian inference became a cornerstone of modern statistics and machine learning, offering a formal framework for updating beliefs in light of new evidence.


## Summary
### **Deductive Reasoning**:

- **What it is**: Deductive reasoning involves drawing specific conclusions from general principles or premises. If the premises are true, the conclusion must also be true.
- **Example**: All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.
- **Nature**: It's a top-down approach that guarantees certainty in its conclusions, provided the premises are correct.

### **Inductive Reasoning**:

- **What it is**: Inductive reasoning involves drawing general conclusions from specific examples or observations. The conclusions are probable, but not certain.
- **Example**: Every swan I've seen is white. Therefore, all swans might be white.
- **Nature**: It's a bottom-up approach that suggests likelihood, not certainty.

### **Plausible Reasoning**:

- **What it is**: Plausible reasoning, according to Pólya, is about reasoning that seems likely to be true but is not necessarily proven. It’s the kind of reasoning we often use when we don’t have all the facts but still need to make a decision or hypothesis.
- **Example**: If you know that most people with a cough in the winter have a cold, and someone is coughing in December, you might plausibly reason that they have a cold.
- **Nature**: It's more flexible and creative, involving guesses, analogies, and conjectures. It’s reasoning based on what seems likely or reasonable, even if it isn’t logically guaranteed.

### **Comparison**:

- **Certainty**: Deductive reasoning provides certainty, inductive reasoning provides probable conclusions, and plausible reasoning is more about what seems reasonable or likely, even in the face of uncertainty.
- **Usage**: Deductive and inductive reasoning are more rigid and formal, while plausible reasoning is often used in everyday problem-solving, scientific discovery, and hypothesis formation, where not all information is available.


The difference between deductive reasoning and inductive reasoning is clear.  The difference between between inductive reasoning and plausible reasoning is not very clear.  Here is more explanation.

## Difference Between Inductive Reasoning and Plausible Reasoning

plausible reasoning often acknowledges uncertainty and works with probabilities, while inductive reasoning aims to find general rules, even if those rules are derived from limited evidence.

We could indeed think of **inductive reasoning** as a more specific type of plausible reasoning. Here's why:

- **Plausible Reasoning**: this approach deals with what seems likely or reasonable, acknowledging that the conclusions aren't 100% certain. It involves making inferences based on prior knowledge, intuition, or partial information, often without rigorous evidence.
    
- **Inductive Reasoning**: This is a more formal process where one observes specific instances and tries to generalize them into a rule or theory. Although the rule might never be 100% certain (because it’s based on observation and not pure logic), the goal is to get as close as possible to a reliable conclusion.
    

In this sense, inductive reasoning can be seen as a **subset** of plausible reasoning, where the plausibility is grounded in observed patterns and the goal is to find a generalizable rule, even if it’s not guaranteed to be true in all cases.

Plausible reasoning, on the other hand, encompasses a broader range of reasoning processes, including those where you might not have enough data to form a general rule but still need to make a decision based on what seems most likely.

### Plausible Reasoning and Its Role in Machine Learning

Plausible reasoning, particularly through the lens of Bayesian inference, bridges the gap between deductive and inductive reasoning by introducing a probabilistic approach. In the realm of machine learning, this probabilistic reasoning allows for the continuous updating and refinement of models as new data becomes available. It enhances the robustness of machine learning algorithms, enabling them to learn from data, make predictions, and adapt to new information. This approach forms the foundation of many modern machine learning techniques, which rely on probabilistic models to manage uncertainty and improve decision-making.

In conclusion, the evolution of reasoning from Aristotle's deductive logic to Bacon's inductive methodology, and finally to Bayes' plausible reasoning, illustrates a trajectory towards increasingly sophisticated methods for understanding and interpreting the world. Each step in this historical progression has contributed to the development of powerful tools that underpin modern scientific inquiry and technological advancements, particularly in the field of machine learning. By combining the certainty of deductive reasoning, the empirical richness of inductive reasoning, and the flexibility of plausible reasoning, we can navigate complex, data-rich environments with greater confidence and precision.





