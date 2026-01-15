---
title: AI for Coding - VS Code + Claude-3.5 Sonnet
date: 2024-08-24 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - Tools
tags:
  - Claude
  - Copilot
  - VScode
---


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## Introduction [[2022-09-24-AI_Copilot|2022 Copilot]]

It's pretty amazing how things changes by LLM in two years.   I wrote an blog about using VS Code + Copilot in 2022.   GitHub Copilot is powered by OpenAI's Codex model, which is a descendant of the GPT-3 model.   It's pretty lame to pay $10/month for such an old antique.  The newly small language model (e.g. Microsoft Phi3) probably perform better than Codex.

Some YouTube videos show how to replace the Copilot LLM using Claude-3.5 Sonnet.  It is supposed to have a lot better performance than Codex. 

There are two method to setup the VS Code:
1. Easy setup but expensive:  ClaudeDev
2. Slightly complicated but flexible and much cheaper:  ContinueDev + Sonnet + Qwen

A table compares method 1 and method 2:

|   VS Code    |                  Method 1                  |      Method 2      |                 Comment                  |
| :----------: | :----------------------------------------: | :----------------: | :--------------------------------------: |
|   **LLM**    |              Claude-3.5 Sonet              | Claude-3.5 Sonnet  | Input $3/M-tokens<br>Output $15/M-tokens |
| **Plug-in**  |                 ClaudeDev                  |    ContinueDev     |                                          |
| **Function** |               Agent for task               | Code and Auto-fill |                                          |
|  **Agent**   |                   Claude                   |         --         |          No agent for Method 2           |
|   **Code**   |                   Claude                   |       Claude       |                                          |
| **Autofill** |                     --                     |       Owen2        |          Owen2 is local, free!           |
| **Comment**  | **Agent is inmature<br>Cost is expensive** |                    |                                          |

## Claude-3.5 Sonnet (>100B) + ClaudeDev

Setup is easy.   Just follow the YouTube and claim the $5 free coupon.

I tried a couple of tasks.   **None of them works!**   There is environment issue that this agent **CANNOT solve**!!!  Waste of time and money.   Thank God Claude offers $5 free quota.

## Claude-3.5 Sonnet (>100B) + Qwen2 (2B?) + ContinueDev

Step 1:  Add Continue Plug-in:  this is pretty straightforward.
Step 2:  Setup the Claude-3.5 Sonnet as the LLM in the Continue




兩年內 LLM (大型語言模型) 帶來的變化令人驚訝。我在2022年寫了一篇關於使用 VS Code + Copilot blog。GitHub Copilot是由OpenAI的Codex模型驅動，而Codex模型是GPT-3模型的後裔。每月支付10美元使用這麼舊的產品實在有點離譜。新的小型語言模型（例如，Microsoft Phi3）可能表現得比Codex更好。

## Reference

Sonnet-3.5 ClaudeDev Extension: https://www.youtube.com/watch?v=5FbZ8ALfSTs&t=678s
Sonnet-3.5  + Qwen2 + ContinueDev:  https://www.youtube.com/watch?v=Vxsx7Il-KMA