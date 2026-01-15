---
title: AI for Coding - Cursor + LLM
date: 2024-08-24 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - Science
tags:
  - Claude
  - Copilot
  - GPT4
  - VScode
---


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## Introduction [[2022-09-24-AI_Copilot|2022 Copilot]] [[2024-08-24-Claude_Sonnet_Copilot|2024 Copilot]]

I used VS Code with Copilot in 2022 and changed to different plugins such as ContinueDev and ClaudeDev in 2023.    In 2024/08, Karpathy posted in X that he switched to use "Cursor", a branch from VS Code but adding the LLM capability.   It is a startup company where founders come from MIT.   Since I believe Karpathy's judgement by watching his YouTube videos, I decided to try Cursor to use Cursor.

Cursor can be seen as Method 2, i.e. non-agentic approach.      

There are two method to setup the VS Code:
1. Easy setup but expensive:  ClaudeDev
2. Slightly complicated but flexible and much cheaper:  ContinueDev + Sonnet + Qwen

A table compares method 1 and method 2:

|   VS Code    |                  Method 1                  |      Method 2       |                 Comment                  |
| :----------: | :----------------------------------------: | :-----------------: | :--------------------------------------: |
|   **LLM**    |              Claude-3.5 Sonet              |  Claude-3.5 Sonnet  | Input $3/M-tokens<br>Output $15/M-tokens |
| **Plug-in**  |                 ClaudeDev                  | ContinueDev, Cursor |                                          |
| **Function** |               Agent for task               | Code and Auto-fill  |                                          |
|  **Agent**   |                   Claude                   |         --          |          No agent for Method 2           |
|   **Code**   |                   Claude                   |       Claude        |                                          |
| **Autofill** |                     --                     |        Owen2        |          Owen2 is local, free!           |
| **Comment**  | **Agent is inmature<br>Cost is expensive** |                     |                                          |

## Example

I used transformer model as an example.

Prompt






兩年內 LLM (大型語言模型) 帶來的變化令人驚訝。我在2022年寫了一篇關於使用 VS Code + Copilot blog。GitHub Copilot是由OpenAI的Codex模型驅動，而Codex模型是GPT-3模型的後裔。每月支付10美元使用這麼舊的產品實在有點離譜。新的小型語言模型（例如，Microsoft Phi3）可能表現得比Codex更好。

## Reference

Sonnet-3.5 ClaudeDev Extension: https://www.youtube.com/watch?v=5FbZ8ALfSTs&t=678s
Sonnet-3.5  + Qwen2 + ContinueDev:  https://www.youtube.com/watch?v=Vxsx7Il-KMA