---
date: 2024-08-08 23:10:08
title: Test Obsidian Dataview Plugin
tags:
  - Obsidian
category: Tools
---

## Introduction

Obsidian 提供方便的筆記功能。非常有用的功能是插件。插件可以擴展 Obsidian 的基本功能，使使用者能夠根據自己的需求和工作流程進行個性化設置。例如，使用 Markdown 語法撰寫筆記的使用者可以通過安裝特定插件來增強文本編輯能力，包括語法高亮、數學公式支持、圖表繪製等。

此外，還有一些插件可以幫助使用者進行知識管理。例如，"Daily Notes" 插件可以自動生成每日筆記模板，方便使用者記錄日常事務和靈感；"Backlinks" 插件則能展示筆記之間的引用關係，有助於構建知識網絡。

Obsidian 的社群還提供了大量第三方插件，由開發者們持續更新和維護。這些插件涵蓋了從數據可視化到時間管理等各個方面，幾乎可以滿足不同領域使用者的需求。如果你有程式能力，還可以自己開發專屬插件，以進一步提升工作效率。

對於需要進行任務管理的使用者，可以考慮安裝 "Kanban" 插件，這個插件允許你在 Obsidian 中創建看板視圖，更直觀地管理任務和專案進度。

Obsidian 的插件系統使其成為一個強大且靈活的工具。通過不斷探索和嘗試不同的插件，你會發現 Obsidian 能夠極大地提升你的生產力和創造力。

幾個常用的插件
- Copilot
- Text Generator
- Dataview
- Awesome Image
- Image with Relative Path (manually install)



## Built-in : Mermaid
```mermaid
flowchart TD

    A[Christmas] -->|Get money| B(Go shopping)

    B --> C{Let me think}

    C -->|One| D[Laptop]

    C -->|Two| E[iPhone]

    C -->|Three| F[fa:fa-car Car]
```



## Plugin: Copilot


## Plugin: Text Generator

可以 download 很多 model template!
Ctrl-P:  Text Generator: show model from template
Change the model to  "Change the model to better align with the evolving needs of our target audience.



## Plugin: Image with Relative Path

https://github.com/csdjk/lcl-obsidian-html-local-img-plugin

Git clone code 到 ./obsidian/plugins

接著 manually install。主要把 main.js (javascript) compile to main.ts (可以執行)
[Build a plugin - Developer Documentation (obsidian.md)](https://docs.obsidian.md/Plugins/Getting+started/Build+a+plugin#:~:text=In%20Obsidian%2C%20open%C2%A0%2A%2ASettings%2A%2A.%202.%20In%20the%20side%20menu%2C,now%20ready%20to%20use%20the%20plugin%20in%20Obsidian.)

* npm i  (or npm install)
* npm run dev

#### MacOS and Linux 設定 softlink

> $ link -s ../media media

#### Windows 設定 softlink
首先要用 PowerShell 但是用**系統管理員**執行：
<img src="/media/image-20240809223751.png" alt="20240809223751" style="zoom:60%;" />

接著到 \_post folder 下執行
> PS New-Item -Path media -ItemType SymbolicLink -Value ../media

