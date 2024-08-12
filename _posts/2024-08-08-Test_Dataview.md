---
date: 2024-08-08 23:10:08
title: Test Obsidian Dataview Plugin
tags:
  - Obsidian
category: Tools
---

## Introduction

Obsidian 提供方便的筆記功能。

## How to use data view for Obsidian data view plugin

Dataview，一款類似 SQL 語法 (資料庫的結構化查詢語言, Structured Query Language) 的查詢語言。 Dataview 可以快速查詢特定條件的筆記，並且用 table (表格), list (清單), task (待辦事項) 呈現這些筆記條目。

有兩種方法建立欄位
### 在文章或是筆記加上 Metadata
1. 利用 YAML (Yet Another MarkUP Language) 在上下 ```---``` 包住一段格式化的文字。
```yaml
---
data : 2023-08-08
aliases: [test]
categories: 
- AI
tags: [item1, item2, item3]
---
```

在 YAML 區域定義的欄位都可以在後續被 Dataview 當作欄位搜尋到。想要了解更 YAML 格式如何寫，可以參考 [Obsidian 的官方文件](https://help.obsidian.md/Advanced+topics/YAML+front+matter)。

### 使用 inline field 加入欄位





```dataview
table 
from ""
where (date(today) - file.cday <= dur(20 days)) and contains(file.name, "2024")
sort file.ctime DESC
```


Files contains LLM tag

```dataview
table 
from ""
where contains(tags, "LLM")
sort file.ctime DESC
```


```dataview
table 
from ""
where contains(categories, "AI")
sort file.ctime DESC
```

