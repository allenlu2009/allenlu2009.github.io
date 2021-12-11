---
title: Machine Learning Database
categories:
- AI
tags: [database, machine learning, ML]
typora-root-url: ../../allenlu2009.github.io
---

AI, 特別是深度學習的 ABC 是 Algorithm, Big data, and Computation.  

1. Algorithm: CNN (image, vision), LSTM (voice), Transformer (NLP), etc.
2. Big data: ImageNet, and other big datasets.
3. Computation: GPU, TPU/NPU/APU, etc.

其中 (2) 顯然和 database 高度相關。不過早期的 big data 都是非常結構化的資料，簡單的 table or dataframe 類似 csv 就可以用來 training. 似乎沒有強烈的 database 需求。

## AI 幾個趨勢

1. 從雲 (cloud) 擴散到端/終端裝置 (edge/device)。對於 cloud 而言，有無限的 computation and storage resources, 1-3 都不是問題。目前端主要是 inference 為主，2 並不重要。不過 on-device learning 越來越重要，edge AI 需要 on-device database 處理 real-time update data.

2. 結構化資料變成非結構化資料。甚至 fused structured/unstructured data, e.g. image+voice, video+radar, etc.

3. Centralized learning (and database) 變成 distributed learning (and database).  由於隱私考量，還有 federated learning, etc.

## Our Goal

Edge or on-device learning.  可以 real-time training (~1sec)! base on dynamically updated data.  不是等到 idle time or sleep time 才 training.  此時要同時解決 database + learning 的問題。另外也需要支持 distributed learning or federated learning 和 cloud or 其他的 edge device exchange information.

## 方法

Method 1: Database is database;  Learning is learning.

Method 2: Keep your friends close, and your enemies closer -> Keep your ML close, and your data closer. 直接在 database 支持 ML!
WHY?
    *Real-time, low latency
    * AutoML for database!  Extract feature automatically

## Database candidate [@choudhuryTopDatabases2020]

Method 1:

MySQL
    Most popular open-source relational database manageemnet systems (RDBMS).
    Acquired by Oracle, paid for commerical application.
MariaDB
    Open source, similar to MySQL (by MySQL inventor)
MongoDB
    Document database. Store data in JSON-like documents.  It seems useful for non-stuctured data.

 PostgreSQL
    Extensibility.  Tensorflow support PostgreSQL.

Method 2:
MLDB
    Open source real time prediction endpoints.  Integrate ML functions.

Redis
    built-in Lua scripting, Redis-ML.

MindsDB

隨著網際網路的發展，我們把一台一台伺服器變成多台伺服器。當開始建立資料備份時，需要加一個緩衝層來調整所有的查詢，投入更多硬體。最後，需要將資料切分多個集群上，並重構大量的應用邏輯以適應這種切分。不久之後，你就會發現被自己數月前的設計資料結構限制住了。

隨著web2.0的興起，關聯式資料庫本身無法克服的缺陷越來越明顯，主要表現為如下幾點：
1.對資料高併發讀寫的需求
2.對海量資料的高效率存儲和訪問的需求。
3.對資料庫的高可擴展性和高可用性的需求。
4.資料庫事務一致性需求。
5.資料庫寫實性和讀寫時性需求。
6.對複雜SQL的查詢，特別是對關聯查詢的需求。

NoSQL是Notonly SQL的縮寫，NoSQL不使用SQL作為查詢語言。其資料存儲可以不需要固定的表格模式，也經常避免使用SQL的join操作，一般有水準可擴展性的特徵。

NoSQL又分成四大類：
1.Key-Value，如Redis。
2.Document-Oriented，如MongoDB。
3.Wide Column Store，如Cassandra。
4.Graph-Oriented，如Neo4J。
而本篇要介紹的主角則是Key-Value的Redis。

### MySQL and MariaDB

MySQL 和 MariaDB 看來非常相似。但有以下差異

* MySQL is owned by Oracle;  MariaDB is open source.
* MariaDB support JSON-like data (MySQL does), and support python connector (MySQL does)?

Suggest to use MariaDB. (give up at the last minute!)

passwd: aluxxx4 or axxxxxxz (mac)
![](/media/../media/img-2021-10-03-23-19-10.png)
![](/media/../media/img-2021-10-03-23-23-37.png)

**Instead, install MySQL** because it is supported by LibreOffice BASE

![](/media/../media/img-2021-10-03-23-41-26.png)
![](/media/../media/img-2021-10-03-23-48-47.png)
use lagacy authentication!
![](/media/../media/img-2021-10-03-23-50-13.png)
![](/media/../media/img-2021-10-03-23-51-05.png)
![](/media/../media/img-2021-10-03-23-51-59.png)
![](/media/../media/img-2021-10-03-23-52-27.png)
![](/media/../media/img-2021-10-03-23-53-28.png)

### Plan

* Front-end: LibreOffice BASE (or MS Access? Yes)
* Backend-end:  MySQL

#### How to Input .sql file into the database?

1. Use MySQL workbench to import an existing .sql database
2. 如果沒有 existing database, 手動 keyin all record, only for very small test database
3. Use LibreOffice BASE or MS ACCESS to create table

#### How to access the database?

Set account
Create a script to generate the table

一個資料庫管理系統（DataBase Manage-ment System，DBMS）中可以有很多個資料庫，也可以有很多組使用者帳戶，當使用者想要存取資料庫內容時，必須先通過帳戶驗證，下圖說明它們的關係。



# <img src="/media/image-20211005222449975.png" alt="image-20211005222449975" style="zoom:80%;" />

## MySQL 簡介

MySQL 是一套快速、功能強大的資料庫管理系統。所謂資料庫管理系統（Database Management System, 簡稱為 DBMS），它是透過一組程式模組來組織、管理、儲存和讀取資料庫的資料，任何使用者在操作資料庫時，都需要透過資料庫管理系統來處理。

目前 MySQL 已經成為最流行的開源資料庫，被廣泛地應用在網路上的中小型網站中，也逐漸用於更多大規模網站和應用。

------

## MySQL Workbench 簡介

MySQL Workbench 是一款專為資料庫架構師、開發人員和 DBA 打造的一個統一的視覺化工具。MySQL Workbench 提供了資料建模工具、SQL 開發工具和全面的管理工具 (包括伺服器配置、使用者管理、備份等)，可在 Windows、Linux 和 Mac OS 上使用。

新版的 MySQL Workbench 6 介面如上圖所示。最大的改進在於圖形化使用者介面 (GUI) 和工作流，同時為開發人員和 DBA 提供更加現代和精簡的設計、開發和管理資料庫的工具。
