---
title: Web Crawler or Scraper
date: 2022-10-16 23:10:08
categories:
- AI
tags: [Python, Crawler, Scraper]
typora-root-url: ../../allenlu2009.github.io


---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


### Citation



[@seleniumWriteYour2022] : official website example

[@allenSelenium4New2021] : compare selenium 3  and selenium 4

[@tailemiWebScraper2021] : web scraper 教學



### Abstract

website -> (crawl/scrape -> unstructured data -> parse ->  (database) -> presentation)

是否有機會讓 database left shift?  (1) 分析 unstructured data (with date and meta-data of course); (2) 甚至可以 crawl/scrape data automatically.

剛好找到一個例子：[ImportFromWeb | Web scraping in Google Sheets - Google Workspace Marketplace](https://workspace.google.com/marketplace/app/importfromweb_web_scraping_in_google_she/278587576794)

使用 excel 作爲 front-end.  利用 data crawler scrapes web site 可以自動 update 資料。

Why left shift?  (1) keep raw data for future analysis/verification; (2) 可以 present date or time sequence evolution; (3) for missing data, 可以主動出擊 (active search). 



## Introduction

AI 世界, data is the king.  Data 從何而來？(1) 有人整理好的 public dataset 或是花錢買或收集的 private dataset;     (2) 從 Internet 爬 (crawl) 或抓 (scrape) 出來再整理。

爬或抓是第一步；整理是第二步。本文聚焦在第一步。

分析:  

selenium (Python) 3.x or 4.x

scraper (GUI)



整理:  BeautifulSoup











## Reference
