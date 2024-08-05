# How to test

$$ a^2 + b^2 = c^2 $$

this is a Pythagoras  equation, $\alpha$ is a test. $\alpha$ is the correct symbol.  $\beta$ and $\alpha \text{ and } \gamma$

/gpt where is the capital of Japan



# AI Markdown 編輯器

這裡比較 AI 文本編輯器，包含 Typora (目前沒有 AI), Notion, and iA Writer.  重點在於對與 markdown 語法和 AI 的支持。

我們也將討論如 Typora, Notion, 和 iA Writer 這些 AI 文本編輯器的功能和優點，並將它們與傳統的文本編輯器進行比較。

這些編輯器在許多 Linux，Mac，甚至 Windows 系統上都能找到。

## Typora + ChatGPT

Typora 是一款 Markdown 文本編輯器，簡潔的界面和即時預覽功能使其成為寫作和筆記的理想工具。它支持一系列 Markdown 擴展語法，包括表格、數學公式和代碼塊等。Typora 的自動保存和版本控制功能也使得文件管理變得更加輕鬆。然而，與 vi 和 emacs 相比，Typora 缺少了命令行模式和強大的自定義功能。

## Notion

Notion is an emerging note-taking and project management tool with rich features, including embedded databases, embedded pages, templates, Markdown support, etc. These features make Notion not just a text editor, but a powerful organization tool. Notion has very good cross-platform support and can integrate with various third-party applications, such as Slack, Google Calendar, etc.

## iA Writer

iA Writer 是一款集中於無干擾寫作的文本編輯器。它的界面極簡，並提供了一種名為「焦點模式」的功能，這可以幫助寫作者專注於當前的句子或段落。iA Writer 支持 Markdown 語法，並且具有強大的導出功能，可以將文件導出為 PDF、Word 或 HTML 格式。但是，iA Writer 的定價較高，並且與 vi 和 emacs 相比，其自定義功能有限。

https://sourceforge.net/software/compare/Notion-AI-vs-Typora-vs-iA-Writer/

# Notion AI / Typora / iA Writer 比較

在這個部分，我們將比較 Notion AI、Typora、iA Writer 這三種編輯器對 markdown 和 AI 的支持。

- 

## 支持平台

Notion 有最好的支持度。對於 Mac 和 Windows 也可以使用網頁版本，非常方便。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/79b49a29-3da1-4933-ae65-860896f4a291/a3228ad7-a837-4476-a5f8-5ef86a65581d/Untitled.png)

## 費用

Notion 本身提供免費版本。但是 Notion AI 使用後需要升級 Notion 為付費版本 ($10/month)，再加上 AI 部分 ($8/month)。其實並不便宜。

Typora 之前免費，目前需要付費 ($10) 購買，不過是一次性。但是 Typora 沒有 AI 功能，需要使用 ChatGPT 外掛。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/79b49a29-3da1-4933-ae65-860896f4a291/55c2238a-e8c3-4dfb-8092-521fd99a6811/Untitled.png)

## Markdown

在 vi 中，使用 'x' 刪除字符，'i' 進入插入模式，'u' 撤銷；在 emacs 中，使用 'Ctrl+d' 刪除字符，'Ctrl+k' 刪除行，'Ctrl+/' 撤銷；在 Nano 中，使用 'Del' 刪除字符，'Ctrl+k' 刪除行，'Alt+u' 撤銷。

| Action | Notion        | Typora         | iA Writer |
| ------ | ------------- | -------------- | --------- |
| Text   | Good          | Good           |           |
| Image  |               | Local/External |           |
| Math   | Katex (Latex) | Mathjax        |           |
|        |               |                |           |

## 搜尋/取代 (Search/Replace)

在 vi 中，使用 '/' 進行搜尋，':%s/old/new/g' 進行取代；在 emacs 中，使用 'Ctrl+s' 進行搜尋，'M-%' 進行取代；在 Nano 中，使用 'Ctrl+w' 進行搜尋，'Alt+r' 進行取代。

| Action        | Emacs | Vim               | Nano |
| ------------- | ----- | ----------------- | ---- |
| Find          | C-s   | /                 | C-w  |
| Find next     | C-s   | n                 | C-w  |
| Find previous | C-r   | N                 | wc   |
| Replace       | M-%   | :%s/find/replace/ | C-\  |

# 結論

在選擇適合的文本編輯器時，最重要的是考慮你的需求和舒適度。如果你需要強大的自訂功能，那麼 vi 或 emacs 可能是好選擇。如果你是初學者或只需要進行基本的文本編輯，那麼 Nano 可能是最佳選擇。