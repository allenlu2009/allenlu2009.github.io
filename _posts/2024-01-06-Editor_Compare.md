---
title: 簡單文本編輯器
date: 2024-01-06 23:10:08
categories:
- Language
tags: [Linux, Editor]
typora-root-url: ../../allenlu2009.github.io


---





# 文本編輯器

這裡比較簡單的文本編輯器 (text editor), 包含 vi, emacs, and nano (or pico), and e3

這些編輯器主要用於簡單的文本編輯和程式碼編寫，在許多 Linux，Mac，甚至 Windows 系統上都能找到。

## vi

vi 是一個在 UNIX 系統上普遍使用的文字編輯器。它允許用戶在命令模式和插入模式之間切換，從而提供了強大的編輯功能。一些用戶可能會發現 vi 的使用方式有些不直觀，但一旦熟悉之後，它的效率非常高。

## emacs

emacs 是另一個在 UNIX 系統上廣泛使用的文本編輯器。它以其強大的自定義功能和內置的 Lisp 解釋器而聞名。與 vi 相比，emacs 的學習曲線可能更為陡峭，但一旦掌握，它可以非常有效地處理各種編輯任務。

## nano

Nano 是一個簡單易用的文本編輯器，特別適合於新手使用。它的界面直觀，並且在底部的菜單提供了常用命令的快捷方式。雖然 Nano 不如 vi 或 emacs 功能強大，但對於基本的文本編輯任務來說，它是一個很好的選擇。

## e3

E3 是一個相對簡單而功能強大的文本編輯器，與 vi, emacs 和 nano 一樣，提供了基本的編輯功能，同時具有直觀的使用界面。儘管它可能不像 vi 和 emacs 那樣具有豐富的特性，但 e3 的輕量級設計使其成為一個選擇良好的選項，特別是對於那些想要快速編輯文本而不需學習複雜命令的使用者。

# 快捷鍵比較

在這個部分，我們將比較 vi, emacs, nano 和 e3 這四種編輯器的快捷鍵。雖然每個編輯器都有其獨特的命令和快捷鍵，但有一般的 guideline

- Emacs 使用 Ctrl (C) 或 Alt (M) 键的组合。
- Vim 重度依賴模態編輯，有用於導航（h，j，k，l）和編輯（x，dd，yy）的模式。
- Nano 使用 Ctrl 键的组合，如 ^P，^N 进行基本导航。
- E3 可以模擬 emacs, vi, 和 nano 的 bindkeys.

## 移動游標 (Navigation)

在 vi 中，使用 'h', 'j', 'k', 'l' 來移動光標；在 emacs 中，使用 'Ctrl+p', 'Ctrl+n', 'Ctrl+f', 'Ctrl+b' 來移動光標；在 Nano 中，則可以直接使用方向鍵進行導航。

| Action            | Emacs | Vim  | Nano   |
| ----------------- | ----- | ---- | ------ |
| Move cursor left  | C-b   | h    | ←      |
| Move cursor right | C-f   | l    | →      |
| Move cursor up    | C-p   | k    | C-p    |
| Move cursor down  | C-n   | j    | C-n    |
| Previous word     | M-b   | b    | Alt+←  |
| Next word         | M-f   | w    | Alt+→  |
| Beginning of line | C-a   | 0    | C-a    |
| End of line       | C-e   | $    | C-e    |
| Top of file       | Alt+← | gg   | C-Home |
| Bottom of file    | Alt+→ | G    | C-End  |

## 編輯 (Editing)

在 vi 中，使用 'x' 刪除字符，'i' 進入插入模式，'u' 撤銷；在 emacs 中，使用 'Ctrl+d' 刪除字符，'Ctrl+k' 刪除行，'Ctrl+/' 撤銷；在 Nano 中，使用 'Del' 刪除字符，'Ctrl+k' 刪除行，'Alt+u' 撤銷。

| Action           | Emacs   | Vim  | Nano    |
| ---------------- | ------- | ---- | ------- |
| Delete character | C-d     | x    | C-k     |
| Delete line      | C-k     | dd   | kk      |
| Insert mode      |         | i    |         |
| Undo             | C-/     | u    | Alt+u   |
| Redo             | C-?     | C-r  | No redo |
| Copy line        | M-w     | yy   | Alt+6   |
| Paste            | C-y     | p    | C-u     |
| Save             | C-x C-s | :w   | C-o     |
| Quit             | C-x C-c | :q   | C-x     |

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
