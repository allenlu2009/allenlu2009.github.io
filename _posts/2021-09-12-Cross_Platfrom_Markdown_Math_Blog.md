---
title: 跨平臺 Markdown Plus MathJax Blog Editing 經驗分享
categories:
- AI
tags: [ML, EM, Bayesian, MAP]
typora-root-url: ../../allenlu2009.github.io
---


## 跨平臺 Markdown+MathJax Blog Editing 分享

我常用的計算平臺包含：MacBook air/pro (Mac OS),  PC (Windows 10), and iPad air (iPad OS).

常常使用的 computing platform and offline blog editors 如下：

## Computing Platform and Editor

MacBook air/pro (mobile at work)  - **Marsedit (paid), mweb (paid, Markdown)**

Windows (fixed at home) - WLW (Window Live Writer, free), **VScode (free, Markdown), Typora (free, Markdown)**

iPad air (for portability and photo) - **mweb (free for editing, not for creating, publishing, Markdown)**

## 推薦的 Markdown+MathJax Editor

### 爲什麽使用 Markdown editor for Blog

* Math equation input: Markdown editor 大多可以插入 latex-based 數學公式 (e.g. MathJax or KaTex).  搭配 [Mathpix Snip](https://mathpix.com/) (大推！) 是我用過最好的 math equation generator!  Example: $i \hbar \frac{d}{d t}\mid \Psi(t)\rangle=\hat{H}\mid \Psi(t)\rangle$
* 主要的 blog platforms (e.g. Wordpress, 特別是 Github) 都支持 Markdown.  
* 可携性大爲提高。

因此我從 WYSIWYG editor 像是 Marsedit (Mac), WLW (PC) 改用 Markdown editors.

### 數學公式 rendering for Blog

最早的數學公式 rendering 來自 Latex，主打 professional and static publising 例如 thesis, paper, etc. pdf files.

Latex 的龐大和 static 特性並不適合用於 dynamic rendering 的 blog.  因此有兩種變形：MathJax and KaTex.   MathJax 比較接近 Latex.  KaTex 主打快速 dynamic rendering, 有一些 equation numbering and reference 並不支持，**因此我主要使用 MathJax.**

<img src="/media/image-20210911234257339.png" alt="image-20210911234257339" style="zoom:80%;" />

Latex/MathJax 的輸入可以使用 Mathpix Snip capture, 非常有用！

### 分享我用過的 Markdown+Math editors

#### mweb (built-in MathJax support)

* **優點:** (1) 支持 markdown editing, content management, figure management, and blog publishing; (2) 跨 Mac OS and iPad OS or iOS;  (3) 支持 iCloud sync between MacBook pro and iPad air.

* **缺點:** **(1) 沒有 Windows version.**  (2) mweb3.0 對於 math equation 支持比較差，有一些常用的 math symbol (e.g. Lagrangian) 不支持。不過 mweb4.0 似乎有改善。**(3) 另外對於 Apple M1 晶片的穩定性很差！**

#### Typora (built-in MathJax support)

* **優點:**
  * 設定簡單，基本打開  (Jekyll _post) directory 就可以編輯。下圖左是 directory file list.  下圖右是 editing&preview together window.
  * **UI 非常簡潔，不支持 side-by-side markdown and rendering display. 但是使用一陣子發覺 Typora 直接 rendering output to display 很棒。**不會像 mweb editing 時，另一個 display 動來動去。特別在小銀幕非常適合，mweb 的 iOS 版採用一樣的做法。可惜沒有 iOS/iPad OS version
  * 跨 Windows/Mac OS/Linux, 但沒有 iOS version.
* **缺點:**  
  * 只有 Markdown editing, preview, file list, 但是沒有 blog publishing.
  * Another big disadvantage: math equation number 支持不好 \label{} 常常有問題。

<img src="/media/image-20210912100220717.png" alt="image-20210912100220717" style="zoom:80%;" />

* 設定 root-path for image directory (下面兩步都要做！)
  * 在本文加上： typora-root-url: ../../allenlu2009.github.io

  ![image-20210814233107185](/media/image-20210814233107185.png)

  * Typora: Format: Image: Use Image Root Path: set to the above directory

  * 設定 copy and paste image path:  Typora: File : Preference: Image :  設定 copy and paste image directory.

<img src="/media/image-20210912100913945.png" alt="image-20210912100913945" style="zoom:80%;" />

#### VSCode + Math Preview Extension

* **優點:**  

  * VSCode 整合 (via extension) git version control and Github pull/push, 可能對 Github posting 比較容易
  * VSCode 有很多的 extensions, 例如 jekyll 可以直接 preview post to github blog.  或是 mathjax 以及其他 preview 的功能。

* **缺點:**  

  * **VSCode 有 built-in markdown preview!  但完全不支持 Math rendering!!** 需要 install plug-in.  另外 preview image 非常多坑！
  * 除非是 coding 達人，不然不推用 VSCode 做 math blog!

* VSCode extension 有兩個 extensions  **(1) Markdown+Math:  only support KaTex;  (2) Markdown Preview Enhaced: Default KaTex, Optional MathJax**, changed in setting.
  <img src="/media/image-20210912001358757.png" alt="image-20210912001358757" style="zoom:40%;" /> and <img src="/media/image-20210912001314540.png" alt="image-20210912001314540" style="zoom:40%;" />

* 兩者各有缺點，所以兩個都用。

  * Markdown+Math (KaTex only): 下圖左是 Markdown, 下圖右是 Markdown+Math preview.  首先這個 markdown 支持 dark mode; 再來 math 只支持 KaTex, 所以 \label{} 以及 cross reference equation 不支持。

<img src="/media/image-20210912085129732.png" alt="image-20210912085129732" style="zoom:90%;" />

* Markdown Preview Enhanced (Default KaTex, change to MathJax in setting): 下圖左是 Markdown, 下圖右是 Markdown Preview Enhanced (Setting: KaTex -> MathJax).   看起來還不錯。但只要左邊編輯 markdown, 右邊數學公式就出現亂碼！所以只能用於最後確認效果。

<img src="/media/image-20210912081052675.png" alt="image-20210912081052675" style="zoom:100%;" />

* Image preview 一堆問題！同樣分成 built-in preview 和 Markdown Preview Enhaced.  總結來説都很爛，但是 Markdown Preview Enhanced (差) 比 built-in preview (爛) 好。
  * image reference 格式:  built-in preview 只接受標準 markdown 格式： ![text](...jpg) ；preview enhanced 支持標準格式以及 html 格式 : <img src =“...">"
  * remote image:  由於 secuity concern, built-in preview 只接受 https;  preview enhanced 支持 http or https.  e.g.
  ![https](https://ww1.sinaimg.cn/mw690/81b78497jw1emfgwkasznj21hc0u0qb7.jpg)
  ![http](http://ww1.sinaimg.cn/mw690/81b78497jw1emfgwkasznj21hc0u0qb7.jpg)
  * local file system image:  這是最糟糕的部分！只支持 local directory 為 root 的絕對 path!!!  也就是説，無法另外設定 root path, 也不能用 ../media/ 往上 path (因爲 root 沒有更上面的 directory).  我最後只能在 _post directory 之下做一個 symbolic link : PS>  new-item -itemtype symboliclink -path ./  -name media -value ../media
  * 完全不建議一般人用 VSCode 作 markdown blog!

## Cloud Platform

跨平臺不只是 computing platforms, Windows/Mac OS/iOS, 更重要是要有 cloud platform 同步到一個 database!  從不同的 computing platforms 要很容易讀寫這個 cloud platform.  基本上只有幾個常見 cloud platforms 能達到這個要求：

* Google Drive
* Microsoft Onedrive
* Apple iCloud
* Dropbox

我選擇使用 iCloud, 原因很簡單。因爲 mweb 只支持 iCloud and Dropbox.   Dropbox 的 free quota 只有 2GB.  iCloud 的 free quota 5GB.   本文是先用 MacBook Pro 的 mweb create and start the article.  再使用 PC Typora 纂寫大部分内容。

最後再切回 iPad Air 使用 MWeb 繼續，並且拍一張照片結束。這是目前我比較滿意的寫作方式。
最後再再切回 MacBook Pro to publish, have fun!

![text](/media/16101804667280/16102111356622.jpg)

![test](/media/image-20210911234257339.png)

<img src="/media/image-20210911234257339.png" alt="image-20210911234257339" style="zoom:80%;" />
