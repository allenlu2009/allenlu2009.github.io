---
title: Jekyll Memo for Github Blog
date: 2021-06-30 08:29:08
categories:
- Language
tags: [Jekyll, Github]
typora-root-url: ../../allenlu2009.github.io
---

幾個重點

## Header
* title line:  no other :,  wrong example:  title: Math AI : xxx => the second : to be removed!

* tags: [xxx, xxx, xxx]

## Table
* 目前 Jekyll + Next theme 造成 table column width 非常寬。 I don't know the exact reason.  I changed the xxx/xxx.github.io/_sass/_common/scaffolding/tables.scss
    * width: 300px;
    * table-layout: auto; 

## Equation

* \\$\\$ math equation \\$\\$ => leave empty lines "before" and "after" \\$\\$ \\$\\$! 也就是上下各要空一行！

* \$\{\{ \}\}\$  => \$\{ \\{ \\}\}\$.  如果要打 {, 一定要加 \\{. 

* Equation number:  必須先加上 header 如下。Reference: https://jdhao.github.io/2018/01/25/hexo-mathjax-equation-number/

* The commands \tilde, \dot, \ddot, \hat, and \bar mess up with subscripts;  https://github.com/mathjax/MathJax/issues/2474
   solution for me:  add a \_ instead of _!!!
```
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
```

Equation 本體
```
$$\begin{equation}
E=mc^2
\end{equation}\label{eq1}$$
```
或是
```
$$\begin{align}
E=mc^2  \\         => auto number 
p = mv \nonumber \\  => without equation number
F = ma  \label{eqF} \\.  
\end{align}$$
```

Equation citation use `$\eqref{eq1}$`

* Mweb 可以直接產生 equation number!
* Typora 需要 enable :preference :Markdown :Auto Numbering Math Equations".  不過結果很奇怪。所有的 equation 都有 number in Typora!  但 Jekyll 正常。 


## Image
Markdown resize image 似乎有問題，需要另外的 plug-in => No!

我找到一個 work around in Mweb!  使用 <img src ...., width=""> 取代 Mweb copy and paste image.

不過後來我發現 typora 可以直接做，所有 method 2 is using Typora

### Method 1: Mweb

* Mweb: copy and paste image 自動產生。如下圖中的
```
![-w414](/media/16286850167880.jpg )
```
此時同時在 editing window and preview window 都有圖。如下圖左和右上。
* 我找到的 work around 加在後一行。只會在 preview window 有圖。如下圖右下。
```
<img src="/media/16286850167880.jpg" width="414">
```

![-w993](/media/16289455797795.jpg)


* 問題是這種 image resize 只對 mweb 有效。在 Jekyll 之後的截圖如下 (127.0.0.1:4000)，就不對。
*  Jekyll 直接忽略 [-w414] in the first image! 因此第一行產生原圖。但接受第二行的 image size
![](/media/16289465237748.jpg)


* I check the html source code

    ![](/media/16289463211491.jpg)

    * 第一行轉譯的結果：alt="-w245" 顯然被忽略。
    * 第二行轉譯的結果：width="245" 是正確結果。 
* 我找到的方法是把第一行改成第二行。

### Method 2: Typora, How? 

* 首先要解決的是 root path 的問題！ Jekyll (and therefore github) 有 root path 的觀念。For my local root directory:  /Users/allenlu/Onedrive/allenlu2009.github.io.   文章是在 root: /_posts/xxx;  image 是在 /media/xxx

* Mweb 似乎自動解決這個問題。 Image 直接 refer to:  /media/xxx.

* Typora 如何設定？ 有兩個方法

  * 直接在本文加上： typora-root-url: ../../allenlu2009.github.io

    ![image-20210814233107185](/media/image-20210814233107185.png)

  * Typora: Format: Image: Use Image Root Path: set to the above directory 

* Typora insert image 必須先設定 image save path

  * Preference: Image: Copy image to custom folder: 
    * /Users/allenlu/OneDrive/allenlu2009.github.io/media

* Typora 在設定 display path 之後和 Jekyll 一樣，可以 display image, 但是不會 scale image size!!

* 如果 mweb 改成 “\<src img xxx\>”  之後 OK.

* 不過我發現有更好的方法，就是直接用 typora 的 image zoom 設定。自動就會轉成 <src img,  , zoom xxx> 可以 image resize!!



### Editor

Mweb

Typora

VS Code

* VS code default math rendering tool is KaTex, which is different from MathJax.  KaTex does not support /label and cross reference.   So I install "Mardown Preview Enhance" and switch the default math engine from KaTex to MathJax.

<img src="/media/image-20210911004231947.png" alt="image-20210911004231947" style="zoom:50%;" />

* 問題是 MathJax mode is buggy!!  Not support \boldsymbol!!

## 結論

1. 使用 Typora for image resize (zoom), 當然要設好 typora-root-url (for display), and image save path.  結果是 typora, Mweb, Jekyll/github OK.
2. 使用 Mweb,  需要手動改變 image to <src img ...., width="xxx">

推薦使用 1!!!