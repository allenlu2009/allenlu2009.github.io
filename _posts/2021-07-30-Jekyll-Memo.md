---
title: Jekyll Memo for Github Blog
date: 2021-06-30 08:29:08
categories:
- Language
tags: [Jekyll, Github]
---

幾個重點

Header
* title line:  no other :,  wrong example:  title: Math AI : xxx => the second : to be removed!

* tags: [xxx, xxx, xxx]

Table
* 目前 Jekyll + Next theme 造成 table column width 非常寬。 I don't know the exact reason.  I changed the xxx/xxx.github.io/_sass/_common/scaffolding/tables.scss
    * width: 300px;
    * table-layout: auto; 

Equation

* \$\$ math equation \$\$ => leave empty lines "before" and "after" \$\$ \$\$! 也就是上下各要空一行！

* \$\{\{ \}\}\$  => \$\{ \\{ \\}\}\$.  如果要打 {, 一定要加 \\{. 


Image
* resize image 似乎有問題，需要另外的 plug-in