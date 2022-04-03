---
title: Improve Engineer Efficiency - Python Image Library
date: 2021-12-11 18:10:08
categories: 
- Language
tags: [Python, Image, PIL, CV]
typora-root-url: ../../allenlu2009.github.io
---



## View Image Tools

Python 有幾個常用的 image library, 使用例如 image read: imread, etc.



#### CV2

import cv2

最常用的是  

raw_input_image = cv2.imread(input_path)



Install:  

conda install -c conda-forge opencv







#### PIL (pillow)

sci.... image

這個部分主要是看 input data sample or output data sample.





## 可視化 Tools

可視化 tool 主要是用圖形化界面看 training or inference 的結果。



Tensorboard (Google)



Tensorboardx (Facebook)



Visdom (Facebook)

https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter04_advanced/4_2_1_visdom/



Visdom是Facebook在2017年发布的一款针对PyTorch的可视化工具。[官网](https://github.com/facebookresearch/visdom),visdom由于其功能简单，一般会被定义为服务器端的matplot，也就是说我们可以直接使用python的控制台模式进行开发并在服务器上执行，将一些可视化的数据传送到Visdom服务上，通过Visdom服务进行可视化

Install:  conda install -c conda-forge visdom



