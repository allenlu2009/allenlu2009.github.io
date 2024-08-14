---
title: Julia Code Snip 
date: 2022-09-20 09:10:08
categories:
- AI
tags: [Score, ML, Fisher Information]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


## Why Julia?

我非常希望能有一個像 Matlab 的 programming tools: 

* Cross platform (Windows, Mac, Linux)
* Free and very easy to install and launch (Matlab 顯然既非 free 也不容易 install 和 launch)
* 非常直覺的數學格式 (scaler and vector) 支持 (Python is out!)
* 支持 math symbol or unicode symbol, 最好是 code 而不是 comment (e.g. Jupiter)  
* 完整的 library, e.g. linear algebra, DSP, distribution, etc.
* 非常重要是 graphics 的功能!  e.g.  Matlab, Matlibplot, 



Julia 是我很看好的 candidate.  不過 Julia 有幾個問題：

* Syntax 變化快：Julia 0.7 之前 support "linspace()", 後來完全不 support!!   這讓很多舊 code 無法直接使用。同時影響 Copilot 之類 AI low code, no code 在 Julia 的有用性。
* 同一個功能有不同 packages.  例如 plot 有四五種不同 packages:  PyPlot, Plots, Gadfly(?)



### Latex Math Symbols

In the Julia REPL and several other Julia editing environments, you can type many Unicode math symbols by typing the backslashed LaTeX symbol name followed by tab.  (e.g. \pi [tab])




### Plots

第一個問題是要用 PyPlot or Plots package!  建議使用 PyPlot 因爲和 Matlab, Python 的使用經驗一致！



**PyPlot: Plot sin with axis/range**

```python
using PyPlot
x = range(0, 2pi; length=1000)   # or step=0.01
y = sin.(x)

figure()   # if using VS code
plot(x, y, color="red", linewidth=2.0, linestyle="-")
xlabel("x")
ylabel("sin(x)")
title("sin(x)")
grid("on")
display(gcf()) # if using VS code


```



**Plots: Plot sin with axis/range**

```julia
using Plots
plot(sin, 0, 2π, xlims=(-10, 10), ylims=(-2,2))  # No axis(), use xlims/ylims instead
```

<img src="/media/image-20220920205256123.png" alt="image-20220920205256123" style="zoom:33%;" />

**Plots: Add a new curve**

```julia
using Plots
plot(sin, 0, 2π, xlims=(0, 8), ylims=(-2,2))  # No axis(), use xlims/ylims instead
plot!(cos, 0, 2π)
```

<img src="/media/image-20220920205221394.png" alt="image-20220920205221394" style="zoom:33%;" />



**Plot normal distribution**

```julia
using Plots, Distributions

L1 = Normal(5, 1)
# Normal{Float64}(μ=5.0, σ=1.0)
L2 = Normal(5, 5)
# Normal{Float64}(μ=5.0, σ=5.0)

plot(x->pdf(L1, x), xlims=(-10, 20))
plot!(x->pdf(L2, x))
```



<img src="/media/image-20220920205804346.png" alt="image-20220920205804346" style="zoom:33%;" />



Amend:

Plots 好像更 general? 可以看 backend 是否是 PyPlot 決定是否用：

```julia
# return using_pyplot
function using_pyplot()
    # if using PyPlot, return true
    if typeof(Plots.backend()) == Plots.PyPlotBackend
        return true
    else
        return false
    end
end
```



Plots 如果在 VS code 不 work, 可以強迫 display

```julia
plot(cir_x, cir_y, color="red")
p = plot!(lin_cir_x, lin_cir_y, color="blue")
display(p)
```

