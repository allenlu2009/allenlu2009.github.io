---
title: 增進工程師效率 Julia Linear Algebra
---

# Use Julia for Linear Algebra


```julia
using LinearAlgebra
```


```julia
A = [1 2 3; 2 3 4; 4 5 6]
```




    3×3 Array{Int64,2}:
     1  2  3
     2  3  4
     4  5  6




```julia
eigvals(A)
```




    3-element Array{Float64,1}:
     10.830951894845311     
     -0.8309518948453025    
      1.0148608166285778e-16




```julia
det(A)
```




    0.0




```julia
x = range(0, 10, length=1000)
```




    0.0:0.01001001001001001:10.0




```julia
using PyPlot
grid()
plot(x, sin.(x))
```


![png](/media/output_6_0.png)





    1-element Array{PyCall.PyObject,1}:
     PyObject <matplotlib.lines.Line2D object at 0x140288630>




```julia

```
