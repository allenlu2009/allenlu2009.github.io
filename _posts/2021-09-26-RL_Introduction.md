---
title: Reinforcement Learning 
date: 2021-09-29 23:10:08
categories:
- AI
tags: [ML, VAE, Autoencoder, Variational, EM]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

### Markov Decision Process

1. States

2. (Transition) Model:  transition matrix  T(s, a, s') = Pr(s' \mid s, a)

3. Actions: up, down, left, right A(s)

4. Reward:  R(s) or R(s, a) or R(s, a, s') all math equivalent

* Markovian property: only present matters for the transition model.
* Stationary

Solution => Policy:  $\pi(s) \to a$   and $\pi^*$ = up, up right, right, right

Why policy instead of a plan (trace)

* work everywhere
* robust against probabilistic model
*
* Delayed reward
* Minor reward changes matter => reward is domain knowledge

## Design Reward design the MDP is the key!!! the teacher to learn!! the domain knowledge
