---
title: Cross Entropy
date: 2025-03-23
tags: ["cross_entropy", "entropy"]
---

In life, there are times when each day seems like a repeat of the previous one, and before you know it, those days have passed in a blink. Then there are moments when a single new experience leaves a lasting impact on us. These memorable events—trips, special occasions, and so on—share one common trait: they don’t happen often. Even though an event may initially surprise us, if it begins to occur regularly, we quickly become accustomed to it, and its impact diminishes.

From the perspective of informational value, everyday experiences like going to school, working, or having lunch and dinner don’t necessarily provide much new “information” for us. In contrast, a trip to a new place, the first day at school and work, those experiences certainly give us a lot of new input.  Most people, for example, are more likely to remember their summer holiday than the routine of work or school throughout the year.

Like this, one way to quantify information is by considering how frequently an event occurs—how surprising it is, a concept also known as *surprisal*. If an event happens frequently, it is not surprising, and therefore, it carries a low informational value. We can think of the magnitude of information as being inversly proportional to the probability of incident $x$, $p(x)$.

$$
\text{Information} \sim {1\over p(x)}
$$

# Entropy is Expected Information
Taking the logarithm of the above equation, we have $\log {1\over p(x)}$. When $p(x)$ is $1$ - a certain event, the informational value of the event is zero, which brings $\log {1\over p(x)}$ to $0$. 

Entropy($H$) is defined as the expected value of information for the distribution $p$. 

$$
\begin{align*}
H(p) &= \mathbb{E}[\log {1\over p(x)}] \cr 
&= \mathbb{E}[- \log p(x)] \cr 
&= - \sum p(x) \log p(x) \cr
\end{align*}
$$

# Cross Entropy is Surprisal Between Two Distributions
The cross entropy between distribution $p$ (the true distribution) and $q$ (the model's predicted distribution) is defined as follows:

$$
H(p,q) = - \sum p(x) \log q(x) 
$$

Entropy measures the expected informational value of an event $x$ under the true distribution $p$. In contrast, cross entropy evaluates the expected suprisal of event $x$ under model's predicted distribution $q$ while assuming weights(probabilities) of event $x$ follow the probability from the true distribution $p$. In other words, cross entropy can be understood as the expected number of bits (basic unit of information) required to encode events from true distribution $p$ when using a coding scheme (or model) optimized for the distribution $q$.

 Cross entropy is important in machine learning because it is one of the most commonly used loss functions for classification tasks. It helps evaluate how well a model’s predictions match the actual distribution.

# Binary Classification Example
For binary classification problems with two possible classes (0 and 1), we represent the target probabilities as $p$ and the model’s predicted probabilities as $q$. Typically, the target probabilities are defined as $p \in \lbrace y, 1-y \rbrace$ and $q \in \lbrace \hat{y}, 1-\hat{y}\rbrace$, where $y$ is the probability of class 1 and $\hat{y}$ is the predicted probability of class 1. The cross entropy between the target and predicted distributions is then defined as:

$$
\begin{align*}
H(p,q) &= -\sum_i p_i \log q_i  \cr
&= -y \log \hat{y} - (1-y) \log (1-\hat{y})
\end{align*}
$$

## Reference
<a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">Wikipedia Entropy</a> 

<a href="https://en.wikipedia.org/wiki/Cross-entropy">Wikipedia Cross entropy</a> 

<a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html">PyTorch CrossEntropyLoss</a> 