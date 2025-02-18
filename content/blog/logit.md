---
title: Logit
date: 2025-02-10
tags: ["logit"]
---

In machine learning, the term ‘logit’ frequently appears. While many people feel familiar with it, revisiting the core principles often reveals missing links in our understanding.

At its core, the logit function transforms probabilities (ranging between 0 and 1) into an unbounded scale, mapping them to the entire set of real numbers. In practice, many classification models—particularly neural networks—output logits instead of direct probability estimates. This approach enhances numerical stability in loss computation, a topic worth exploring in more detail another time.
<br></br>

## So, what is Logit?

Logit is also called as log-odds, then what are odds?

The odds of an event are defined as the ratio of the probability of success $q$ to the probability of failure $1-q$:

$$\text{odds} = {q \over 1-q}$$

For example:

•	If $q=0.5$, the odds are 1 (an equal chance of success and failure).

•	If $q>0.5$, the odds are greater than 1 (success is more likely).

•	If $q<0.5$, the odds are less than 1 (success is less likely).

The **logit function** is simply the logarithm of these odds:

$$\text{logit}(q) = \text{log}({q \over 1-q})$$

This transformation maps probabilities $(0,1)$ to the entire real number range $(-\infin, \infin)$

## Logit and Sigmoid

Sigmoid function is basically just the inverse of logit. From the above, we can derive:


$$e^{\text{logit}} = {q\over 1-q}$$
$$e^{-\text{logit}} = {1-q \over q} = {1 \over q} - 1$$
$$q = {1 \over 1+e^{-\text{logit}}}=\sigma(\text{logit})$$


where $\sigma$ is the sigmoid function.


This relationship is the foundation of logistic regression. In logistic regression, we model the probability of a binary outcome by applying the sigmoid function to a linear combination of the input features:


$$q = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$


Taking the logit of both sides gives:


$$\text{logit}(q) = \theta^T x$$


In other words, logistic regression is a linear model—but in the space of log-odds rather than probabilities.