---
title: Maximum A Posteriori 
date: 2025-02-18
tags: ["maximum-a-posteriori"]
---

In machine learning, we often estimate the parameters of models. Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) are core concepts for parameter estimation in probabilistic models.  Think of MLE and MAP as different ways of making educated guesses. MLE is like making a guess based *only* on the observed data, while MAP uses observed data *plus* any prior knowledge you might already have.

<img src="./giftbox.jpg" alt="image for understanding MAP" style="max-width:600px; height:auto;">

Imagine you're trying to figure out what your partner wants for Valentine's Day. If you see them stop and stare at a jewelry display, you might guess they want something in that category. This is similar to MLE. (A more detailed explanation of MLE can be found in this <a href="../likelihood_MLE/likelihood_MLE.md">previous post</a>).  MAP, on the other hand, incorporates prior beliefs or knowledge in addition to the observed data. Let's say you already know your partner has allergies to metals other than gold. Now, you can narrow your search not only to jewelry but specifically to gold jewelry!


## MAP
MAP uses Bayes' Theorem to combine the likelihood of the data with this prior belief:

$$ P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)} $$

Looking closely, we can see what we used in MLE within this formula. MLE tries to find the parameters $\theta$ that maximize $P(X|\theta)$. Since we're looking for a function of $\theta$, $P(X)$ in the denominator is irrelevant for optimization. The new component is $P(\theta)$, which represents our prior belief.


MAP aims to maximize the *posterior probability* $P(\theta | X)$, which is proportional to $P(X|\theta) P(\theta)$:

$$ \hat{\theta} = \arg\max_{\theta} P(X | \theta) P(\theta) $$

Therefore, MAP considers not only how well the model fits the data ($P(X | \theta)$ - the likelihood) but also how plausible the parameters themselves are ($P(\theta)$ - the prior).

## Reference
 <a href="../likelihood_MLE/likelihood_MLE.md">Previous post about Likelihood and MLE</a>