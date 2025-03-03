---
title: Regularization
date: 2025-03-01
tags: ["regularization"]
---
Regularization is a fundamental technique in machine learning used to **prevent overfitting**. The most widely used techniques, **L1 and L2 regularization**, introduce a penalty term to the loss function to encourage simpler, more generalizable models.

Overfitting is particularly common in complex models, such as deep neural networks with many parameters. These models can “memorize” the training data, including noise, rather than learning meaningful patterns. Regularization helps mitigate this issue by penalizing large weight values in the model. The core idea is to include the magnitude of model parameters as a penalty term in the loss function:
$$L + \lambda ||w||_p$$

Here, $L$ represents the original loss function, $\lambda$ is the regularization strength, and $||w||_p$ is the $p$-norm of the weight vector $w$.

By minimizing this regularized loss, we encourage the weights to become smaller, therefore reducing model complexity. But why do we want smaller weights?

In the beginning of training, weights are often initialized close to zero. This means the training process initially focuses on minimizing the original loss $L$. Once $L$ reaches a certain level of minimization, the regularization term, $\lambda ||w||_p$, starts to play a more significant role, encouraging the model to reduce the magnitude of the weights. If some weights are reduced to near zero, it indicates that these weights have little influence on reducing the overall loss. We can effectively remove these weights without significantly impacting performance. Keeping weights large only increases model complexity without providing any real benefit. In other words, by driving unnecessary weights towards zero, we can simplify the model.

Furthermore, without regularization, model weights can grow excessively large during training as the model attempts to perfectly fit the training data. This overfitting is precisely what we aim to prevent with regularization. By penalizing large weights, we discourage the model from relying too heavily on individual data points, promoting the learning of more general patterns.

# L1 Regularization(Lasso)
L1 regularization adds an absolute value penalty term to the objective loss function:

$$
\begin{align*}
L_{total} &=L+\lambda ||w||_1 \cr 
&=L+\lambda \sum_i^n|w_i|
\end{align*}
$$

where $L_{total}$ is the overall loss function, $\lambda$ is regularization strength, $|w_i|$ represents the absolute value of parameter $w_i$. 

Interesting property of L1 regularization is that it encourages sparsity, meaning that it sets some parameters to exactly zero. Due to this property, L1 penalty becomes useful for feature selection, as it effectively removes irrelevant features by eliminating their corresponding weights.

### Gradient of L1 regularization

$$
\frac{\partial L_{total}}{\partial w_i} = \begin{cases} 
\frac{\partial L}{\partial w_i} + \lambda & \text{if } w_i > 0 \cr
\frac{\partial L}{\partial w_i} - \lambda & \text{if } w_i < 0 \cr
\text{undefined} & \text{if } w_i = 0 
\end{cases}
$$

The gradient of L1 penalty term is constant (either +$\lambda$ or -$\lambda$) regardless of the weight's magnitude. This means that all weights experience the same shrinking force regardless of their size. Therefore, for small weights, L1 completely zeros them out over time. 

# L2 Regularization(Ridge) 
Meanwhile, L2 regularization adds a squared penalty term to the loss function:

$$
\begin{align*}
L_{total} &= L + \lambda ||w||_2 \cr
&= L+\lambda \sum_i^nw_i^2
\end{align*}
$$

### Gradient of L2 regularization
$$ \frac{\partial L_{total}}{\partial w_i}  = \frac{\partial L }{\partial w_i} + 2w_i\lambda $$

Unlike in L1 loss, where the gradient is constant, L2 regularization scales proportionally with the magnitude of weight $w_i$. It means that larger weights have larger shrinking force and smaller weights shrink less. Because L2 regularization applies a force proportional to the weight size, it encourages all weights to be small but typically not exactly zero.

As L2 regularization prevents large weights by punishing them with larger gradient, it promotes smoothness, and helps numerical stability in the model with multicollinearity.

# MAP interpretation
In Bayesian statistics, we aim to find the most probable set of model parameters *given* the observed data and prior belief. Bayes' theorem gives us the posterior distribution: 
$$
p(\theta|x) = {p(x|\theta)p(\theta)\over p(x)}
$$

Maximum a Posterior (MAP) seeks to find $\theta$ that maximizes this posterior probability distribution. In machine learning, we often solve this problem by taking negative logarithm and minimize it. Taking the negative log:

$$
\begin{align*}
\argmin_\theta \\{- \log p(\theta |x)\\} &=\argmin_\theta \\{-\log p(x|\theta) -\log p(\theta) + \log p(x)\\} \cr
&=\argmin_\theta \\{-\log p(x|\theta) -\log p(\theta) \\}
\end{align*}
$$

Here we can find $- \log p(x|\theta)$, which corresponds to usual loss function(negative log likelihood), and $- \log p(\theta)$ from prior distribution over the parameters which will become our regularization term.

### L1 regularization as Laplace prior
A random variable($w$) has a Laplace distribution if its probability density function is:

$$
p(w|\mu,b) = {1\over 2b}e^{(- {|w-\mu| \over b})}
$$

where $\mu$ is a location parameter and $b>0$ is a scale parameter. Let's consider a prior distribution for each parameter $w_i$ to be independent and identically distributed Laplace with mean $0$ and scale parameter $b$:
$$
p(w_i) = {1\over 2b}e^{(- {|w_i| \over b})}
$$

Taking the negative logarithm, $-\log p(w_i)$, we obtain:

$$
\log(2b) + {|w_i| \over b}
$$

Computing the gradient of this:

$$
\frac{\partial}{\partial w_i} {|w_i| \over b} = \begin{cases}
{1\over b} &\text{if } w_i> 0 \cr
-{1\over b} &\text{if } w_i< 0 \cr
\text{undefined} &\text{if } w_i= 0 
\end{cases}
$$

Comparing this gradient to the gradient of the L1 regularization term, we observe that ${1\over b}$ corresponds to $\lambda$. So, minimizing negative log posterior with a Laplace prior is equivalent to minimizing the Loss function with an L1 regularization.

### L2 regularization as Gaussian prior
Gaussian distribution form its probability density function as:
$$
p(w|\mu, \sigma ^2) = {1 \over \sqrt{2 \pi \sigma ^2}} e ^{- {(w-\mu)^2\over 2 \sigma ^2}}
$$
where the parameter $\mu$ is the mean of the distribution and the parameter $\sigma^2$ is the variance.

Let's consider a prior distribution for each parameter $w_i$ to be independent and identically distributed gaussian with mean $0$ and variance $\sigma^2$:
$$
p(w_i) = {1 \over \sqrt{2 \pi \sigma ^2}}e^{- {w_i^2\over 2 \sigma ^2}}
$$

Similar to the L1 regularization case, taking the negative logarithm of $p(w_i)$, we get:
$$
{1\over 2} \log(2\pi\sigma^2) +{w_i^2 \over 2\sigma^2} 
$$

Computing the gradient of this:
$$
\frac{\partial}{\partial w_i} {w_i^2 \over 2\sigma^2}  = {w_i \over \sigma^2}
$$
From above L2 loss, we can put ${1\over \sigma^2} = 2\lambda$, and therefore minimizing negative log posterior with a gaussian prior is equivalent to minimizing the loss function with an L2 regularization term.
## Reference
 <a href="https://en.wikipedia.org/wiki/Laplace_distribution">Wikipedia Laplace distribution</a> 

<a href="https://en.wikipedia.org/wiki/Normal_distribution">Wikipedia Gaussian distribution</a>
 