---
layout: post
title:  "Symmetrized TD Learning"
date:   2026-07-07
categories: representation learning
---
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<style>
  .post-abstract {
    background-color: #fcfcfc;
    border-left: 5px solid #2a7ae2;
    padding: 20px;
    margin: 30px 0;
    font-style: italic;
    font-size: 0.95em;
    color: #444;
    line-height: 1.6;
  }
</style>

<div class="post-abstract">
I continue examining when deep TD learning fits the value function, following up on the theoretical results of Tang and Munos (2023). Since the symmetry of the "key" matrix $A\doteq D(I-\gamma P)$ is crucial to the equivalence between supervised value learning and TD learning, I consider whether TD learning can be modified to use the symmetric portion of $A$ only. In the end, I have some empirical results testing the "symmetrized" TD learning. Unfortunately, the symmetrized TD learning is not of obvious practical value, since it requires using $A^{-1}$, which amounts to solving the MDP. 

Still, this post provides a concise derivation of the key results, and empirical validation.
</div>

## Refresher
The last post derived theoretical conditions for when TD learning fits the value function. Earlier theoretical results (Tang and Munos (2023)), found that if the Markov Chain is *reversible* ($\mu_i P_{ij} = \mu_j P_{ji}$), then TD-Learning is the exact same as gradient descent on a symmetrized value error. Let our estimate be $v_\theta$  and the value error vector be $e=V-v_\theta$. Finally, recall the definition of the *key matrix* $A=D(I-\gamma P)$. Then, the value error is the following:

$$
\begin{align}
E(v) &\doteq \frac{1}{2} e^T A e \\
&= \frac{1}{2} e^\top S e
\end{align}
$$

Where $S = \frac{1}{2}(A + A^\top)$, and the second line follows from the fact that $E$ is quadratic, meaning only the symmetric part of $A$ matters. Since $E$ is a (weighted) norm, it can only be zero if $e=0$. This implies the typical value error $\overline{VE}^2 = e^\top D e^\top$ and $E$ have the same minimum.

### TD Learning
TD learning is based on bootstrapping. Its update is proportional to:

$$
\dot{\theta} \propto (D\delta) \nabla v_\theta
$$

Next, I show that $Ae = D \delta$, giving us the form:

$$
\dot{\theta} \propto \begin{aligned} (D\delta) \nabla v_\theta &= Ae \nabla v_\theta\end{aligned}
$$

Proof that $Ae = D \delta$.

$$
\begin{align}
Ae &= D(I-\gamma P)(V - v_\theta) \quad \text{definition}\\
&= D(R - (I-\gamma P)v_\theta) \quad \text{since } (I-\gamma P)V = R\\
&= D(R  + \gamma Pv_\theta - v_\theta) \\
&= D\delta \\
\end{align}
$$

The takeaway is that the update $\theta$ due to TD learning is proportional to $Ae \nabla v_\theta$. Also, note that $A$ is positive semi-definite since the bellman operator $T$ is a contraction on $\| \cdot \|_D$.[^1].


### Gradient Descent on E
This is *almost* the same as the direction followed by direct gradient descent on $E$, which follows the negative gradient of $E$:

$$
\begin{align}
- \nabla_\theta E &= -\nabla(e^\top A e) \\
&= -\nabla(e^\top S e) \\
&= -\nabla \cdot (V-v_\theta)^\top S (V-v_\theta) \\
&= Se\nabla v_\theta
\end{align}
$$

Tang and Munos used the near equivalence of these two expressions to show that TD learning is the same as gradient descent on $E$ exactly when $S = A$ (the MDP is reversible).

### Symmetrized TD Learning
However, in general, $A$ is not symmetric, and $A = S + K$, with skew-symmetric component $K \doteq \frac{1}{2}(A-A^\top)$. This means that the skew-symmetric component is what can derail TD learning. The natural question is if we can remove the skew-symmetric component. Then, the TD learning update would take a step in the direction of $\nabla_\theta E$. We can rewrite $Se$ as $S(I-\gamma P)^{-1} \delta$. This yields an update of the following form, weighting the TD error by $S(I-\gamma P)^{-1}$ that will minimize $E$.

$$
\dot{\theta} \propto S(I-\gamma P)^{-1} \delta \nabla v_\theta
$$

## Whirlpool Experiments


To test this symmetrized TD learning, I compare to *exact* TD learning. Exact TD learning makes applies the expected update due to TD, rather than constructing the an unbiased estimate from samples. This removes the noise in TD learning. Notice it doesn't depend on a sampled batch: instead it computes $\delta$ for all states, and then weights it by $\mu$ in the final loss.
```
def td_loss(params):
	v = network.apply(params, ALL_STATES) # length N vector
	TD_targets = R_π + λ * P_π @ v
	td_errors = v - stop_gradient(TD_targets)
	loss = 0.5 * jnp.sum(mu * (td_errors ** 2))
return loss
```

For all experiments, I train a convolutional network to evaluate a fixed policy. I use a fixed policy due to training PPO to near convergence. The  I also train with *exact* Monte Carlo (i.e. gradient descent on $e^\top D e$), removing the variance due to noisy estimates of $V$ and noisy gradient estimates. Perhaps unsurprisingly, I find the symmetrized obtains much better value error with a neural network compared to TD learning. More suprisingly, I find it has better value erro than Monte Carlo. In the following experiment, I train a policy to near convergence with PPO. 

Symmetric TD found a significantly stronger policy than exact MC (regression on $V$) or exact TD.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_3/Percent Greedy Correct FourRooms.png' | relative_url }}" alt="Misaligned_TD" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 1: Exact Algorithms performing policy evaluation of a strong policy on Four Rooms.</em></p>
</div>

Surprisingly, gradient descent on $E$ found the strongest policy, suggesting there is something useful about weighting errors by $S$...

### Whirlpool Environment
To "stump" TD, I created a more asymetric environment: a whirlpool with a reward in the center. The start state is a random state in the outer ring, and the transition function pushes the agent clockwise 90% of the time, with the agents intended action succeeding 10% of the time. The value function is the following (note the reward is given on the transition *into* the center state, following the standard Gym API, leading to the terminal state having zero value.)

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_3/whirlpool_val.png' | relative_url }}" alt="Misaligned_TD" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 2: Value of the whirlpool MDP. Note it is radially symmetric.</em></p>
</div>
On this environment, TD was much weaker than the other methods, while gradient descent on $E$ and symmetric TD were the strongest.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_3/whirlpool_ve.png' | relative_url }}" alt="Misaligned_TD" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 3: Value Error on of the whirlpool MDP for exact methods (lower is better).</em></p>
</div>

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_3/whirlpool_greedy_acc.png' | relative_url }}" alt="Misaligned_TD" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 4: Greedy Policy prediction accuracy on of the whirlpool MDP for exact methods (higher is better).</em></p>
</div>

### Practical issues
Symmetric TD uses $(I-\gamma P)^{-1}$, essentially requiring solving the Bellman equation.

### Penalizing Skew-Symmetric Portion
A related idea is as follows: subtract $Ke \nabla V$ from the TD update. This is because the difference between the TD update $(Ae \nabla v)$ and gradient of $E$  $(Se \nabla v)$ is exactly $Ke$:

$$\dot{\theta} - \nabla_\theta E = (Ae - Se) \nabla v = Ke \nabla v$$

This implies that $\dot{\theta} - Ke \nabla = \nabla_\theta E$. To make the TD update gradient descent on $E$, simply add $Ke \nabla v$ to the $TD$ update. However, this too requires a value oracle to get $e$, or Monte Carlo to estimate it, removing the advantage of TD learning.

-------
[^1]: Specifically, it holds that $x^\top S x \geq (1-\gamma)\|x\|_D^2$ for all $x$. The proof is as follows. Expanding $x^\top A x = x^\top D x - \gamma x^\top D P x$, we see that the first term is $\|x\|_D^2$ and the second is at most $ \gamma \|x\|_D \|Px\|_D$ by the Cauchy-Shwartz inequality. Since $T$ is a contraction in $\| \cdot \|_D$, $\|Px\|_D \leq \|x\|_D$, implying the second term is at most $\gamma \|x\|_D^2$. Thus, we have $x^\top S x = x^\top A x \geq (1-\gamma) \|x \|_D^2$. 
