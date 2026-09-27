---
layout: post
title:  "Non-linear Value Learning"
date:   2026-09-25
categories: deep RL
---
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['$$', '$$'], ['\\(', '\\)']],
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
I explain the difference between non-linear value function learning for RL, and show the trajectories taken by each algorithm on the spiral MDP, an environment created by <a href="https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf">Tsitsiklis and Van Roy (1997)</a> to break non-linear TD learning.
</div>

A key concept in RL is learning the value function with *bootstrapping*. But there are several objectives that all make use of bootstrapping, all of which differ in subtle but important ways. For instance, we have *Bellman Error Minimization*, *TD Learning*, *Partially Fitted Value Iteration*, not to mention *Mean Squared Projected Bellman Error*, **oh my**... 

Making things more confusing, standard treatment of these algorithms (e.g. from Sutton and Barto) uses linear value function estimation. In that case, the span of the tangent line of the value estimate is the same as the hypothesis class itself. As we shall see, when the value estimate is not linear in its parameters, intuitions from linear RL break down. 

In this post, I demonstrate the behavior of the following *non-linear* value function learning methods on the spiral MDP:
- *Fitted Value Iteration (FVI)*
- *TD Learning*
- *Bellman Error Minimization*
- *Projected Bellman Error Minimization*
- *Monte Carlo / Supervised Value Learning*

### The Spiral MDP

Tsitsiklis and Van Roy devised a counterexample to demonstrate that TD learning fails on-policy with non-linear function approximation. As we will see, the spiral MDP also breaks fitted value iteration.

The problem is a Markov Reward Process (MRP) and value function estimator. The MRP has three states, thus the value function estimate, called $$v_\theta$$, outputs a three-dimensional vector that depends on a single parameter $$\theta$$. The reward is zero everywhere, meaning the true value is $$V^\pi = \mathbf{0}$$. The magnitude of the estimate $$v_\theta$$ approaches zero as $$\theta \rightarrow -\infty$$.

The value function, and its estimate are in three dimensions. However, $$v_\theta$$ will always live in a two-dimensional subspace. We visualize that subspace in **Figure 1**. The spiral shows our function class -- all possible values of $$v_\theta$$ (projected to the 2-D subspace). The sinusoidal curves show the value function as the $$\theta$$ changes - with each color representing a different state value.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-5.png' | relative_url }}" alt="The Spiral MDP" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 1: The Spiral MDP.</em></p>
</div>

What's nasty about this MRP -- at least from the point of view of TD learning and friends -- is that the Bellman Operator points towards the true value center, but when *fitting it as a stationary target* (like most bootstrapping methods), $$v_\theta$$ fits the Bellman Estimate of the value, yet it diverges from $$V^\pi$$.

A few technical details: the Bellman Operator is closed in this 2D space. If you don't care why, you can skip this paragraph. For those who are curious: because the stationary distribution is uniform ($$\mu = \frac{1}{3}\mathbf{1}$$), the transition matrix $$P$$ is doubly stochastic, meaning $$(1,1,1)$$ is a left eigenvector of $$P$$ with eigenvalue 1 ($$\mathbf{1}^\top P = \mathbf{1}^\top$$). Consequently, for any vector $$x$$ in the 2D subspace orthogonal to $$(1,1,1)$$ (where $$\mathbf{1}^\top x = 0$$), we have $$\mathbf{1}^\top (Px) = (\mathbf{1}^\top P)x = \mathbf{1}^\top x = 0$$. Thus, $$Px$$ remains in this 2D subspace. Because the reward is zero everywhere ($$R = \mathbf{0}$$), the Bellman operator $$Tv = \gamma Pv$$ leaves this subspace invariant.

Technical detail 2: The state-visitation distribution in the spiral MDP is exactly one third times the identity. Thus the on-policy weighted distance reduces to a constant scalar times the Euclidean distance. Technically, all inner products should be weighted by the on-policy distribution (stored in the matrix $$D$$ in my earlier posts), but because $$D = \frac{1}{3}I$$, I ignore it in this post. See the final section of the [original paper](https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf) for the description of the MRP.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-7.png' | relative_url }}" alt="Bellman Operator" style="max-width: 65%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 2: Bellman Operator.</em></p>
</div>

### Algorithm 1: (Partially) Fitted Value Iteration (FVI)

(Partially) Fitted Value Iteration (FVI) is the most common value learning algorithm in deep RL. For example, DQN and PPO both perform FVI. Generally, FVI is thought of as a batch algorithm. It repeatedly fits the Bellman Operator. In expectation, it works as follows:
```
Algorithm: FVI
Repeat:
0. Initialize Target Network u <- v.
1. Estimate Tu = R + γPu.
2. Compute v by minimizing the length of Tu-v.
```

The distinction between batch FVI and PFVI lies entirely in the final step, when a new value estimate is obtained as $$v_\theta \gets \operatorname{argmin}_{v_\theta} \|Tu - v_\theta\|^2$$. In classic linear FVI, the best fit for $$\theta$$ can be obtained using least squares regression. 

Deep variants perform gradient descent on $$\theta$$. But for how many gradient updates? If we fit $$\theta$$ for just one step of gradient descent, we get TD learning. If we run gradient descent until convergence, we get full batch FVI. The crux of the problem is that gradient descent doesn't give us a global minimum. And the local optimum obtained is unfortunately worse than what we started with. 

I run FVI on the spiral problem, with 200 gradient steps per Bellman update. The learning dynamics shown are in **Figure 3**. The left panel shows the parameter value with each update -- notice that it is diverging away from the best estimate $$\theta^* = -\infty$$. The middle panel shows the trajectory on the spiral. The trajectory appears to be going in the wrong direction, despite a better fit being available in the function class. How can this be? As the right panel shows: the loss landscape is non-convex. The global minimum estimator appears to be far to the right, around $$\theta=-5$$. But there is a different local minimum in the other direction. Because the loss landscape is non-convex, each round of FVI gets further away from $$Tv$$. 

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-8.png' | relative_url }}" alt="Learning Dynamics of FVI" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 3: Learning Dynamics of FVI.</em></p>
</div>

Because our $$v_\theta$$ is non-convex, fitting the Bellman operator can lead to a worse value estimate, breaking the contraction argument for why standard exact value iteration converges. In short, the "fitted" component of FVI breaks the guarantee of value iteration. **Figure 4** shows a longer run. Note that the parameters traverse the fixed loss landscape, since the loss changes with each application of $$T$$ (but I only show the first loss).

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-10.png' | relative_url }}" alt="Longer Run of FVI" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 4: Longer Run of FVI.</em></p>
</div>

### Algorithm 2: TD Learning
Tsitsiklis and Van Roy's original 1997 paper analyzed TD learning. As mentioned above, TD learning takes a single gradient step to fit the Bellman Operator before updating the value function. A single gradient step is a *locally linear* approximation to the true optimal direction. As such, each step of TD learning can be understood as taking a small step towards *the projection of* $$Tv_\theta$$ onto *the space orthogonal to the tangent line of* $$v_\theta$$.

**Figure 5** shows the dynamics of TD learning. Again, a single application of the Bellman Operator to the starting value estimate is shown in purple. The green line shows the projection of $$Tv_\theta$$ onto the span of $$\nabla_\theta v$$ (a length 3 vector with entries $$\frac{dv}{d\theta}(s_i)$$ for each state). Notice that this line points away from the center due to the direction of the spiral and angle of $$T$$. 

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-11.png' | relative_url }}" alt="TD Learning" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 5: TD Learning.</em></p>
</div>

```python
def td_update(θ, v, dv, α):
    δ = T(v) - v
    θ = θ + α * dv.T @ δ
    return θ
```

Note that in the linear case, the span of the tangent line *is* the space of representable functions. In this case, FVI and TD learning have the same fixed point.

### Algorithm 3: Bellman Error Minimization (BEM)
Bellman Error Minimization (Baird, L. (1995)) solves $$\min_{\theta} \|Tv_\theta - v_\theta\|_2^2$$. This looks *a lot* like TD learning and FVI. The key difference is that BEM differentiates both $$v_\theta$$ AND $$T v_\theta$$ with respect to $$\theta$$. Whereas TD learning and FVI fix $$T v_\theta$$ (e.g. with a target network).

$$
\begin{align}
\nabla_\theta \|Tv_\theta - v_\theta\|_2^2 &= 2(Tv_\theta - v_\theta)^\intercal \nabla_\theta (Tv_\theta - v_\theta) \\
&= 2 \delta^\intercal (\nabla_\theta T v_\theta - \nabla_\theta v_\theta)
\end{align}
$$

The presence of $$\nabla_\theta T v_\theta$$ means that BEM techniques must incorporate information about how $$Tv$$ changes with $$v$$. More precisely, BEM methods must estimate $$P \nabla_\theta v_\theta$$ -- the expected value of the gradient one step into the future -- as well as $$\delta$$, and take their product. Since the expectation of a product does not equal the product of expectations, a naive sample-based update requires two independent transitions from the same state (the *double sampling issue*). However, just as in GTD methods, one can circumvent double sampling by introducing an auxiliary weight vector to estimate the expected future gradient product without requiring two next-state samples. In our experiments on the spiral problem, we can compute all terms in the gradient exactly.

Results are shown in Figure 6. Note that BEM succeeds on this problem.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-15.png' | relative_url }}" alt="Bellman Error Minimization" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 6: Bellman Error Minimization.</em></p>
</div>

### Algorithm 4: Mean Squared Projected Bellman Error (MSPBE)
The MSPBE is the length of the green vector in Figure 5. Letting $$\Pi_\theta$$ be the linear projection onto the span of $$\nabla_\theta v$$, we have:

$$
\text{MSPBE} = \| \Pi_\theta (T v_\theta - v_\theta) \|^2 = \| \Pi_\theta \delta_\theta \|^2
$$

Since we have just one parameter, $$\Pi_\theta$$ is the $$3\times3$$ matrix:

$$
\Pi_\theta = \frac{1}{\|\nabla_\theta v\|^2} (\nabla_\theta v) (\nabla_\theta v)^\intercal
$$

$$
\text{MSPBE} = (\Pi_\theta \delta_\theta)^\intercal(\Pi_\theta \delta_\theta) = \delta_\theta^\intercal \Pi_\theta^\intercal\Pi_\theta \delta_\theta = \delta_\theta^\intercal \Pi_\theta \delta_\theta
$$

Where the last line follows since repeated application of a projection amounts to a single projection.


$$
\begin{align}
\text{MSPBE} &= \frac{1}{\|\nabla_\theta v\|^2}  \delta_\theta^\intercal (\nabla_\theta v) (\nabla_\theta v)^\intercal \delta_\theta \\
&= \frac{1}{\|\nabla_\theta v\|^2}  (\delta_\theta^\intercal \nabla_\theta v)^2
\end{align}
$$

We see that the numerator is the square of the expected TD update, and the denominator is the square of the length of the tangent vector. 

In practice, there are numerous issues minimizing this quantity. The gradient of the MSPBE contains the product of value network Hessians. Secondly, estimating it requires estimating $$\delta_\theta$$ as a local function of $$\theta$$. Finally, it once again contains a product of expectations, which would naively require double sampling. Methods such as GTD avoid double sampling by using a two-time-scale algorithm with an auxiliary weight vector to track the projected Bellman error.

Still, for our toy problem, we can compute the gradient of the MSPBE exactly - and observe that it works! At each step, it can be thought of as changing $$\theta$$, re-applying the Bellman operator, and measuring the length of the green line. The right panel shows the true MSPBE - or the length of the green vector - as learning progresses.

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-12.png' | relative_url }}" alt="MSPBE Minimization" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 7: MSPBE Minimization.</em></p>
</div>


### Algorithm 5: Monte Carlo
Removing all sampling noise, Monte Carlo value learning is just supervised learning of the center of the circle. Does this suffer from a highly non-convex landscape like FVI?  

It does not. As $$\theta$$ decreases, the error monotonically decreases as well, leading to a convex loss landscape. Note that its loss, shown in the fourth panel of Figure 8, has quite a similar shape to the MSPBE error, which is shown in the third panel. 

<div style="text-align: center; margin: 30px 0;">
  <img src="{{ '/assets/images/blog_post_4/image-13.png' | relative_url }}" alt="Value Error Minimization" style="max-width: 100%; border-radius: 4px;">
  <p style="color: gray; font-size: 0.85em; margin-top: 5px;"><em>Figure 8: Monte Carlo / Value Error Minimization.</em></p>
</div>


### Summary
In short, all one-step methods either fail (TD and FVI), or face the double sampling issue (BEM and MSPBE minimization). In our experiments, we take advantage of access to the true transition matrix and reward function to bypass the double sampling issue. Additionally, we get strong performance from multi-step methods like MC / VEM, due to the absence of sampling variance. 