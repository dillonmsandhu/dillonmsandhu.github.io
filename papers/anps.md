---
layout: page
title: "Approximate Next Policy Sampling: Replacing Conservative Target Policy Updates in Deep RL"
permalink: /paper/anps/
---

<p style="font-size: 1.1em; margin-bottom: 1.5rem;">
  <strong>Authors:</strong> Dillon Sandhu, Ronald Parr<br>
  <strong>Conference:</strong> <em>Reinforcement Learning Journal (RLJ), 2026</em>
</p>

<!-- Quick Links Section (Stacked or spaced nicely) -->
<div style="margin-bottom: 2rem;">
  <p>
    <a href="{{ '/assets/docs/ANPS RLJ Camera Ready.pdf' | relative_url }}" target="_blank" style="margin-right: 15px;">📄 <strong>Paper PDF</strong></a>
    <a href="https://github.com/dillonmsandhu/next-policy-sampling" target="_blank" style="margin-right: 15px;">💻 <strong>Code</strong></a>
    <!-- Uncomment when poster is ready: -->
    <!-- <a href="{{ '/assets/docs/my-poster.pdf' | relative_url }}" target="_blank">🖼️ <strong>Poster PDF</strong></a> -->
  </p>
</div>

<h3>Abstract</h3>
<blockquote style="font-size: 1.05; color: #555; border-left: 4px solid #ccc; padding-left: 12px; margin-left: 0;">
  We revisit a classic "chicken-and-egg" problem in reinforcement learning: to safely improve a policy, the value function must be accurate on the state-visitation distribution of the updated policy. That distribution over states is unknown and cannot be sampled for the purposes of training the value function. Conservative updates solve this problem, but at the cost of shrinking the policy update. This paper explores an alternative solution, Approximate Next Policy Sampling (ANPS), which addresses the problem by modifying the training distribution rather than constraining the policy update. ANPS is satisfied if the distribution of the training data approximates that of the next policy. To demonstrate the feasibility and efficacy of ANPS, we introduce Stable Value Approximate Policy Iteration (SV-API). SV-API modifies the standard approximate policy iteration loop to hold the target policy fixed while an iteratively updated behavioral policy gathers relevant experience. It only commits to a new policy once a convergence criterion has been met. If certain stability criteria are met, the update is guaranteed to be safe; otherwise, it remains no less safe than standard approximate policy iteration. Applying SV-API to PPO yields Stable Value PPO (SV-PPO), which matches or improves performance on high-dimensional discrete (Atari) and continuous control benchmarks while executing substantially larger target policy updates. These results demonstrate the viability of ANPS as a new solution to this classic challenge in RL.
</blockquote>

<br>

<h3>Citation</h3>
<pre><code>@article{sandhu2026anps,
  title={Approximate Next Policy Sampling: Replacing Conservative Target Policy Updates in Deep RL},
  author={Sandhu, Dillon and Parr, Ronald},
  journal={Reinforcement Learning Journal},
  year={2026}
}</code></pre>