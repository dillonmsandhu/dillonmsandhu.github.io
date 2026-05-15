---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: page
---
<style>
  /* This makes all links on the page blue */
  a {
    color: #007bff !important;
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
  /* This hides the huge banner in the Clean Blog theme */
  .masthead {
    display: none !important;
  }
</style>


<div style="display: flex; align-items: flex-start; gap: 20px; margin-bottom: 30px; margin-top: 100px;">
  <img src="{{ 'assets/images/headshot.png' | relative_url }}" alt="Dillon Sandhu" width="200" style="border-radius: 8px;">
  <div>
    Hello -- I'm Dillon, a Computer Science PhD student at Duke University, advised by Ron Parr. Before graduate school, I worked as a business analyst and economic consultant. 
    <br><br>
    I study Reinforcement Learning, focusing on the relationship between value function approximation, policy optimization, and the distribution of the training data.
    <br><br>
    My intended graduation year is 2028.
  </div>
</div>

## Publications

* **[Approximate Next Policy Sampling: Replacing Conservative Target Policy Updates in Deep RL](https://arxiv.org/abs/2605.05481)** *Dillon Sandhu, Ronald Parr* preprint, forthcoming at RLC, 2026.
