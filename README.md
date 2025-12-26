Deconstructing driving behaviors in interactions with pedestrians at uncontrolled crosswalks: an imitation learning method
================================================================================

OVERVIEW
--------
This repository provides an imitation-learning framework to recover human drivers’ yielding behaviors when interacting
with pedestrians at uncontrolled crosswalks. We introduce a GAIL-PPO approach to clone real-world yielding policies
from expert trajectory data, and then use the learned policy to generate arbitrary numbers of yielding trajectories
in simulation, enabling downstream research on pedestrian crossing behaviors.

Main idea:
- Use GAIL-PPO to clone yielding strategies from real-world expert demonstrations.
- The learned policy can generate unlimited yielding trajectories in simulation for downstream pedestrian studies.
- GAIL-PPO outperforms other baselines, converges during training, and the discriminator loss approaches the
  theoretical optimum in the generator–discriminator game.


HIGHLIGHTS
----------
1) Proposed method: GAIL-PPO for cloning yielding behaviors at uncontrolled crosswalks.
2) Unlimited simulation rollouts: generate any number of yielding trajectories for downstream research.
3) Strong convergence: stable training and better performance than several baselines.
4) Discriminator optimum: discriminator loss nearly reaches the theoretical optimum during adversarial training.


CODE CONTENTS
-------------
This repository provides six different GAIL-style algorithms. Two representative implementations are:

(1) gail_yielding_trpo_intersection
    - A TRPO-based GAIL implementation aligned with Ho & Ermon (2016).
    - TRPO is used as the behavior generator (policy).

(2) gail_yielding_ppo_intersection
    - Our proposed GAIL-PPO implementation.
    - PPO is used as the behavior generator (policy).


REPRODUCTION GUIDE
------------------

1) Dependencies
   - Python 3.7.0
   - torch==1.7.0
   - gym==0.10.5
   - numpy==1.21.6
   - tqdm==4.67.1

   Installation (pip):
   pip install torch==1.7.0 gym==0.10.5 numpy==1.21.6 tqdm==4.67.1


2) Environment
   A basic prerequisite is:
   "You may adapt or rebuild your own environment following Gym APIs."

   We provide a Gym-style environment:
   - intersectionYield_mdp_v1.py


3) Expert Data
   Expert demonstrations extracted from the SinD dataset:
   - expert_chongqing_merge
   - expert_tianjin_merge
   - expert_changchun_merge


4) Training (cloning yielding trajectories from expert data)
   Go to the PPO-based implementation directory:
   cd gail_yielding_ppo_intersection

   Run training:
   python train_yield.py --resume

   Notes:
   - "--resume" indicates whether to resume training from an existing checkpoint
     (or start from scratch depending on your local code setting).


5) Trajectory Generation (roll out learned yielding policy in simulation)
   After training:
   python run_yield.py --num_episodes 100

   Notes:
   - "--num_episodes" sets how many trajectories you want to generate (any number).


CHECKPOINTS AND OUTPUT FILES
----------------------------
For the intersection case, checkpoints are saved under:
gail_yielding_ppo_intersection\ckpts\intersectionYieldWorld-v1\r_lim_ppo

Typically you will see three files:
1) Policy network:
   - Input: state
   - Output: 2D Gaussian distribution over actions (intersection scenario)

2) Value function:
   - Input: state
   - Output: a scalar value

3) Additional training artifact:
   - e.g., discriminator parameters, optimizer state, or metadata (depending on implementation)

Visualization of the policy/value networks corresponds to the analyses reported in Section 5.4 and 5.5 of the paper.


BATCH TESTING (GENERATE TRAJECTORIES UNDER MULTIPLE WEIGHTS)
-----------------------------------------------------------
run_all.bat is provided to generate trajectories using different saved weights, enabling batch evaluation of
trajectory generation performance.


VISUALIZATION
-------------
We visualize generated trajectories across training iterations and observe that they gradually become more human-like.
The file "yielding_traj.gif" demonstrates this evolution process.


REFERENCES
----------
Code reference:
- https://github.com/hcnoh/gail-pytorch

Papers:
- Ho, J. & Ermon, S. (2016). Generative Adversarial Imitation Learning.
  https://arxiv.org/abs/1606.03476

- Related work:
  https://arxiv.org/abs/2209.02297


CITATION
--------
If you use this repository in your research, please cite:
Deconstructing driving behaviors in interactions with pedestrians at uncontrolled crosswalks: an imitation learning method

(You may replace this section with the official BibTeX once available.)
