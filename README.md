# Yielding_simulation
Deconstructing Human Driver Yielding Policies at Unsignalized Pedestrian Crossings With Generative Adversarial Imitation Learning


Yielding at unsignalized locations is a complex driving behavior often associated with traffic conflicts involving pedestrians. However, relatively few yielding models exist, largely because current simulation tools struggle to capture the inherent randomness and dynamic nature of yielding. This study aims to propose a yielding model by deconstructing the yielding policies of human drivers. First, we propose a novel yielding simulation method that applies Generative Adversarial Imitation Learning (GAIL) within the Distance-velocity (DV) framework to replicate the yielding policies of human drivers. Experimental results demonstrate that the proposed method successfully replicates human drivers' yielding patterns and achieves superior performance compared to other baseline approaches. Second, we propose a yielding decision map where yielding decisions are probabilistically modeled to capture the rational aspects of the human drivers' decision-making process. Simulation results indicate that the proposed decision map effectively reflects the rational decision-making of human drivers and generates human-like yielding trajectories. Finally, we introduce an innovative measure, referred to as `policy sensitivity,' for evaluating the response behavior of road users during interactions. This measure effectively captures how sensitive behavior policies are during traffic conflicts, offering a novel perspective on characterizing behavioral preferences throughout the interaction process. This work advances realistic modeling of vehicle-pedestrian interactions in microscopic simulations.

# Environment

The grid_mdp_v1.py is my custom environment. You need to create your own environment based on GYM.

# Method

Run train_yield.py to extract the expert's evasion policy.
Run test_yield.py to test the expert's evasion policy.


# Requirement 
torch==1.7.0 <br>
gym==0.10.5 <br>
python==3.7 <br>
numpy ==1.21.6 <br>
tqdm== 4.67.1






