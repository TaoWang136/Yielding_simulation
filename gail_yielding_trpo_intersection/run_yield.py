import os
import json
import argparse
import pickle
import numpy as np
import torch
import gym
print(gym.__file__)
# from models.pg import PolicyGradient
# from models.ac import ActorCritic
from models.trpo import TRPO



def main(env_name, num_episodes):
    ckpt_path = "ckpts"
    ckpt_path = os.path.join(ckpt_path, env_name)

    with open(os.path.join(ckpt_path, "model_config.json")) as f:
        config = json.load(f)

    if env_name not in ['intersectionYieldWorld-v1']:
        print("The environment name is wrong!")
        return
    env = gym.make(env_name)
    # env.reset()
    state_dim = 5#len(env.observation_space.high)
    discrete = False
    action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    model = TRPO(
        state_dim, action_dim, discrete, config
    ).to(device)

    if hasattr(model, "pi"):
        model.pi.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, "policy.ckpt"), map_location=device
            )
        )

    rwd_mean = []
    total_obs=[]
    for i in range(0,num_episodes):
        rwds = []
        done = False
        ob = env.reset()

        init_ob=[np.append(ob,[0,0])]
        
        obs=[]
        while not done:
            act = model.act(ob)
            ob, rwd, done, info = env.step(act)
            rwds.append(rwd)
            obs.append(np.append(ob,act))
        obs=init_ob+obs   
        total_obs.append(obs)

        rwd_sum = sum(rwds)
        rwd_mean.append(rwd_sum)

    env.close()
    with open('./first_revise/gail_yielding_trpo_intersection/generate_obs_trpo_intersection.pkl', 'wb') as file:
        pickle.dump(total_obs, file)
    
    
    rwd_std = np.std(rwd_mean)
    rwd_mean = np.mean(rwd_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="intersectionYieldWorld-v1")

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Type the number of episodes to run this agent"
    )##
    args = parser.parse_args()
    main(args.env_name, args.model_name, args.num_episodes)
