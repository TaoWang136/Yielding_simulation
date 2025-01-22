import os
import json
import argparse
import pickle
import numpy as np
import torch
import gym

from models.pg import PolicyGradient
# from models.ac import ActorCritic
# from models.trpo import TRPO

# from models.ppo import PPO


def main(env_name, model_name, num_episodes):
    ckpt_path = "ckpts"
    ckpt_path = os.path.join(ckpt_path, env_name)

    with open(os.path.join(ckpt_path, "model_config.json")) as f:
        config = json.load(f)

    if env_name not in ['GridWorld-v1']:
        print("The environment name is wrong!")
        return

    env = gym.make(env_name)
    env.reset()
    state_dim = len(env.observation_space.high)
    discrete = False
    action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "pg":
        model = PolicyGradient(
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
        print('obs',ob)
        init_ob=[np.append(ob,0)]
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
    with open('C:/Users/14487/python-book/驾驶员让行模拟论文/generate_obs_pg.pkl', 'wb') as file:
        pickle.dump(total_obs, file)
    
    
    rwd_std = np.std(rwd_mean)
    rwd_mean = np.mean(rwd_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="GridWorld-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pg",
        help="Type the model name to train. \
            The possible models are [pg, ac, trpo, gae, ppo]"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=165,
        help="Type the number of episodes to run this agent"
    )####记得修改这里233
    args = parser.parse_args()
    main(args.env_name, args.model_name, args.num_episodes)
