import os
import json
import pickle
import argparse
from models.nets import Expert
import torch
import gym
print(gym.__file__)

from models.gail import GAIL


def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    if env_name not in ['GridWorld-v1']:
        print("The environment name is wrong!")
     
        
    
    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    #env.reset()

    state_dim = len(env.observation_space.high)

    discrete = False
    action_dim = env.action_space.shape[0]
    print('action_dim',action_dim)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"



    model = GAIL(state_dim, action_dim, discrete, config[env_name], seed=2).to(device)##2这个种子还可

    results = model.train(env)

    #env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="GridWorld-v1")
    args = parser.parse_args()

    main(**vars(args))
