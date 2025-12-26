import os
import json
import pickle
import argparse
from models.nets import Expert
import torch
import gym
from models.gail import GAIL
from torch.utils.tensorboard import SummaryWriter


def main(env_name, resume=False):
    # ----- ① Path preparation -----
    ckpt_root = "ckpts"
    os.makedirs(ckpt_root, exist_ok=True)
    if env_name not in ["intersectionYieldWorld-v1"]:
        raise ValueError("Wrong environment name")
    ckpt_path = os.path.join(ckpt_root, env_name)
    os.makedirs(ckpt_path, exist_ok=True)

    # ----- ② Read configuration -----
    with open("config.json") as f:
        config = json.load(f)
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ----- ③ Create environment & model -----
    env = gym.make(env_name)
    state_dim  = 3
    state_dim  = 5
    action_dim = env.action_space.shape[0]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    model = GAIL(state_dim, action_dim, False,
                 config[env_name], seed=3).to(device)

    # ----- ④ If --resume is specified, load the existing weights -----s
    if resume:
        for name, net in [("policy_epoch1200_chongqing", model.pi),
                          ("value_epoch1200_chongqing",  model.v),
                          ("discriminator_epoch1200_chongqing", model.d)]:
            fpath = os.path.join(ckpt_path, f"{name}.ckpt")
            if os.path.isfile(fpath):
                net.load_state_dict(torch.load(fpath, map_location=device))
                print(f"✓ Loaded {name} weights from {fpath}")
            else:
                print('Not loading')

    # ----- ⑤ Training (pass in ckpt_path to allow internal saving as needed) -----
    results = model.train(env, ckpt_path=ckpt_path)   # Needs to be received in train()


    # # ----- ⑥ Save the latest weights after training is complete -----
    # torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy_tianjin.ckpt"))
    # torch.save(model.v .state_dict(), os.path.join(ckpt_path, "value_tianjin.ckpt"))
    # torch.save(model.d .state_dict(), os.path.join(ckpt_path, "discriminator_tianjin.ckpt"))
    # # with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        # pickle.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="intersectionYieldWorld-v1")
    parser.add_argument("--resume",   action="store_true",
                        help="resume training from existing ckpt")
    args = parser.parse_args()
    main(**vars(args))
