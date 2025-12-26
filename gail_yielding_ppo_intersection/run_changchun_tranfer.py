import os
import json
import argparse
import pickle
import numpy as np
import torch
import gym
from typing import Tuple, Any, List

from models.ppo import PPO


# ---------------------------
# Utils
# ---------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_reset(env) -> np.ndarray:

    out = env.reset()
    ob = out[0] if isinstance(out, tuple) else out
    return np.asarray(ob, dtype=np.float32)


def env_step(env, action: Any) -> Tuple[np.ndarray, float, bool, dict]:

    out = env.step(action)
    if len(out) == 5:
        ob, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        ob, r, done, info = out
    return np.asarray(ob, dtype=np.float32), float(r), bool(done), info


def ensure_numpy_action(act: Any) -> np.ndarray:

    if isinstance(act, np.ndarray):
        return act.astype(np.float32).ravel()
    if torch.is_tensor(act):
        return act.detach().cpu().numpy().astype(np.float32).ravel()
    return np.array(act, dtype=np.float32).ravel()


def load_config(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "model_config.json"), "r") as f:
        return json.load(f)


def build_model(state_dim: int, action_dim: int, discrete: bool, config: dict, device: str) -> PPO:

    model = PPO(state_dim, action_dim, discrete, config).to(device)
    return model


# ---------------------------
# Rollout per checkpoint
# ---------------------------
def run_one_checkpoint(env,
                       model: PPO,
                       ckpt_path: str,
                       num_episodes: int,
                       action_dim: int) -> List[List[np.ndarray]]:

    if not hasattr(model, "pi"):
        raise RuntimeError("Model has no attribute 'pi' for loading policy weights.")


    state_dict = torch.load(ckpt_path, map_location=get_device())
    model.pi.load_state_dict(state_dict)

    all_episodes = []


    with torch.no_grad():
        for _ in range(num_episodes):
            done = False
            ob = env_reset(env)

            placeholder_act = np.zeros((action_dim,), dtype=np.float32)
            epi = [np.concatenate([ob, placeholder_act], axis=0)]

            while not done:
                act = model.act(ob)           
                act_np = ensure_numpy_action(act)
                ob, _, done, _ = env_step(env, act_np)
                epi.append(np.concatenate([ob, act_np], axis=0))

            all_episodes.append(epi)

    return all_episodes


# ---------------------------
# Main
# ---------------------------
def main(env_name: str,
         num_episodes: int,
         start_epoch: int,
         end_epoch: int,
         step_epoch: int,
         out_dir: str):

    ckpt_root = os.path.join("ckpts", env_name)
    config = load_config(ckpt_root)

    if env_name not in ['intersectionYieldWorld-v1']:
        print("The environment name is wrong!")
        return


    _env = gym.make(env_name)
    try:
        state_dim = 5   # 
        discrete = False
        action_dim = _env.action_space.shape[0]
    finally:
        _env.close()
        del _env

    device = get_device()
    model = build_model(state_dim, action_dim, discrete, config, device)

    os.makedirs(out_dir, exist_ok=True)
    policy_dir = os.path.join(ckpt_root, 'r_lim_ppo')

    epochs = list(range(start_epoch, end_epoch + 1, step_epoch))
    print(f"Planned epochs: {epochs}")

    for epoch in epochs:
        ckpt_name = f"policy_epoch_tianjin_transfer{epoch}.ckpt"
        ckpt_path = os.path.join(policy_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[Skip] {ckpt_path} not found")
            continue


        env = gym.make(env_name)
        print(f"[Run ] epoch={epoch} | ckpt={ckpt_path}")
        try:
            episodes = run_one_checkpoint(env, model, ckpt_path, num_episodes, action_dim)
        except KeyboardInterrupt:
            env.close()
            print("\n[Stop] Interrupted by user.")
            return
        except Exception as e:
            env.close()
            print(f"[Error] epoch={epoch}: {e}")
            continue
        finally:
            env.close()
            del env


        traj_path = os.path.join(out_dir, f"policy_epoch_tianjin_transfer{epoch}.pkl")#policy_epoch_tianjin_transfer30
        with open(traj_path, "wb") as f:
            pickle.dump(episodes, f)

        print(f"[Done] epoch={epoch} -> {traj_path}")

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="intersectionYieldWorld-v1")
    parser.add_argument("--num_episodes", type=int, default=600)
    parser.add_argument("--start_epoch", type=int, default=30)
    parser.add_argument("--end_epoch", type=int, default=270)
    parser.add_argument("--step_epoch", type=int, default=30)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r".\run_tianjin_transfer"
    )
    args = parser.parse_args()

    main(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        step_epoch=args.step_epoch,
        out_dir=args.out_dir
    )
