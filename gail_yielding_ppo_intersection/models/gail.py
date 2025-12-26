import numpy as np
import torch
from torch.nn import Module
from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
from torch.cuda import FloatTensor
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import pickle
import random
import os
import pandas as pd
from tqdm import tqdm
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

# ====================== dataloading ======================
def load_expert():
    #expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert.csv',index_col=0)
    expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert_tianjin_merge.csv',index_col=0)    

    return expert

loaded_data_list = load_expert()

# ====================== ckpt  ======================
ckpt_root = "ckpts"
if not os.path.isdir(ckpt_root):
    os.mkdir(ckpt_root)
env_name = 'intersectionYieldWorld-v1'

ckpt_path = os.path.join(ckpt_root, env_name)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

ckpt_rlim_dir = os.path.join(ckpt_path, 'r_lim_ppo')
os.makedirs(ckpt_rlim_dir, exist_ok=True)



class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
        seed=43#42
    ) -> None:
        super().__init__()
        self.set_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        # self.pi = InteractionExtractor()
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = np.array(state, dtype=np.float32)         
        state = FloatTensor(state)
        distb = self.pi(state)
        action = distb.sample().detach().cpu().numpy()
        return action    
        
    def train(self, env, ckpt_path, render=False):
        # ============= train =============
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        num_epochs = self.train_config["num_epochs"]
        minibatch_size = self.train_config["minibatch_size"]
        horizon = self.train_config["horizon"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]     # GAE-λ
        eps = self.train_config["epsilon"]
        c1 = self.train_config["vf_coeff"]
        c2 = self.train_config["entropy_coeff"]
        normalize_advantage = self.train_config["normalize_advantage"]

        # ============= Optimizer =============
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v  = torch.optim.Adam(self.v.parameters(),  lr)
        opt_d  = torch.optim.Adam(self.d.parameters())

        # ============= Expert data =============
        expert = loaded_data_list
        #car_id = pd.unique(expert['id_car'])
        car_id = pd.unique(expert['track_id'])

        # selected car_v, car_x, r as the state, and car_a as the action
        # exp_obs  = torch.FloatTensor(expert.loc[:, ['car_v', 'car_x', 'r']].values)
        # exp_acts = torch.FloatTensor(expert['car_a'].values).unsqueeze(1)
        
        
        exp_obs  = torch.FloatTensor(expert[['x', 'y', 'v','phi','r']].values)
        exp_acts = torch.FloatTensor(expert[['a','omega']].values)

        
        
        

        # ============= Metrics Recording for the Training Process (Newly Added) =============
        # Mean value for each episode (i)
        d_loss_hist   = []  # Discriminator loss
        ppo_loss_hist = []  # PPO overall objective (mean over minibatches per update)
        vf_loss_hist  = []  # Mean of value loss
        entropy_hist  = []  # Mean of policy entropy

        # Additionally retain the return mean: consistent with your original code
        rwd_iter_means = []

        # ============= Main training loop =============
        for i in tqdm(range(500)):
            rwd_iter = []
            obs, acts = [], []
            rets, advs, gms = [], [], []
            steps = 0

            # ------- Sample rollouts until the required number of steps is reached-------
            while steps < exp_obs.shape[0]:
                ep_obs, ep_acts = [], []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms, ep_lmbs = [], []

                done = False
                ob = env.reset()
                t = 0

                while not done and steps < exp_obs.shape[0]:
                    act = self.act(ob)
                    ep_obs.append(ob)
                    obs.append(ob)
                    ep_acts.append(act)
                    acts.append(act)

                    ob, rwd, done, info = env.step(act)
                    ep_rwds.append(rwd)
                    ep_gms.append(gamma_ ** t)
                    ep_lmbs.append(lambda_ ** t)

                    t += 1
                    steps += 1

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                # ------- tensor -------
                ep_obs  = FloatTensor(np.array(ep_obs, dtype=np.float32))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                ep_gms  = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                # ------- Obtain "cost" from the discriminator and compute GAE -------
                # Note: Use detach() to prevent the discriminator's computation graph from being carried into the subsequent GAE computation graph.
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets  = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                ep_rets = ep_disc_rets / ep_gms
                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat((self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1) + gamma_ * next_vals - curr_vals

                # GAE
                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)
                gms.append(ep_gms)

            # ------- Aggregate the trajectories of this round -------
            rwd_iter_means.append(np.mean(rwd_iter))

            obs  = FloatTensor(np.array(obs, dtype=np.float32))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms  = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # ------- Old policy log prob, used for the PPO ratio r-------
            self.pi.eval()
            old_log_pi = self.pi(obs).log_prob(acts).detach()

            # ==================== Optimize discriminator D ====================
            self.d.train()
            device = next(self.d.parameters()).device
            exp_obs  = exp_obs.to(device)
            exp_acts = exp_acts.to(device)

            # Note: obs/acts are created using FloatTensor (on GPU). If your default dtype is not on GPU, you need to use .to(device).
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad(set_to_none=True)
            d_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                nov_scores, torch.ones_like(nov_scores)
            )
            d_loss.backward()
            d_loss_i = d_loss.item()  # Scalar loss of the discriminator for this update
            print('D loss:', d_loss_i)
            opt_d.step()

            # ==================== Optimize policy π and value function V====================
            self.pi.train()
            self.v.train()
            max_steps = num_epochs * (num_steps_per_iter // minibatch_size)

            # Temporarily collect metrics from each minibatch in this update
            ppo_loss_mb = []
            vf_loss_mb  = []
            entropy_mb  = []

            for _ in range(max_steps):
                minibatch_indices = np.random.choice(range(steps), minibatch_size, False)
                mb_obs  = obs[minibatch_indices]
                mb_acts = acts[minibatch_indices]
                mb_advs = advs[minibatch_indices]
                mb_rets = rets[minibatch_indices]

                mb_distb = self.pi(mb_obs)
                mb_log_pi = mb_distb.log_prob(mb_acts)
                mb_old_log_pi = old_log_pi[minibatch_indices]

                r = torch.exp(mb_log_pi - mb_old_log_pi)

                L_clip = torch.minimum(
                    r * mb_advs, torch.clip(r, 1 - eps, 1 + eps) * mb_advs
                )
                L_vf = (self.v(mb_obs).squeeze() - mb_rets) ** 2
                S    = mb_distb.entropy()

                opt_pi.zero_grad(set_to_none=True)
                opt_v.zero_grad(set_to_none=True)

                loss_ppo = (-1) * (L_clip - c1 * L_vf + c2 * S).mean()
                loss_ppo.backward()
                opt_pi.step()
                opt_v.step()

                # Record the scalar metrics of the current minibatch
                ppo_loss_mb.append(loss_ppo.item())
                vf_loss_mb.append(L_vf.mean().item())
                entropy_mb.append(S.mean().item())

            # Push the mean of this update into the history
            d_loss_hist.append(d_loss_i)
            ppo_loss_hist.append(float(np.mean(ppo_loss_mb)))
            vf_loss_hist.append(float(np.mean(vf_loss_mb)))
            entropy_hist.append(float(np.mean(entropy_mb)))

            # ============= Periodically save weights (consistent with your original approach) =============
            if (i + 1) % 5 == 0:
                torch.save(self.pi.state_dict(), os.path.join(ckpt_rlim_dir, f"policy_epoch_chongqing{i + 1}.ckpt"))
                torch.save(self.v.state_dict(),  os.path.join(ckpt_rlim_dir, f"value_epoch_chongqing{i + 1}.ckpt"))
                torch.save(self.d.state_dict(),  os.path.join(ckpt_rlim_dir, f"discriminator_epoch_chongqing{i + 1}.ckpt"))


        log_df = pd.DataFrame({
            "iter": list(range(1, len(d_loss_hist) + 1)),
            "d_loss": d_loss_hist,
            "ppo_loss": ppo_loss_hist,
            "vf_loss": vf_loss_hist,
            "entropy": entropy_hist,
            "mean_return": rwd_iter_means[:len(d_loss_hist)]  # 对齐长度
        })
        os.makedirs(ckpt_path, exist_ok=True)
        log_csv  = os.path.join(ckpt_path, "training_losses_tianjin_transfer.csv")
        log_df.to_csv(log_csv, index=False, encoding="utf-8-sig")


        return rwd_iter_means
        
        
        
        
        
        

