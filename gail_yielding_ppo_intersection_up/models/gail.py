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

# ====================== 数据加载（与你原先一致） ======================
def load_expert():
    #expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert.csv',index_col=0)
    expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert_tianjin_merge.csv',index_col=0)    

    return expert

loaded_data_list = load_expert()

# ====================== ckpt 目录（与你原先一致） ======================
ckpt_root = "ckpts"
if not os.path.isdir(ckpt_root):
    os.mkdir(ckpt_root)
env_name = 'intersectionYieldWorld-v1'

ckpt_path = os.path.join(ckpt_root, env_name)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)
# 用于保存权重的子目录
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
        # ============= 训练超参 =============
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        num_epochs = self.train_config["num_epochs"]
        minibatch_size = self.train_config["minibatch_size"]
        horizon = self.train_config["horizon"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]     # GAE 的 λ
        eps = self.train_config["epsilon"]
        c1 = self.train_config["vf_coeff"]
        c2 = self.train_config["entropy_coeff"]
        normalize_advantage = self.train_config["normalize_advantage"]

        # ============= 优化器 =============
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v  = torch.optim.Adam(self.v.parameters(),  lr)
        opt_d  = torch.optim.Adam(self.d.parameters())

        # ============= 专家数据（与你原来一致） =============
        expert = loaded_data_list
        #car_id = pd.unique(expert['id_car'])
        car_id = pd.unique(expert['track_id'])

        # 这里你选择了 car_v, car_x, r 作为状态，car_a 作为动作
        # exp_obs  = torch.FloatTensor(expert.loc[:, ['car_v', 'car_x', 'r']].values)
        # exp_acts = torch.FloatTensor(expert['car_a'].values).unsqueeze(1)
        
        
        exp_obs  = torch.FloatTensor(expert[['x', 'y', 'v','phi','r']].values)
        exp_acts = torch.FloatTensor(expert[['a','omega']].values)

        
        
        

        # ============= 训练过程的度量记录（新增） =============
        # 每一轮（i）的均值
        d_loss_hist   = []  # 判别器 loss
        ppo_loss_hist = []  # PPO 总目标（每轮对minibatch均值）
        vf_loss_hist  = []  # value loss 的均值
        entropy_hist  = []  # 策略熵 的均值

        # 额外保留回报均值：与你原代码一致
        rwd_iter_means = []

        # ============= 训练主循环 =============
        for i in tqdm(range(270)):
            rwd_iter = []
            obs, acts = [], []
            rets, advs, gms = [], [], []
            steps = 0

            # ------- 采样 rollouts，直到凑满步数 -------
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

                # ------- 张量化 -------
                ep_obs  = FloatTensor(np.array(ep_obs, dtype=np.float32))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                ep_gms  = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                # ------- 由判别器得到“成本”，并算 GAE -------
                # 注意 detach()：避免把判别器图带入到后面的 GAE 图
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

            # ------- 汇总该轮的轨迹 -------
            rwd_iter_means.append(np.mean(rwd_iter))

            obs  = FloatTensor(np.array(obs, dtype=np.float32))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms  = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # ------- 旧策略 log prob，用于 PPO 比率 r -------
            self.pi.eval()
            old_log_pi = self.pi(obs).log_prob(acts).detach()

            # ==================== 优化判别器 D ====================
            self.d.train()
            device = next(self.d.parameters()).device
            exp_obs  = exp_obs.to(device)
            exp_acts = exp_acts.to(device)

            # 注意：obs/acts 由 FloatTensor 创建（在 GPU），如果你的默认类型不是GPU，需要 .to(device)
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad(set_to_none=True)
            d_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                nov_scores, torch.ones_like(nov_scores)
            )
            d_loss.backward()
            d_loss_i = d_loss.item()  # 本轮判别器标量损失
            print('D loss:', d_loss_i)
            opt_d.step()

            # ==================== 优化策略 π 与价值函数 V ====================
            self.pi.train()
            self.v.train()
            max_steps = num_epochs * (num_steps_per_iter // minibatch_size)

            # 本轮临时收集各 minibatch 的指标
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

                # 记录当前minibatch的标量
                ppo_loss_mb.append(loss_ppo.item())
                vf_loss_mb.append(L_vf.mean().item())
                entropy_mb.append(S.mean().item())

            # 把该轮的均值推入历史
            d_loss_hist.append(d_loss_i)
            ppo_loss_hist.append(float(np.mean(ppo_loss_mb)))
            vf_loss_hist.append(float(np.mean(vf_loss_mb)))
            entropy_hist.append(float(np.mean(entropy_mb)))

            # ============= 定期保存权重（与你原先一致） =============
            if (i + 1) % 30 == 0:
                torch.save(self.pi.state_dict(), os.path.join(ckpt_rlim_dir, f"policy_epoch_tianjin_transfer{i + 1}.ckpt"))
                torch.save(self.v.state_dict(),  os.path.join(ckpt_rlim_dir, f"value_epoch_tianjin_transfer{i + 1}.ckpt"))
                torch.save(self.d.state_dict(),  os.path.join(ckpt_rlim_dir, f"discriminator_epoch_tianjin_transfer{i + 1}.ckpt"))


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

        # 返回便于上层做可视化
        return rwd_iter_means
        
        
        
        
        
        

