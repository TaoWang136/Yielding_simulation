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
    expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert_chongqing_merge.csv',index_col=0)    

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
ckpt_rlim_dir = os.path.join(ckpt_path, 'r_lim_ac')
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
        
    def train(self, env, ckpt_path,render=False):
        lr = 1e-3
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters())
        opt_v  = torch.optim.Adam(self.v.parameters(),  lr)
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)

             
        # ============= 专家数据（与你原来一致） =============
        expert = loaded_data_list
        #car_id = pd.unique(expert['id_car'])
        car_id = pd.unique(expert['track_id'])

        # 这里你选择了 car_v, car_x, r 作为状态，car_a 作为动作
        # exp_obs  = torch.FloatTensor(expert.loc[:, ['car_v', 'car_x', 'r']].values)
        # exp_acts = torch.FloatTensor(expert['car_a'].values).unsqueeze(1)
        
        
        exp_obs  = torch.FloatTensor(expert[['x', 'y', 'v','phi','r']].values)
        exp_acts = torch.FloatTensor(expert[['a','omega']].values)


                # ==================== 初始化记录列表 ====================
        loss_d_list = []
        loss_v_list = []
        L_pi_list = []
        en=[]

        rwd_iter_means = []
        for i in tqdm(range(1200)):  #tqdm(
            #print('################第i轮',i)

            rwd_iter = []
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []
            steps = 0
            while steps <exp_obs.shape[0]:#exp_acts.shape[0]-1:
                #print('步',steps,exp_acts.shape[0])
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []
                done = False
                ob = env.reset()                             
                t=0
                while not done and steps <exp_obs.shape[0]:
                    #print('状态',ob.shape)
                    act = self.act(ob) ###神经网络拟合，输入状态输出动作。
                    
                    #ob.unsqueeze(0)                     
                    #print('输出动作',act)
                    ep_obs.append(ob)
                    obs.append(ob) 
                    ep_acts.append(act)
                    acts.append(act) 
                    ob, rwd, done, info = env.step(act)
                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)
                    t += 1
                    steps += 1
                    # print('t',t)
                    # print('steps',steps)

                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                ep_obs = FloatTensor(np.array(ep_obs, dtype=np.float32))               
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)
       
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat((self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1)+gae_gamma * next_vals- curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)])
                advs.append(ep_advs)
                gms.append(ep_gms)
                #print('一轮结束')

            
            rwd_iter_means.append(np.mean(rwd_iter))
            #print('变形前',obs)
            obs = FloatTensor(np.array(obs, dtype=np.float32))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)
            if normalize_advantage:
                #print('是否进行了归一化？')
                advs = (advs - advs.mean()) / (advs.std())
            
            #####这个地方开始往后开始优化d          
            self.d.train()
            device = next(self.d.parameters()).device
            exp_obs = exp_obs.to(device)
            exp_acts = exp_acts.to(device)
            #print('exp_obs',exp_obs.shape)#torch.Size([2102, 14])
            #print('obs',obs.shape)#torch.Size([2102, 14])
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss_d = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores))
            loss_d.backward()
            print('loss',loss_d)
            opt_d.step()

            ####开始优化v
            rets = (rets - rets.mean()) / rets.std()
            self.v.eval()
            delta = (rets - self.v(obs).squeeze()).detach()###这里的rets感觉是原文的Q函数，
            self.v.train()
            opt_v.zero_grad()
            loss_v = (-1) * gms * delta * self.v(obs).squeeze()#
            loss_v.mean().backward()
            opt_v.step()
            
            
            ##########开始训练pi
            self.pi.train()
            distb = self.pi(obs)

            opt_pi.zero_grad()
            #print('disc',disc.shape)
            #print('advantage',advantage.shape)
            #print('distb.log_prob(acts)',distb.log_prob(acts).shape)
            #loss = (-1) * disc.unsqueeze(-1) * advantage * distb.log_prob(acts)##我感觉loss的维度不太对unsqueeze(-1
            loss_pi = (-1) * gms* advs.squeeze() * distb.log_prob(acts)
            #print('loss',loss.shape)
            loss_pi.mean().backward()
            #print('loss.mean()',loss.mean())
            opt_pi.step()
            new_params = get_flat_params(self.pi).detach()
                        
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy
            set_params(self.pi, new_params)
            
            if (i + 1) % 50 == 0:
                torch.save(self.pi.state_dict(), os.path.join(ckpt_rlim_dir, f"policy_epoch{i + 1}.ckpt"))
                torch.save(self.v.state_dict(),  os.path.join(ckpt_rlim_dir, f"value_epoch{i + 1}.ckpt"))
                torch.save(self.d.state_dict(),  os.path.join(ckpt_rlim_dir, f"discriminator_epoch{i + 1}.ckpt"))
        
            loss_d_list.append(loss_d.item())
            loss_v_list.append(loss_v.mean().item())
            L_pi_list.append(loss_pi.mean().item())
            en.append(disc_causal_entropy.item())
        
        

        log_df = pd.DataFrame({
            "iter": list(range(1, len(loss_d_list) + 1)),
            "loss_d": loss_d_list,
            "loss_v": loss_v_list,
            "loss_pi": L_pi_list,
            'en':en
        })
        
        
        log_csv  = os.path.join(ckpt_path, "training_losses_ac.csv")

        log_df.to_csv(log_csv, index=False, encoding="utf-8-sig")
       
        return rwd_iter_means
        
        
        
        
        
        
        

