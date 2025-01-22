import numpy as np
import torch
from torch.nn import Module
from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
from torch.cuda import FloatTensor
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import pickle
import sys
sys.path.append('C:/Users/14487/python-book/yielding_imitation/my_code/gail_yielding_ppo/models')
import random
from experts import load_expert
#with open('C:/Users/14487/python-book/驾驶员让行模拟论文/new_list_one.pkl', 'rb') as file:
loaded_data_list = load_expert()


class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
        seed=42
    ) -> None:
        super().__init__()
        self.set_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
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
        state = FloatTensor(state)
        distb = self.pi(state)
        
        action = distb.sample().detach().cpu().numpy()
        return action

    def train(self, env, render=False):
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        num_epochs = self.train_config["num_epochs"]
        minibatch_size = self.train_config["minibatch_size"]
        horizon = self.train_config["horizon"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]
        eps = self.train_config["epsilon"]
        c1 = self.train_config["vf_coeff"]
        c2 = self.train_config["entropy_coeff"]
        normalize_advantage = self.train_config["normalize_advantage"]
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v = torch.optim.Adam(self.v.parameters(), lr)
        
        opt_d = torch.optim.Adam(self.d.parameters())
        exp_rwd_iter = []
        exp_obs = []
        exp_acts = []
      
        ##################RL
        #print('stop')
        rwd_iter_means = []
        for i in range(num_iters):
            #print('i',i)
            expert=loaded_data_list[i]
            D=expert[:,6]
            V=expert[:,7]
            length=expert.shape[0]-1
            exp_obs= FloatTensor(np.column_stack((D[0:length],V[0:length])))
            exp_acts = FloatTensor(expert[:,8][0:length]).unsqueeze(1)#([300, 1])>[32, 2])
            print('exp_obs',exp_obs.shape)
            print('exp_acts',exp_acts.shape)

            rwd_iter = []
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < exp_acts.shape[0]-1:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                ob = env.reset()
                while not done and steps < exp_acts.shape[0]:
                
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
                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
             
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)
                
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                #print('ep_costs',ep_costs.shape)#([100])
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                #print('ep_disc_rets',ep_disc_rets.shape)#[100])
                ep_rets = ep_disc_rets / ep_gms
                #print('ep_rets',ep_rets.shape)#[100]
                rets.append(ep_rets)
                
                #####往下是计算优势
                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1)+gamma_ * next_vals- curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)])
                #print('ep_advs',ep_advs.shape)#[100]
                advs.append(ep_advs)
                #print('len(advs)',len(advs))
                gms.append(ep_gms)
                #print('gms',len(gms))

            rwd_iter_means.append(np.mean(rwd_iter))
            #print("Iterations: {},   Reward Mean: {}".format(i + 1, np.mean(rwd_iter)))
            obs = FloatTensor(np.array(obs))
            #print('obs',obs.shape)#([300, 3])
            acts = FloatTensor(np.array(acts))
            #print('acts',acts.shape)#([300, 1])
            rets = torch.cat(rets)
            #print('rets',rets.shape)#([300])
            advs = torch.cat(advs)
            #print('advs',advs.shape)#([300])
            gms = torch.cat(gms)
            #print('gms',gms.shape)#([300])

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()
            self.pi.eval()
            old_log_pi = self.pi(obs).log_prob(acts).detach()
            
            #####这个地方开始往后开始优化d          
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)
            #print(exp_scores,nov_scores)
            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores))
            #print(loss)
            loss.backward()
            opt_d.step()
            

            ####开始优化v
            self.pi.train()
            self.v.train()
            max_steps = num_epochs * (num_steps_per_iter // minibatch_size)
            #print('max_steps',max_steps)

            for _ in range(max_steps):
                minibatch_indices = np.random.choice(
                    range(steps), minibatch_size, False
                )
                mb_obs = obs[minibatch_indices]
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

                S = mb_distb.entropy()

                opt_pi.zero_grad()
                opt_v.zero_grad()
                loss = (-1) * (L_clip - c1 * L_vf + c2 * S).mean()
                #print(loss)
                loss.backward()
                opt_pi.step()
                opt_v.step()

        return rwd_iter_means
