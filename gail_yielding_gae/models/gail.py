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
sys.path.append('C:/Users/14487/python-book/yielding_imitation/my_code/gail_yielding_gae/models')
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
      
        ##################RL
        #print('stop')
        rwd_iter_means = []
        for i in range(num_iters):
            expert=loaded_data_list[i]
            D=expert[:,6]
            V=expert[:,7]
            length=expert.shape[0]-1
            exp_obs= FloatTensor(np.column_stack((D[0:length],V[0:length])))
            exp_acts = FloatTensor(expert[:,8][0:length]).unsqueeze(1)#
            # print('exp_obs',exp_obs.shape,exp_acts.shape)
            rwd_iter = []
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < exp_acts.shape[0]-1:
                #print('step',steps,exp_acts.shape[0])
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []
                done = False
                ob = env.reset()
                # print('ob',ob.shape)

                t=0
                while not done and steps < exp_acts.shape[0]:
                
                    act = self.act(ob) ###神经网络拟合，输入状态输出动作。  
                    
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

                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)

                
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1)+gae_gamma * next_vals- curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)])
                advs.append(ep_advs)
                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()
            
            #####这个地方开始往后开始优化d          
            self.d.train()
            # print('exp_obs',exp_obs.shape)
            # print('obs',obs.shape)
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)
            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores))
            loss.backward()
            opt_d.step()
            

            ####开始优化v
            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()
            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
                return hessian

            g = get_flat_grads(((-1)* (self.v(obs).squeeze() - rets) ** 2).mean(), self.v).detach()###就是目标函数，求最小化
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))
            new_params = old_params + alpha * s
            set_params(self.v, new_params)
            
            
            ##########开始训练pi
            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)
                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                #print('obs',obs.shape)
                distb = self.pi(obs)
                old_mean = old_distb.mean.detach()
                old_cov = old_distb.covariance_matrix.sum(-1).detach()
                mean = distb.mean
                cov = distb.covariance_matrix.sum(-1)

                return (0.5) * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - self.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                    ).mean()
            print('kld',kld())
            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()
                return hessian + cg_damping * v
                
            g = get_flat_grads(L(), self.pi).detach()
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()
            new_params = rescale_and_linesearch(g, s, Hs, max_kl, L, kld, old_params, self.pi)
            
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy
            set_params(self.pi, new_params)

        return rwd_iter_means
