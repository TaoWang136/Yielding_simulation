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
sys.path.append('C:/Users/14487/python-book/yielding_imitation/my_code/gail_yielding_pg/models')
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
        use_baseline = self.train_config["use_baseline"]
        opt_d = torch.optim.Adam(self.d.parameters())
        exp_rwd_iter = []
        exp_obs = []
        exp_acts = []
        opt_v = torch.optim.Adam(self.v.parameters(), lr)
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        ##################RL
        #print('stop')
        rwd_iter_means = []
        for i in range(num_iters):
            #print('i',i)
            expert=loaded_data_list[i]
            
            D=expert[:,6]
            V=expert[:,7]
            print(D[0],V[0])
            length=expert.shape[0]-1
            exp_obs= FloatTensor(np.column_stack((D[0:length],V[0:length])))
            exp_acts = FloatTensor(expert[:,8][0:length]).unsqueeze(1)#([300, 1])>[32, 2])

            rwd_iter = []
            obs = []
            acts = []
            rets = []
            gms = []
            cost=[]

            steps = 0
            while steps < exp_acts.shape[0]-1:
                #print('step',steps,exp_acts.shape[0])
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                done = False
                ob = env.reset()
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

                    t += 1
                    steps += 1

                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                ep_gms = FloatTensor(ep_gms)

                
                ep_cost = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_cost
                ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)
                gms.append(ep_gms)
                
                cost.append(ep_cost)

            rwd_iter_means.append(np.mean(rwd_iter))

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            gms = torch.cat(gms)
            cost=torch.cat(cost)
            
            #print('obs,acts,cost',obs.shape,cost.shape)


            
            #####这个地方开始往后开始优化d          
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            print('exp_scores',exp_obs.shape,obs.shape)
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
            #if normalize_return:
            rets = (rets - rets.mean()) / rets.std()
            if use_baseline:
                print('True')
                self.v.eval()
                delta = (rets - self.v(obs).squeeze()).detach()
                #print('delta',delta)

                self.v.train()
                opt_v.zero_grad()
                loss = (-1) * gms * delta * self.v(obs).squeeze()
                loss.mean().backward()
                opt_v.step()
            
            
            ##########开始训练pi
            self.pi.train()
            distb = self.pi(obs)
            opt_pi.zero_grad()
            if use_baseline:
                loss = (-1) * gms * delta * distb.log_prob(acts)
            else:
                loss = (-1) * gms * distb.log_prob(acts) * rets
            loss.mean().backward()
            opt_pi.step()
            new_params = get_flat_params(self.pi).detach()
                        
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy
            set_params(self.pi, new_params)

        return rwd_iter_means
