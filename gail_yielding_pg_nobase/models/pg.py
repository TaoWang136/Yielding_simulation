import numpy as np
import torch

from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class PolicyGradient(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

        self.v = ValueNetwork(self.state_dim)

    def get_networks(self):
        if self.train_config["use_baseline"]:
            return [self.pi, self.v]
        else:
            return [self.pi]

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
        horizon = self.train_config["horizon"]
        discount = self.train_config["discount"]
        normalize_return = self.train_config["normalize_return"]
        use_baseline = self.train_config["use_baseline"]

        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        if use_baseline:
            opt_v = torch.optim.Adam(self.v.parameters(), lr)

        rwd_iter_means = []
        rwd_iter = []

        i = 0
        steps = 0
        while i < num_iters:
            print('i',i)
            obs = []
            acts = []
            rwds = []
            disc_rwds = []
            disc = []

            t = 0
            done = False
            ob = env.reset()
            while not done:
                act = self.act(ob)
                obs.append(ob)
                acts.append(act)
                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)
                print('done',done)

                rwds.append(rwd)
                disc_rwds.append(rwd * (discount ** t))
                disc.append(discount ** t)
                t += 1
                steps += 1
                print('steps',steps)
                if steps == num_steps_per_iter:
                    rwd_iter_means.append(np.mean(rwd_iter))
                    print(
                        "Iterations: {},   Reward Mean: {}"
                        .format(i + 1, np.mean(rwd_iter))
                    )
                    i += 1
                    steps = 0
                    rwd_iter = []

                if horizon is not None:
                    if t >= horizon:
                        print('t',t,horizon)
                        done = True
                        break

            rwd_iter.append(np.sum(rwds))
            obs = FloatTensor(np.array(obs))
            print('obs',obs.shape)
            acts = FloatTensor(np.array(acts))
            disc = FloatTensor(disc)
            disc_rets = FloatTensor(
                [sum(disc_rwds[i:]) for i in range(len(disc_rwds))]
            )
            rets = disc_rets / disc

            if normalize_return:
                rets = (rets - rets.mean()) / rets.std()

            if use_baseline:
                self.v.eval()
                delta = (rets - self.v(obs).squeeze()).detach()
                print('delta',delta)

                self.v.train()
                opt_v.zero_grad()
                loss = (-1) * disc * delta * self.v(obs).squeeze()
                loss.mean().backward()
                opt_v.step()

            self.pi.train()
            distb = self.pi(obs)
            opt_pi.zero_grad()
            print('disc',disc)
            print('distb',distb)
            print('acts',acts)
            print('rets',rets)
            
            print('distb.log_prob(acts)',distb.log_prob(acts))
            
            if use_baseline:
                print('disc,delta * distb.log_prob(acts)',disc.shape,delta.shape,distb.log_prob(acts).shape)
                loss = (-1) * disc * delta * distb.log_prob(acts)
            else:
                loss = (-1) * disc * distb.log_prob(acts) * rets
            print('loss',loss.shape)
            loss.mean().backward()
            opt_pi.step()

        return rwd_iter_means
