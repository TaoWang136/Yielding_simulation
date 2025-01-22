import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pickle
import random
import sys
sys.path.append('C:/Users/14487/python-book/yielding_imitation/my_code/gail_yielding_gae/models')
from experts import load_expert
#with open('C:/Users/14487/python-book/驾驶员让行模拟论文/new_list_one.pkl', 'rb') as file:
loaded_data_list = load_expert()
# import sys
        
#with open('C:/Users/14487/python-book/驾驶员让行模拟论文/new_list_one.pkl', 'rb') as file:
    #loaded_data_list = pickle.load(file)#['locx_ped0','locy_ped1','locx_veh2','locy_veh3',L4''D5'V6','a7']]


class GridEnv1(gym.Env):

    def __init__(self):
        self.viewer = None
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=np.array([-38,0]),high=np.array([41,5]))
        self.seed()
        self.number=-1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        
    def step(self,a):###输入动作[a]
        a = np.clip(a[0], -2,2)
        #print('a',a)
        self.t=self.t+1
        #print('t',self.t)

        self.time_step=self.chosen_trajectory[self.t]##到了下一个时刻[pedx(用环境),pedy(用环境),veh_x(生成),veh_y(环境),v(生成),a]
        costs = 1
        self.state = np.array([self.state[0]-0.417*self.state[1],self.state[1]+0.417*a])
      
        if self.t<self.chosen_trajectory.shape[0]-1:
            done=False
        else:
            done=True
        
        return self.state, -costs, done, {}

    def reset(self):
        self.t=0
        self.chosen_trajectory =loaded_data_list[self.number]#random.choice(loaded_data_list)
        self.time_step=self.chosen_trajectory[self.t]##第一时刻x1,y1,x2,y2,vp,l,d,v,a
        self.state =np.array([self.time_step[6],self.time_step[7]])#d,v
        #self.state =np.array([random.randint(10,35),random.randint(8,13)])
        self.number=self.number+1
        print('self.number',self.number)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()

