import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pickle
import random
import pandas as pd


def load_expert():
    #expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert_changchun_merge.csv',index_col=0)
    expert = pd.read_csv('C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert_chongqing_merge.csv',index_col=0)
    return expert
    
loaded_data_list = load_expert()
#car_id=pd.unique(loaded_data_list['id_car'])
car_id=pd.unique(loaded_data_list['track_id'])


class intersectionYieldEnv1(gym.Env):

    def __init__(self):
        self.viewer = None
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,)
        self.observation_space = spaces.Box(low=np.array([3,3,4]),high=np.array([3,4,4]))
        self.seed()
        self.number=0
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]       
    def step(self,a):###[a] 
        self.t=self.t+1
        acc=a[0]
        omega=a[1]       
        
        true_v_d=[self.chosen_trajectory.loc[self.t,'x'],self.chosen_trajectory.loc[self.t,'y']] 
        #['x', 'y', 'v','pi','r']        
        pi_next = self.state[3] + omega
        if abs(omega)<1e-3:
        
            self.state = np.array([
                # x_next
                self.state[0]
                + (self.state[2]) * (np.cos(self.state[3]))
                + 0.5*(acc) * (np.cos(self.state[3])),

                # y_next
                self.state[1]
                + (self.state[2]) * (np.sin(self.state[3]))
                + 0.5*(acc) * (np.sin(self.state[3])),

                # v_next  #### v
                self.state[2] + acc,

                # pi_next #### pi
                self.state[3] + omega,
                # r non change
                compute_risk(
                    self.state[2] + acc,###v
                    [
                        self.chosen_trajectory.loc[self.t, 'ped1_x']- 
                        (            self.state[0]
                + (self.state[2]) * (np.cos(self.state[3]))
                + 0.5*(acc) * (np.cos(self.state[3]))-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped1_x'] != 0 else 0,
                        
                        
                        self.chosen_trajectory.loc[self.t, 'ped1_y']- 
                        (self.state[1]
                + (self.state[2]) * (np.sin(self.state[3]))
                + 0.5*(acc) * (np.sin(self.state[3]))-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped1_y'] != 0 else 0,
                
                
                        self.chosen_trajectory.loc[self.t, 'ped2_x']-
                         (            self.state[0]
                + (self.state[2]) * (np.cos(self.state[3]))
                + 0.5*(acc) * (np.cos(self.state[3]))-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped2_x'] != 0 else 0,                   
                        
                        self.chosen_trajectory.loc[self.t, 'ped2_y']- 
                        (self.state[1]
                + (self.state[2]) * (np.sin(self.state[3]))
                + 0.5*(acc) * (np.sin(self.state[3]))-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped2_y'] != 0 else 0,
                
                
                        self.chosen_trajectory.loc[self.t, 'ped3_x']-
                         (            self.state[0]
                + (self.state[2]) * (np.cos(self.state[3]))
                + 0.5*(acc) * (np.cos(self.state[3]))-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped3_x'] != 0 else 0,                   
                        
                        self.chosen_trajectory.loc[self.t, 'ped3_y']- 
                        (self.state[1]
                + (self.state[2]) * (np.sin(self.state[3]))
                + 0.5*(acc) * (np.sin(self.state[3]))-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped3_y'] != 0 else 0,
                    ]
                )      
            ])
        
        else:
            self.state = np.array([
                # x_next
                self.state[0]
                + (self.state[2] / omega) * (np.sin(pi_next) - np.sin(self.state[3]))
                + (acc / (omega**2)) * (np.cos(pi_next) - np.cos(self.state[3]))
                + (acc / omega) * np.sin(pi_next),
                # y_next
                self.state[1]
                + (self.state[2] / omega) * (-np.cos(pi_next) + np.cos(self.state[3]))
                + (acc / (omega**2)) * (np.sin(pi_next) - np.sin(self.state[3]))
                - (acc/ omega) * np.cos(pi_next),
                # v_next  #### v
                self.state[2] + acc,
                # pi_next #### pi
                pi_next,
                # r 
                compute_risk(
                    self.state[2] + acc,###v
                    [
                        self.chosen_trajectory.loc[self.t, 'ped1_x']- 
                        (            self.state[0]
                + (self.state[2] / omega) * (np.sin(pi_next) - np.sin(self.state[3]))
                + (acc / (omega**2)) * (np.cos(pi_next) - np.cos(self.state[3]))
                + (acc / omega) * np.sin(pi_next)-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped1_x'] != 0 else 0,
                        
                        
                        self.chosen_trajectory.loc[self.t, 'ped1_y']- 
                        (self.state[1]
                + (self.state[2] / omega) * (-np.cos(pi_next) + np.cos(self.state[3]))
                + (acc / (omega**2)) * (np.sin(pi_next) - np.sin(self.state[3]))
                - (acc/ omega) * np.cos(pi_next)-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped1_y'] != 0 else 0,
                
                
                        self.chosen_trajectory.loc[self.t, 'ped2_x']-
                         (            self.state[0]
                + (self.state[2] / omega) * (np.sin(pi_next) - np.sin(self.state[3]))
                + (acc / (omega**2)) * (np.cos(pi_next) - np.cos(self.state[3]))
                + (acc / omega) * np.sin(pi_next)-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped2_x'] != 0 else 0,                   
                        
                        self.chosen_trajectory.loc[self.t, 'ped2_y']- 
                        (self.state[1]
                + (self.state[2] / omega) * (-np.cos(pi_next) + np.cos(self.state[3]))
                + (acc / (omega**2)) * (np.sin(pi_next) - np.sin(self.state[3]))
                - (acc/ omega) * np.cos(pi_next)-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped2_y'] != 0 else 0,
                
                
                        self.chosen_trajectory.loc[self.t, 'ped3_x']-
                         (            self.state[0]
                + (self.state[2] / omega) * (np.sin(pi_next) - np.sin(self.state[3]))
                + (acc / (omega**2)) * (np.cos(pi_next) - np.cos(self.state[3]))
                + (acc / omega) * np.sin(pi_next)-true_v_d[0]) if self.chosen_trajectory.loc[self.t, 'ped3_x'] != 0 else 0,                   
                        
                        self.chosen_trajectory.loc[self.t, 'ped3_y']- 
                        (self.state[1]
                + (self.state[2] / omega) * (-np.cos(pi_next) + np.cos(self.state[3]))
                + (acc / (omega**2)) * (np.sin(pi_next) - np.sin(self.state[3]))
                - (acc/ omega) * np.cos(pi_next)-true_v_d[1]) if self.chosen_trajectory.loc[self.t, 'ped3_y'] != 0 else 0,
                    ]
                )      
            ])
        
        costs = 1   
        x, y, v, pi = self.state[0], self.state[1], self.state[2], self.state[3]

        if (####chongqiong_on
            (x < -14) or (x > 25)
            or (v > 10) or (v < 0)
            or (pi < 2) or (pi > 3.3)
            or (self.t > self.chosen_trajectory.shape[0] - 2)
            or (y > y_fun_max(x))
            or (y < y_fun_min(x))
            or (self.state[4]>1)
        ):
            done = True
        else:
            done = False   
        return self.state.squeeze(), -costs, done, {}

    def reset(self):
        self.t=0     
                
        self.chosen_trajectory =loaded_data_list[loaded_data_list['track_id']==car_id[self.number]].reset_index(drop=True)#random.choice(loaded_data_list)
        self.state =self.chosen_trajectory.loc[self.t, ['x', 'y', 'v','phi','r']].values
        self.number=(self.number+1)%car_id.shape[0]#55#85#100#46
                     

        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()

def compute_risk(v, ped_coords):
    ped_coords = np.array(ped_coords).reshape(-1, 2)  #  (N_peds, 2)
    r_total = 0.0
    # 设置参数
    if v < 3:  # low
        rx, ry_plus, ry_minus = 4.47, 4.0, 3.97
        bx, by_plus, by_minus = 3.7, 3.08, 3.28
    elif v < 6 and v>3:  # med
        rx, ry_plus, ry_minus = 7.3, 10.36, 5.26
        bx, by_plus, by_minus = 3.5, 2.13, 2.29
    else:  # high
        rx, ry_plus, ry_minus = 9.3, 23.35, 11.5
        bx, by_plus, by_minus = 3.81, 2.0, 2.0
    # Compute the risk field value for each pedestrian's coordinates and sum them up
    for (x, y) in ped_coords:
        if x == 0 and y == 0:  # 
            continue
        if y >= 0:
            exponent = - (np.abs(x / rx) ** bx + np.abs(y / ry_plus) ** by_plus)
        else:
            exponent = - (np.abs(x / rx) ** bx + np.abs(y / ry_minus) ** by_minus)
        r_total += np.exp(exponent)
    return r_total

##Define boundaries to ensure that the generated trajectories stay within reasonable limits

def y_fun_max(x):####chongqiong_on
    return -8.96707281e-06*x**4+8.68578078e-06*x**3 + 2.20474838e-02*x**2 +-5.56983514e-01 *x +-3.70454054e-01
def y_fun_min(x):####chongqiong_on
    return 2.46839867e-05*x**4-1.20925309e-03*x**3 +2.41252410e-02*x**2 + -3.53486550e-01 *x  -4.16808629e+00

       
# def y_fun_max(x):####changchun
    # return -1.08609092e-05*x**4-3.99256072e-04*x**3 + 3.58518590e-02*x**2 -6.30378223e-01 *x +8.43732489e-01
# def y_fun_min(x):####changchun
    # return 1.35540678e-05*x**4 -9.55972566e-04*x**3 +2.13324788e-02*x**2 -1.58695294e-01 *x -4.02169327e+00  
 
# def y_fun_max(x):####tianjin
    # return -1.64039342e-05*x**4+6.81569064e-05*x**3 + 2.13528725e-02*x**2 -4.63326129e-01 *x -1.30486724e+00
# def y_fun_min(x):####tianjin
    # return -2.81277580e-05*x**4+5.92205925e-04*x**3 +2.76954448e-03*x**2 -9.29570509e-02 *x -4.77475453e+00  