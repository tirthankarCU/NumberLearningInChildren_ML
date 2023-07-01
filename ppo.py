import argparse
import sys 
sys.path.append('/NLP_RL_Docker_Version/gym-examples')
import numpy as np 
import pandas as pd 
import time
import utils as U
import model as M
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import json
import copy
import subprocess
import gym 
import gym_examples
'''
    model_0: Naive CNN
    model_1: Full NLP
    model_2: Partial NLP
'''

'''
    easy_0
    medium_1
    difficult_2
'''

def run_best_env():
    number=999
    f,s,t = number//100,(number%100)//10,number%10
    exp_action = [0,3]*f + [1,4]*9 + [2,5]*t
    exp_action_indx = 0
    def human_policy(observation):
        nonlocal exp_action_indx
        # action=int(input())
        try:
            action = exp_action[exp_action_indx]
            exp_action_indx += 1
        except Exception as e:
            action = -1
        return action
    dbg=False 
    episodes=1
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array")
    for _ in range(episodes):
        cumulative_reward,steps=0,0
        observation = env.reset(set_no=number,mx_timeSteps=100,seed=42)
        cnt,mx_iter=0,1002
        while steps<mx_iter:
            print(observation['text'])
            if dbg==True:
                plt.imshow(observation['visual'])
                plt.show()
            action = human_policy(observation)  # User-defined policy function
            observation, reward, terminated, info = env.step(action)
            cumulative_reward+=reward
            steps+=1
            if dbg==True:
                print(f'cumulative_reward {cumulative_reward}; action {action}')
            if terminated:
                break
        print(f'Cumulative Reward ~ {cumulative_reward}; TimeTaken ~ {steps}')
    env.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NLP_RL parameters.')
    parser.add_argument('--model',type=int,help='Type of the model.')
    parser.add_argument('--ease',type=int,help='Level of ease you want to train.')
    args=parser.parse_args()
    run_best_env() #Test it.