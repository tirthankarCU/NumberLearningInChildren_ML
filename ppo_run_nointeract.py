import json 
import logging 
import argparse
import sys 
import os
sys.path.append(f'{os.getcwd()}/gym-examples')
import numpy as np 
import pandas as pd 
import time
import utils as U
import model as M
import model_nlp as MNLP
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import gym 
import gym_examples
import copy

logging.basicConfig(filename='console_output.txt', filemode='w', level = logging.INFO, format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
LOG = logging.getLogger(__name__)

def plot_ppo(ip,op):
    arr=[]
    with open(f'{ip}', 'r') as file:
        arr=json.load(file)
        x,y=[arr[i][0] for i in range(len(arr))], [arr[i][1] for i in range(len(arr))]
        plt.plot(x,y)
        plt.xlabel(f'Frame Number')
        plt.ylabel(f'Cumulative Reward')
        plt.savefig(f'{op}/creward.png',dpi=150)
        LOG.info(f'PLOT PATH: {op}/creward.png')
        plt.show()

def run_agent(number,opt):
    def policy(S):
        nonlocal opt
        if opt == 0:
            dist, value = model(S['visual'])
        elif opt == 1:
            dist, value = model(S['visual'],S['text'])
        action = dist.sample()
        return action.cpu().numpy().item()
    dbg=False
    episodes=1
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array", instr_type = instr_type)
    actionArr, rewardArr = [],[]
    for _ in range(episodes):
        cumulative_reward,steps=0,0
        observation = env.reset(set_no=number,seed=42)
        state = copy.deepcopy(observation)
        observation['state'] = U.pre_process(observation)
        if opt == 1:
            observation['text'] = U.pre_process_text(model,observation)
        while True:
            if dbg==True:
                print(observation['text'])
                plt.imshow(state['visual'])
                plt.show()
            action = policy(observation)  # User-defined policy function
            observation, reward, terminated, info = env.step(action)
            actionArr.append(action)
            rewardArr.append(reward)
            state = copy.deepcopy(observation)
            observation['state'] = U.pre_process(observation)
            if opt == 1:
                observation['text'] = U.pre_process_text(model,observation)
            cumulative_reward += reward
            steps += 1
            if terminated: break
    env.close()
    return {"cumulative_reward":cumulative_reward,"action": actionArr, "reward": rewardArr}

if __name__=='__main__':
    '''
    List of models that are available to be tested
    '''
    models_to_test = [['easy','medium','hard','naive'],['fnlp_easy','fnlp_medium','fnlp_hard','fnlp_naive']]
    for model_to_test in models_to_test:
        for id, val in enumerate(model_to_test):
            model_to_test[id] = val 
    parser = argparse.ArgumentParser(description = 'NLP_RL parameters.')
    parser.add_argument('--instr_type',type = int, default = 0, help = '(0/1) ~ (policy/state)')
    parser.add_argument('--full_test', type = int, default = 0, help = '1 to test all numbers.')
    parser.add_argument('--model',type = int, help = 'Type of the model.')
    args=parser.parse_args()
    '''
    FOR NEW TYPE OF INSTRUCTION (START)
    '''
    instr_type = "policy" if args.instr_type == 0 else "state"
    if instr_type == "state":
        for id, model in enumerate(models_to_test):
            models_to_test[id] = model + '_stateInstr'
    '''
    FOR NEW TYPE OF INSTRUCTION (END)
    '''
    with open('test_path.json','r') as file:
        paths = json.load(file)
    for key,value in paths.items(): 
        if key != models_to_test[args.model]: continue
        LOG.info(f'TEST NAME {key}')
        '''
            PLOT Tr Graph.
        '''
        plot_ppo(value["train_result_plot"],value["output_path"])
        '''
            See Results.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if value["model"]["type"] == 0: # CNN MODEL
            model = M.NNModel().to(device)
        elif value["model"]["type"] == 1: # NLP MODEL
            model = MNLP.NNModelNLP().to(device)
        model.load_state_dict(torch.load(f'{value["model"]["path"]}'))
        
        # TRAIN
        train_set, train_dict, avg_cum = [1],{},0
        if len(value["train_set"]) != 0:
            with open(f'{value["train_set"]}','r') as file_tr:
                train_set = json.load(file_tr)
        for no in train_set:
            train_dict[no] = run_agent(no,value["model"]["type"]) #Call function {action, reward,cumulative reward}
            avg_cum += train_dict[no]["cumulative_reward"]
            LOG.info(f'[TRAIN] No[{no}] ~ Reward[{train_dict[no]["cumulative_reward"]}]')
        LOG.info(f'[IMP] train {avg_cum/len(train_set)}')
        with open(f'{value["output_path"]}/train_dict.json', 'w') as file:
            json.dump(train_dict,file)
        LOG.info(f'Train Dict {value["output_path"]}/train_dict.json')
        
        # TEST
        test_set, test_dict, avg_cum = [1],{},0
        if len(value["test_set"]) != 0:
            with open(f'{value["test_set"]}','r') as file_test:
                test_set = json.load(file_test)
        for no in test_set:
            test_dict[no] = run_agent(no,value["model"]["type"])
            avg_cum += test_dict[no]["cumulative_reward"]
            LOG.info(f'[TEST] No[{no}] ~ Reward[{test_dict[no]["cumulative_reward"]}]')
        LOG.info(f'[IMP] test {avg_cum/len(test_set)}')
        with open(f'{value["output_path"]}/test_dict.json','w') as file:
            json.dump(test_dict,file)
        LOG.info(f'Test Dict {value["output_path"]}/test_dict.json')

        # FULL Test 
        if args.full_test == 1:
            full_test = {}
            avg_cum = 0
            for no in range(1, 1000):
                full_test[no] = run_agent(no,value["model"]["type"])
                avg_cum += full_test[no]["cumulative_reward"]
                LOG.info(f'[FULL TEST] No[{no}] ~ Reward[{full_test[no]["cumulative_reward"]}]')
            LOG.info(f'[IMP] test {avg_cum/len(test_set)}')
            with open(f'{value["output_path"]}/full_test_dict.json','w') as file:
                json.dump(full_test,file)
            LOG.info(f'Full Test Dict {value["output_path"]}/full_test_dict.json')  