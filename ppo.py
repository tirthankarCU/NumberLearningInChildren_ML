import argparse
import sys 
import os
sys.path.append(f'{os.getcwd()}/gym-examples')
import numpy as np 
np.random.seed(seed = 2023)
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
import json
import copy
import gym 
import gym_examples
import math
import time
import numpy as np 
from functools import cmp_to_key
import logging 
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
curr_number = -1
def RESETS(envs, override=True):
    global train_set_counter, train_set, args, curr_number
    if not override:
        temp = train_set[-1] if args.ease>=0 else 1
        envs.reset(set_no = temp)
    set_number=train_set[train_set_counter] if args.ease>=0 else 1
    if episodeNo%time_to_learn == 0: #increment only if time to learn has passed.
        if train_set_counter>=len(train_set)-1:
            train_set_counter=0
        train_set_counter+=1
    curr_number = set_number
    return envs.reset(set_no=set_number)


def STEPS(envs,action):
    return envs.step(action)


def ppo_iter(mini_batch_size, states, statesNlp, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(math.ceil(batch_size / mini_batch_size)):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        if args.model == 0:
            yield states[rand_ids, :], states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        else:
            temp_dict = {}
            for r_indx in rand_ids:
                temp = statesNlpArr[r_indx]
                for key,value in temp.items():
                    if key not in temp_dict:
                        temp_dict[key] = value
                    else:
                        temp_dict[key] = torch.cat((temp_dict[key],value),dim = 0)
            yield states[rand_ids, :], temp_dict, actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, statesNlp, actions, log_probs, returns, advantages, clip_param=0.2):
    global frame_idx
    for _ in range(ppo_epochs):
        for state, stateNlp, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, statesNlp, actions, log_probs, returns, advantages):
            if args.model == 0:
                dist, value = model(state)
            elif args.model == 1:
                dist, value = model(state,stateNlp)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            LOG.debug(f'[action | dist_logProb | dist_newlogProb] {action[0]} {dist.log_prob(action[0])} {new_log_probs}')
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.1 * entropy
            LOG.debug(f'Shapes RS[{ratio.shape}], NLPS[{new_log_probs.shape}], OLPS[{old_log_probs.shape}], S1S[{surr1.shape}], S2S[{surr2.shape}], AS[{advantage.shape}]')
            LOG.debug(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            if frame_idx % 1000 == 0:
                LOG.info(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def test_env(model):
    global max_steps_per_episode, device 
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array")
    state = RESETS(env, override=False)
    state["visual"] = U.pre_process(state)
    if args.model == 1:
        state["text"] = U.pre_process_text(model,state)
    cum_reward=0
    for _ in range(max_steps_per_episode):
        if args.model == 0:
            dist, value = model(state["visual"])
        elif args.model == 1:
            dist, value = model(state['visual'],state['text'])   
        action = dist.sample()
        next_state, reward, done, info = STEPS(env,action.item())
        next_state["visual"] = U.pre_process(next_state)
        if args.model == 1:
            next_state["text"] = U.pre_process_text(model,next_state)
        state=copy.deepcopy(next_state)
        cum_reward += reward 
        if done: break
    return cum_reward

def gen_data(opt): 
    if opt<0: return [], []
    global suffix
    def sum_digits(no)->int:
        res=0
        while no!=0:
            m=no%10
            res+=m 
            no=no//10
        return res 
    valid=[]
    for i in range(1,1000):
        if opt==0 and sum_digits(i)<=9:
            valid.append(i)
        elif opt==1 and sum_digits(i)<=15:
            valid.append(i)
        elif opt==2:
            valid.append(i)
    def compare(i1,i2):
        return sum_digits(i1) - sum_digits(i2)
    m=int(len(valid)*0.1)
    np.random.shuffle(valid)
    train, test = valid[:m],valid[m:]
    train = sorted(train,key=cmp_to_key(compare))
    test = sorted(test,key=cmp_to_key(compare))
    with open(f'results/train_set{suffix[args.model][args.ease]}.json', 'w') as file:
        json.dump(train, file)
    with open(f'results/test_set{suffix[args.model][args.ease]}.json', 'w') as file:
        json.dump(test, file)
    return train, test
                
if __name__=='__main__':
    suffix = [['easy','medium','hard','naive'],['fnlp_easy','fnlp_medium','fnlp_hard','fnlp_naive']]
    parser = argparse.ArgumentParser(description = 'NLP_RL parameters.')
    parser.add_argument('--model',type = int, help = 'Type of the model.')
    # 3 is free flow data 1,2,3...
    parser.add_argument('--ease',type = int, help = '(-1,0,1,2,3) Level of ease you want to train.')
    parser.add_argument('--iter',type = int, default=1000, help = 'Control the number of episodes.')
    parser.add_argument('--instr_type',type = int, default=0, help = '(0/1) ~ (policy/state)')
    parser.add_argument('--log',type = int,default = logging.WARN, help = 'import logging module and send logging.INFO or different options.')
    args=parser.parse_args()
    logging.basicConfig(level = args.log, format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
    LOG = logging.getLogger(__name__)
    train_set,test_set=gen_data(args.ease)
    train_set_counter=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_to_learn = 50
    max_episodes = max(time_to_learn*len(train_set),args.iter)
    LOG.info(f'Number of Episodes Tr[{len(train_set)}]*{time_to_learn} = {max_episodes}')
    max_steps_per_episode_list=[25,50,100,5] # my_estimation
    max_steps_per_episode = max_steps_per_episode_list[args.ease]
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    early_stopping = False
    '''
    FOR NEW TYPE OF INSTRUCTION (START)
    '''
    instr_type = "policy" if args.instr_type == 0 else "state"
    if instr_type == "state":
        for suf in suffix:
            for id, word in enumerate(suf):
                suf[id] = word + '_stateInstr' 
    '''
    FOR NEW TYPE OF INSTRUCTION (END)
    '''
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array", instr_type = instr_type)
    # max_advantage = 20
    # Neural Network Hyper params:
    lr               = 5e-5
    mini_batch_size  = 1
    ppo_epochs       = 1
    if args.model == 0: # Naive model
        model = M.NNModel().to(device) 
    # threshold_reward = envs[0].threshold_reward
    elif args.model == 1: # NLP CNN model
        model = MNLP.NNModelNLP().to(device)
    threshold_reward = env.threshold_reward
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_rewards = []
    
    _start_time = time.time()
    prev_time=time.time()
    action_dict, reward_dict = {}, {}
    episodeNo = 0
    while frame_idx < max_frames:
        if early_stopping: break
        if time.time()-prev_time>300: # Every 5 mins
            prev_time=time.time()
            LOG.warning(f'% Exec Left {100-(frame_idx*100/max_frames)}; Time Consumed {time.time()-_start_time} sec')
        log_probsArr = []
        valuesArr    = []
        statesArr,statesNlpArr    = [],[]
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        state = RESETS(env)
        state["visual"] = U.pre_process(state)
        if args.model == 1:
            state["text"] = U.pre_process_text(model,state)
        episodeNo += 1
        extra_padding = 25
        for _iter in range(max_steps_per_episode+extra_padding):
            if args.model == 0:
                dist, value = model(state["visual"])
            elif args.model == 1:
                dist, value = model(state['visual'],state['text'])
            action = dist.sample()
            if action.item() not in action_dict:
                action_dict[action.item()] = 0
            action_dict[action.item()] += 1
            next_state, reward, done, info = STEPS(env,action.item())
            if reward not in reward_dict:
                reward_dict[reward] = 0
            reward_dict[reward] += 1
            LOG.debug(f'Current action[{action.item()}] reward[{reward}]')
            
            next_state["visual"] = U.pre_process(next_state)
            if args.model == 1:
                next_state["text"] = U.pre_process_text(model,next_state)
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            valuesArr.append(value)
            log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
            rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
            actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
            
            if args.model == 0:
                statesArr.append(state["visual"])
            elif args.model == 1:
                statesArr.append(state["visual"])
                statesNlpArr.append(state["text"])
                
            state = copy.deepcopy(next_state)
            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env(model) for _ in range(5)])
                test_rewards.append([frame_idx,test_reward])
                LOG.warning(f'Discovery {action_dict}, {reward_dict}')
                with open(f'results/test_reward_list_{suffix[args.model][args.ease]}.json', 'w') as file:
                    json.dump(test_rewards, file)
                if test_reward > threshold_reward: early_stop = True
            if frame_idx % 5000 == 0:
                LOG.info(f'Saving Model ...')
                torch.save(model.state_dict(),f'results/model_{suffix[args.model][args.ease]}.ml')
            frame_idx += 1
            if done: break
        if args.model ==0:
            _, next_value = model(next_state["visual"])
        elif args.model == 1:
            _, next_value = model(next_state["visual"],next_state['text'])
        returns = compute_gae(next_value, rewardsArr, masksArr, valuesArr)

        returns   = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr    = torch.cat(valuesArr).detach()
        statesArr    = torch.cat(statesArr)
        actionsArr   = torch.cat(actionsArr)
        advantage = returns - valuesArr
        # assert advantage.all() < max_advantage, f'{returns} {valuesArr}'
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, statesArr, statesNlpArr, actionsArr, log_probsArr, returns, advantage)
    torch.save(model.state_dict(),f'results/model_{suffix[args.model][args.ease]}.ml')
    
