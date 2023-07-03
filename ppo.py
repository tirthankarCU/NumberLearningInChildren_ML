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
import gym 
import gym_examples
import time
import numpy as np 
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
    exp_action = [0,3]*f + [1,4]*s + [2,5]*t
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
        mx_iter=60
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


def RESETS(envs):
    global train_set_counter, train_set, args
    set_number=train_set[train_set_counter] if args.ease>=0 else -1
    if train_set_counter>=len(train_set):
        train_set_counter=0
    train_set_counter+=1
    return envs.reset(set_no=set_number)


def STEPS(envs,actions):
    return envs.step(actions.item())


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

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
    global env, max_steps_per_episode, device 
    state = RESETS(env)
    state["visual"] = U.pre_process(state)
    cum_reward=0
    for _ in range(max_steps_per_episode):
        dist, value = model(state["visual"])      
        action = dist.sample()
        next_state, reward, done, info = STEPS(env,action.cpu().numpy())
        next_state["visual"] = U.pre_process(next_state)
        state=copy.deepcopy(next_state)
        cum_reward += reward 
        if done: break
    return cum_reward

def gen_data(opt):
    def sum_digits(no)->int:
        res=0
        while no!=0:
            m=no%10
            res+=m 
            no=no//10
        return res 
    valid=[]
    for i in range(1000):
        if opt==0 and sum_digits(i)<=9:
            valid.append(i)
        elif opt==1 and sum_digits(i)<=15:
            valid.append(i)
        elif opt==2:
            valid.append(i)
    m=int(len(valid)*0.8)
    np.random.shuffle(valid)
    return valid[:m],valid[m:]
                
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NLP_RL parameters.')
    parser.add_argument('--model',type=int,help='Type of the model.')
    parser.add_argument('--ease',type=int,help='Level of ease you want to train.')
    args=parser.parse_args()
    train_set,test_set=gen_data(args.ease)
    train_set_counter=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_episodes = 1000
    max_steps_per_episode_list=[20,32,64,4] # my_estimation
    max_steps_per_episode = max_steps_per_episode_list[args.ease]
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    early_stopping = False
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array")
    # Neural Network Hyper params:
    lr               = 1e-3
    mini_batch_size  = 8
    ppo_epochs       = 4
    if args.model == 0: # Naive model
        model = M.NNModel().to(device) 
    # threshold_reward = envs[0].threshold_reward
    elif args.model == 1: # NLP CNN model
        pass 
    threshold_reward = env.threshold_reward
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_rewards = []
    
    prev_time=time.time()
    while frame_idx < max_frames:
        if early_stopping: break
        if time.time()-prev_time>60:
            prev_time=time.time()
            print(f'% Exec Left {frame_idx*100/max_frames}')
        log_probsArr = []
        valuesArr    = []
        statesArr    = []
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        state = RESETS(env)
        state["visual"] = U.pre_process(state)
        for _iter in range(max_steps_per_episode):
            dist, value = model(state["visual"])
            action = dist.sample()
            next_state, reward, done, info = STEPS(env,action.cpu().numpy())
            next_state["visual"] = U.pre_process(next_state)
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            valuesArr.append(value)
            log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
            rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
            actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
            if args.model == 0:
                statesArr.append(state["visual"])
            else:
                pass
            state = copy.deepcopy(next_state)
            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env(model) for _ in range(5)])
                test_rewards.append([frame_idx,test_reward])
                with open('results/test_reward_list.json', 'w') as file:
                    json.dump(test_rewards, file)
                if test_reward > threshold_reward: early_stop = True
            frame_idx += 1
            if done: break

        _, next_value = model(next_state["visual"])
        returns = compute_gae(next_value, rewardsArr, masksArr, valuesArr)

        returns   = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr    = torch.cat(valuesArr).detach()
        statesArr    = torch.cat(statesArr)
        actionsArr   = torch.cat(actionsArr)
        advantage = returns - valuesArr
        temp_mini_batch_size = min(_iter,mini_batch_size)
        ppo_update(model, optimizer, ppo_epochs, temp_mini_batch_size, statesArr, actionsArr, log_probsArr, returns, advantage)
        torch.save(model.state_dict(),'results/model.ml')