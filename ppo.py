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


def RESETS(envs):
    return envs.reset()


def STEPS(envs,actions):
    return envs.action_step(actions.item())


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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NLP_RL parameters.')
    parser.add_argument('--model',type=int,help='Type of the model.')
    parser.add_argument('--ease',type=int,help='Level of ease you want to train.')
    args=parser.parse_args()
    run_best_env() # Test it.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_episodes = 1000
    max_steps_per_episode = 100
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    early_stopping = False
    env = gym.make('gym_examples/RlNlpWorld-v0',mx_timeSteps=max_steps_per_episode,render_mode="rgb_array")
    # Neural Network Hyper params:
    lr               = 1e-3
    mini_batch_size  = 5
    ppo_epochs       = 4
    if args.model == 0: # Naive model
        model = M.NNModel().to(device) 
    # threshold_reward = envs[0].threshold_reward
    elif args.model == 1: # NLP CNN model
        pass 
    threshold_reward = env.threshold_reward
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_rewards = []
    
    while frame_idx < max_frames and not early_stopping:
        log_probsArr = []
        valuesArr    = []
        statesArr    = []
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        # states = RESETS(envs)
        state = RESETS(env)
        for _ in range(max_steps_per_episode):
            state = torch.FloatTensor([state]).to(device)
            dist, value = model(state)

            action = dist.sample()
            # next_states, rewards, done = STEPS(envs,actions.cpu().numpy())
            next_state, reward, done = STEPS(env,action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            valuesArr.append(value)
            log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
            rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
            actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
            statesArr.append(state)

            state = next_state
            frame_idx += 1

            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env() for _ in range(10)])
                test_rewards.append([frame_idx,test_reward])
                if test_reward > threshold_reward: early_stop = True

        next_state = torch.FloatTensor([next_state]).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewardsArr, masksArr, valuesArr)

        returns   = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr    = torch.cat(valuesArr).detach()
        statesArr    = torch.cat(statesArr)
        actionsArr   = torch.cat(actionsArr)
        advantage = returns - valuesArr
        ppo_update(ppo_epochs, mini_batch_size, statesArr, actionsArr, log_probsArr, returns, advantage)    
        