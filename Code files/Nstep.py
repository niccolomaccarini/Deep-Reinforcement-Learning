#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        actions = list(range(self.n_actions))
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            b = argmax([self.Q_sa[s,b] for b in range(self.n_actions)])
            actions.append(b)
            a = np.random.choice(actions,p=[epsilon/self.n_actions for i in range(self.n_actions)] + [1 - epsilon])
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            probabilities = softmax(np.array(self.Q_sa[s,:]), temp)
            a = np.random.choice(actions, p=probabilities)
            
        return a
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is a terminal state '''
        T_ep = len(states)
        for i in range(T_ep-1):
            m = np.min([self.n, T_ep - i-1])
            if done and i+m == T_ep:
                g = np.sum([gamma**j*rewards[i+j] for j in range(m)])
                self.Q_sa[states[i],actions[i]] += self.learning_rate*(g - self.Q_sa[states_ep[i],actions_ep[i]])
            else:
                g = np.sum([self.gamma**j*rewards[i+j] for j in range(m)]) + self.gamma**m*np.max([self.Q_sa[states[i+m],a] for a in range(self.n_actions)])
                self.Q_sa[states[i],actions[i]] += self.learning_rate*(g - self.Q_sa[states[i],actions[i]])        
        
        
def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []
    budget = n_timesteps
    
    while budget:
        t = -1
        s = env.reset()
        rewards_ep = []
        actions_ep = []
        states_ep = [s]
        for i in range(max_episode_length):   #Collect episode
            budget = budget - 1
            t += 1
            a = pi.select_action(s, policy = policy, epsilon = epsilon, temp = temp)
            s_next, r, done = env.step(a)
            rewards.append(r)
            rewards_ep.append(r)
            actions_ep.append(a)
            states_ep.append(s_next)
            if done:
                break
            if not budget:
                break
            s = s_next
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

        pi.update(states = states_ep, actions = actions_ep, rewards = rewards_ep, done = done)
        
    return rewards 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()