#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import matplotlib
matplotlib.use('Qt5Agg') # or TkAgg
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
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
        
    def update(self,s,a,r,s_next,done):
        G = r + self.gamma*np.max([self.Q_sa[s_next, b] for b in range(self.n_actions)])
        if not done:
            self.Q_sa[s,a] += self.learning_rate*(G - self.Q_sa[s,a])
        else:
            self.Q_sa[s,a] += self.learning_rate*(r - self.Q_sa[s,a])
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    s = env.reset()
    budget = n_timesteps
    
    while budget:
        budget = budget - 1
        a = pi.select_action(s, policy = policy, epsilon = epsilon, temp = temp)
        s_next, r, done  = env.step(a)
        pi.update(s, a, r, s_next, done)
        rewards.append(r)
        if done:
            s = env.reset()
        else:
            s = s_next
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.000001) # Plot the Q-value estimates during Q-learning execution
        
    return rewards 

def test():
    
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.25

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.01
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()