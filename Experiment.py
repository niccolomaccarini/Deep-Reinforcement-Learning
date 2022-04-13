import numpy as np
import time
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_runs, learning_rate, gamma, policy='egreedy',
                    epsilon=None, smoothing_window=51, plot=False, decay_epsilon):

    reward_results = np.empty([n_repetitions,n_runs]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        rewards = cartpole(n_runs, learning_rate, gamma, policy, epsilon, experience_replay, batch_size, decay_epsilon)
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 10     
    smoothing_window = 1001
    n_runs = 150
    gamma = 0.995

    # Exploration
    policy = 'egreedy' # 'egreedy' or other choices
    epsilon = 1
    epsilon_min = 0.01
    decay_rate = 0.9995
    batch_size = 20
    decay_epsilon = False
    
    # Learning rate
    learning_rate = 0.001
    n = 5
        
    # Plotting parameters
    plot = False
    
    # Nice labels for plotting
    policy_labels = {'egreedy': '$\epsilon$-greedy policy',
                  'softmax': 'Softmax policy'}                  #Something here might need to be changed

    
    ####### Experiments
    
    good_average_reward = 350 # We set this as a benchmark of good average reward reached by the algorithm
    
    Plot = LearningCurvePlot(title = 'Cartpole experiment with $\epsilon$-greedy exploration')    
    policy = 'egreedy'
    epsilons = [0.01, 1 ,0.2]
    for epsilon in epsilons:
        if epsilon == 1:
            decay_epsilon = True
        learning_curve = average_over_repetitions(n_repetitions, n_runs, learning_rate, gamma, policy='egreedy',
                    epsilon=None, smoothing_window=51, plot=False, n=5)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
        decay_epsilon = False
    #policy = 'softmax'
    #temps = [0.01,0.1,1.0]
    #for temp in temps:
     #   learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
      #                                        gamma, policy, epsilon, temp, smoothing_window, plot, n)
       # Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.add_hline(optimal_average_reward_per_timestep)
    Plot.save('cartpole_test.png')

if __name__ == '__main__':
    experiment()
