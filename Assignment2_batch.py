import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from cartpole_environment import CartPoleEnv

###SETUP###
gamma = 0.995
num_actions = 2
state_shape = 4
policy = 'egreedy'
experience_replay = True
decay_epsilon = True #This is to perform dynamic epsilon-greedy exploration by decaying the value of epsilon 
#with time

epsilon = 1  # Epsilon greedy parameter
epsilon_min = 0.01
decay_rate = 0.9995
batch_size = 20  # Size of batch taken from replay buffer

game = CartPoleEnv()

###BUILD THE ARCHITECTURE OF THE MODEL###
def build_architecture(learning_rate = 0.001):
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(24, activation = 'relu')(inputs)   #Tried with 100 nodes also, but apparently there's no improvement
    x = layers.Dense(24, activation = 'relu')(x)
    x = layers.Dense(24, activation = 'relu')(x) #Let's see what happens when removing a layer
    outputs = layers.Dense(2, activation = 'linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer = optimizer, loss = 'mse')
    return model

def select_action(state, policy, epsilon, model):
    if policy == 'egreedy':
            if epsilon > np.random.rand(1)[0]:
                action = random.randrange(game.action_space.n)
            else:
                # Predict action Q-values from environment state
                action_probs = model.predict(np.array([state,]))
                # Take best action
                action = np.argmax(action_probs)
    return action

def experience_replay_update(batch_size, len_history, state_history,state_next_history,
                             rewards_history, action_history, done_history, model):
    # Get indices of samples for replay buffers
    indices = np.random.choice(range(len_history), size = batch_size)

    # Using list comprehension to sample from replay buffer
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = np.array([rewards_history[i] for i in indices])
    action_sample = [action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

    # Build the updated Q-values for the sampled future states
    # Q value = reward + discount factor * expected future reward
            
    y_train = model.predict(state_sample)
    for i in range(len(done_sample)):
        if not done_sample[i]:
            y_train[i][action_sample[i]] = rewards_sample[i] + gamma*np.max(model.predict(np.array([state_next_sample[i],])))
        else:
            y_train[i][action_sample[i]] = rewards_sample[i]
    #Train the model
    model.fit(state_sample, y_train, verbose = 0)

def cartpole(n_runs, learning_rate, gamma, policy, epsilon, experience_replay, batch_size, decay_epsilon):
    
    model = build_architecture(learning_rate)
    
    ###Experience replay buffers###
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    # Maximum replay length
    max_memory_length = 1000000
    # Train the model after a fixed number of actions
    run = 0

    for i in range(n_runs):  # Run until solved
        state = game.reset()
        #state = np.reshape(state, [1,state_shape])
        #state = np.array([state,])
        episode_reward = 0
        run += 1
        n_steps = 0
    
        while True:
            #game.render() #Adding this line would show the attempts of the agent in a pop up window.
            n_steps +=1
            #Select an action according to the policy
            action = select_action(state, policy, epsilon, model)
            
             # Decay probability of taking random action
            if decay_epsilon:
                epsilon *= decay_rate
                epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = game.step(action)
            #state_next = np.reshape(state_next, [1,state_shape])

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fixed number of frames and once batch size is reached
            if len(done_history) > batch_size and not done and experience_replay:
                len_history = len(done_history)
                experience_replay_update(batch_size, len_history, state_history, state_next_history, 
                                         rewards_history,
                                         action_history, done_history, model)

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            
            # If done print the score of current run
            if done:
                print("Run:" + str(run) + ", Steps:" + str(n_steps) + ", Epsilon:" + str(epsilon))
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)

    return episode_reward_history
