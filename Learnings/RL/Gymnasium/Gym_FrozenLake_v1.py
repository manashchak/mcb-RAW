#%% Imports
import gymnasium as gym
import numpy as np
import matplotlib.pylab as plt
import pickle
from tqdm import tqdm

#%% Defs:
def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    #? Note: "is_slippery=True" introduced randomness in actions. If an action is choosen, let's say right, there is a chance it might in other direction.
    #? This makes it harder to solve with traditional maze solving algorithm.

    #* Init matrix for Q-learning (if training)
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n)) #* init a 64x4 array.
    else:
        f = open("Learnings/RL/Gymnasium/frozen_lake_8x8.pkl", "rb")
        q = pickle.load(f)
        f.close()

    #* Init Q parameters (hyperparameters)
    learning_rate_a = 0.9   #* alpha or learning rate
    discount_factor_g = 0.9 #* gamma or discount rate

    #* Init params for Epsilon-Greedy Policy Algorithm
    epsilon = 1                   #* 1 -> 100% random actions. at the begining.
    epsilon_decay_rate = 0.0001   #* over time epsilon will start to decay from 1 to become less and less random
                                    #? number of episodes = 1/epsilon_decay_rate = 1/0.0001 = 10000 times.
    rng = np.random.default_rng() #* random number generator
    
    #* Tracking
    reward_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = env.reset()[0] #* states: 0 to 63, 0=top-left corner, 63=bottom-right corner
        terminated = False     #* True when fall in the hold or reached goal
        truncated  = False     #* True when actions > 200

        while (not terminated and not truncated):
            #? action selection (Epsilon-Greedy Approach)
            if is_training and rng.random() < epsilon:
                #* sample a random action if < epsilon
                action = env.action_space.sample() #* actions: 0=left, 1=down, 2=right, 3=up
            else:
                #* or take action from q table
                action = np.argmax(q[state,:])
            
            #* take a step based on the chosen action
            new_state, reward, terminated, truncated, _ = env.step(action=action)
            #* Q-learning formula
            if is_training: # only update the q table if is_training
                q[state,action] = q[state,action] + learning_rate_a * ( reward + discount_factor_g*np.max(q[new_state,:]) - q[state,action] )
                state = new_state

    
        #* After each episode, decrease epsilon (Epsilon-Greedy Approach)
        epsilon = max(epsilon - epsilon_decay_rate, 0) #* used max(), because don't let it become -ve! 
        if epsilon == 0:
            learning_rate_a = 0.0001 #* help to stabilize the q values after we no longer 'exploring'
        
        #* Tracking
        if reward == 1:
            reward_per_episode[i] = 1

    env.close()

    #* Plotting
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        #* show running sum of every 100 episodes
        sum_rewards[t] = np.sum(reward_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Number of Rewards/100 Episodes")
    if is_training:
        plt.savefig('Learnings/RL/Gymnasium/frozen_lake_8x8_training.png')
    else:
        plt.savefig('Learnings/RL/Gymnasium/frozen_lake_8x8_eval.png')

    #* Save Q table to a file (if is_training)
    if is_training:
        f = open("Learnings/RL/Gymnasium/frozen_lake_8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()



#%% Main
#? run < (num_episodes, is_training=True, render=False)

if __name__ == '__main__':
    run(15000)






