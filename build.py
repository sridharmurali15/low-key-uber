import gym
import numpy as np
import os

env = gym.make("Taxi-v3").env
env.render()

env.action_space,env.observation_space

qTable = np.zeros((env.observation_space.n,env.action_space.n))

alpha = 0.1
gamma = 0.6
epsilon = 0.1


all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(qTable[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = qTable[state, action]
        next_max = np.max(qTable[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        qTable[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        os.system('clear')
        print(f"Episode: {i}")
        
np.save('qTable.npy', qTable)
print("Training finished.\n")


