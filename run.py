import gym
import numpy as np
import time

total_epochs, total_penalties = 0, 0
episodes = 1

env = gym.make("Taxi-v3").env

qTable = np.load('qTable.npy')

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(qTable[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        time.sleep(0.1)
        print(env.render(), end="")
        epochs += 1
        
    total_penalties += penalties
    total_epochs += epochs