#!/usr/local/bin/python3

#%% Setup
import gym
import numpy as np
import matplotlib.pyplot as plt

SIMULATION_LIMIT = 200
TRAINING_LIMIT = 10000

env = gym.make('CartPole-v0')


def run_episode(env, parameters, render=False):
    observation = env.reset()
    totalreward = 0
    for _ in range(SIMULATION_LIMIT):
        if render:
            env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def train():
    best_reward = 0
    best_parameters = None
    reward_history = []
    for i in range(TRAINING_LIMIT):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters
            print('New best after {:} iterations!\n'.format(i), best_reward,
                  best_parameters)
        reward_history.append(best_reward)
        if best_reward >= SIMULATION_LIMIT:
            # We won't find anything better, so stop
            break
    return best_parameters, reward_history


#%% Training: randomly find good parameters
params, hist = train()

#%% Plot learning progress
plt.plot(hist)
plt.show()

#%% Show our best parameters in a rendered simulation
while True:
    run_episode(env, params, render=True)
