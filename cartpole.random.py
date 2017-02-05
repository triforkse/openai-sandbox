#!/usr/local/bin/python3

import gym
import tensorflow as tf
import numpy as np

SIMULATION_LIMIT = 1000
TRAINING_LIMIT = 100000

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

# Randomly find good parameters
best_reward = 0
best_parameters = None
for i in range(TRAINING_LIMIT):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env, parameters)
    if reward > best_reward:
        best_reward = reward
        best_parameters = parameters
        print('New best after {:} iterations!'.format(i), best_reward, best_parameters)
    if best_reward >= SIMULATION_LIMIT:
        # We won't find anything better, so stop
        break

# Show our best parameters in a rendered simulation
while True:
    run_episode(env, best_parameters, render=True)
