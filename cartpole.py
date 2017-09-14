#!/usr/local/bin/python3

import gym

env = gym.make('CartPole-v1')

while True:
    observation = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
