#!/usr/local/bin/python3

import gym

env = gym.make('CartPole-v0')
env.reset()
while True:
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
    env.reset()
