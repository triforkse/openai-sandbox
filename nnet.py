#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sig2deriv(x):
    return x * (1-x)


class NeuralNet:
    alpha = 1

    def __init__(self, input_size, hidden_size=4, output_size=1):
        self.weights_0_1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.weights_1_2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def train(self, observation, target):
        [layer_0, layer_1, layer_2] = self.predict(observation)

        layer_2_error = target - layer_2
        layer_2_delta = layer_2_error * sig2deriv(layer_2)

        layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
        layer_1_delta = layer_1_error * sig2deriv(layer_1)

        self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
        self.weights_0_1 += self.alpha * layer_0.T.dot(layer_1_delta)

        return np.mean(layer_2_error ** 2)

    def predict(self, observation):
        layer_0 = observation
        layer_1 = sigmoid(np.dot(layer_0, self.weights_0_1))
        layer_2 = sigmoid(np.dot(layer_1, self.weights_1_2))
        return [layer_0, layer_1, layer_2]


def main():
    net = NeuralNet(input_size=4)

    errors = []
    for _ in range(1000):
        observation = np.random.random((4, 10))
        target = np.mean(observation, axis=0, keepdims=True).T
        error = net.train(observation.T, target)
        errors.append(error)

    print(net.predict(np.array([0, 0, 1, 1])))
    print(net.predict(np.array([.2, .2, .2, .2])))
    print(net.predict(np.array([0, .33, .66, 1])))

    plt.plot(errors)
    plt.ylim((0, .1))
    plt.show()

main()
