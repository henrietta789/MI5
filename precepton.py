import random
import time

import numpy as np
import matplotlib.pyplot as plt


# Helper function. Activation function.
def step_function(value):
    return 1 if value > 0 else 0


# Trains the network.
def train(data, results):
    N, n = data.shape
    learning_rate = 0.01
    weights = np.random.randn(n, 1)

    fig = plt.figure()
    plt.title('Decision surface for AND gate')
    ax = fig.add_subplot(111)
    ax.scatter(0, 0, c='r')
    ax.scatter(0, 1, c='r')
    ax.scatter(1, 0, c='r')
    ax.scatter(1, 1, c='g')
    Xs = np.linspace(0, 1, 50)

    # w2 * x2 + w1 * x1 + bias = node input
    # from this we can obtain the building blocks for our line: the slope and the intersection with the y-axis

    # w2 * x2 + w1 * x1 = -b
    # w1 * x1 + b = - w2 * x2
    # x2 = -(w1/w2) * x1 - (b/w2)

    # w1 = weights[0]
    # w2 = weights[1]
    # b = weights[2]

    point_1 = - weights[2] / weights[1]  # if x = 0
    point_2 = - weights[2] / weights[0]  # if y = 0
    m = - point_1 / point_2
    intercept = point_1

    y = (m * Xs) + intercept
    line, = plt.plot(Xs, y)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()

    error = 1
    error_rate = 0.000001
    while error > error_rate:
        error = 0

        for i in range(N):
            first_neuron = step_function(np.dot(data[i], weights))
            iter_error1 = results[i] - first_neuron
            weights += learning_rate * iter_error1 * data[i].reshape(n, 1)
            error += iter_error1 ** 2

        ax.scatter(0, 0, c='r')
        ax.scatter(0, 1, c='r')
        ax.scatter(1, 0, c='r')
        ax.scatter(1, 1, c='g')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title('Decision surface for AND gate')
        plt.grid(True)

        point_1 = - weights[2] / weights[1]
        point_2 = - weights[2] / weights[0]
        m = - point_1 / point_2
        intercept = point_1

        y = (m * Xs) + intercept
        plt.plot(Xs, y)
        plt.show()

    return np.array(weights)


# Evaluates the network.
def evaluate(data, weights, results):
    N, n = data.shape

    for i in range(N):
        first_neuron = step_function(np.dot(data[i], weights))

        if np.array_equal(results[i], first_neuron):
            print(data[i], "recognised as", first_neuron)


# Main
def main():
    data = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]])
    results = np.array([0, 0, 0, 1])

    # Training our network
    weights = train(data, results)

    # Recognising test dataset
    evaluate(data, weights, results)


if __name__ == '__main__':
    main()