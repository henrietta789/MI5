from numpy import array,random,dot,linspace
import matplotlib.pyplot as plt
import time

def main():
    datas = array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    results = [0, 0, 0, 1]
    perceptron_learning(datas,results)

def hardlim(val):
    if val<0:
        return 0
    return 1

def perceptron_learning(datas, results):
    plt.ion()
    figure = plt.figure()
    figure.suptitle('AND')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-.5, 1.5))
    plt.ylim((-.5, 1.5))
    plt.grid()
    N, n = datas.shape
    lr=.1
    w=random.randn(n,1)
    E=1
    x = linspace(-5, 5, 30)
    for i in range (N):
        if results[i]==1:
            plt.scatter(datas[i, 1], datas[i, 2], c='black')
        if results[i]==0:
            plt.scatter(datas[i, 1], datas[i, 2], c='red')
    a = [0, -w[0] / w[2]]
    c = [-w[0] / w[1], 0]
    m = (a[1] - a[0]) / (c[1] - c[0])
    plt.plot(x, x * m + a[1])
    while E != 0:
        E = 0
        for i in range(N):
            yi = hardlim(dot(datas[i], w))
            ei = results[i] - yi
            w += lr * ei * datas[i].reshape(n, 1)
            E += ei ** 2
        a = [0, -w[0] / w[2]]
        c = [-w[0] / w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])
        line, = plt.plot(x, x * m + a[1])
        line.set_ydata(x * m + a[1])
        figure.canvas.draw()
        time.sleep(0.5)
        line.remove()
        figure.canvas.flush_events()
        plt.figure(2)
        plt.ylim([-1, 1])
    figure.canvas.flush_events()
