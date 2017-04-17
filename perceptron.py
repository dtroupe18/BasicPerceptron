# a single perceptron
import numpy as np


class Perceptron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        """
        Fit training data
        :param X: Training vectors, X.shape: [samples, features]
        :param y: Target value, y.shape : [samples]
        :return:
        """

        # weights
        self.weight = np.zeros(1 + X.shape[1])

        # Number of misclassifications
        self.errors = []

        for i in range(self.niter):
            err = 0
            for xi, target in zip(X, y):
                delta_w = self.rate * (target - self.predict(xi))
                self.weight[1:] += delta_w * xi
                self.weight[0] += delta_w
                err += int(delta_w != 0)
            self.errors.append(err)
        return self

    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        """ Return class label after unit step """

        return np.where(self.net_input(X) >= 0.0, 1, -1)


