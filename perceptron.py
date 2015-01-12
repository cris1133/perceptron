import numpy as np

class perceptron(object):

    def __init__(self):
        self.weights = []
        self.Nx = 0
        self.activation = 0

    def predict(self, x):
        result = sum(np.array(self.weights)*np.array(x))
        if result > self.activation:
            return 1
        else:
            return -1

    def binarizeY(self, y):
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        return y

    def binarizeX(self, x):
        for N in x:
            for i in range(len(N)):
                if N[i] == 0:
                    N[i] = -1
        return x

    def fit(self, x, y, learning_rate, Niterations):
        y = self.binarizeY(y)
        x = self.binarizeX(x)
        self.Nx = len(x[0])
        self.weights = [0 for n in range(self.Nx)]
        for N in range(Niterations):
            for i in range(len(x)):
                result = self.predict(x[i])
                if result != y[i]:
                    self.weights = [(x[i][n]*result*learning_rate) for n in range(len(self.weights))]
                    self.activation = self.activation + (result*learning_rate)