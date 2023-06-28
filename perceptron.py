import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)  # +1 para o bias
        self.errors = []

        for _ in range(self.n_iterations):
            error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update  # Bias
                error += int(update != 0.0)
            self.errors.append(error)

    def predict(self, X):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(linear_output >= 0.0, 1, -1)
