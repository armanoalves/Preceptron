# Dados de entrada
import numpy as np
from perceptron import Perceptron

X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5]
])
# Rótulos correspondentes
y = np.array([1, 1, 1, -1, -1, -1])

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)

# Testando com os próprios dados de treinamento
predictions = perceptron.predict(X)

print("Rótulos verdadeiros:", y)
print("Rótulos previstos:", predictions)