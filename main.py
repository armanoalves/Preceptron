# Dados de entrada
import numpy as np
from perceptron import Perceptron
import pandas as pd


df = pd.read_csv('dataset.csv')
df = df.iloc[:, :4]
X = df.values
# Rótulos correspondentes
y = np.array([1, 1, 1, -1, -1, -1])

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)

# Testando com os próprios dados de treinamento
predictions = perceptron.predict(X)

print("Rótulos verdadeiros:", y)
print("Rótulos previstos:", predictions)