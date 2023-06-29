# Dados de entrada
import numpy as np
from perceptron import Perceptron
import pandas as pd

df = pd.read_csv('dataset.csv', header=None)
df.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'species']

# Separar as linhas correspondentes à espécie "setosa"
setosa_half1 = df[df['species'] == 'setosa'].iloc[:len(df[df['species'] == 'setosa']) // 2]
setosa_half2 = df[df['species'] == 'setosa'].iloc[len(df[df['species'] == 'setosa']) // 2:]

# Separar as linhas correspondentes à espécie "versicolor"
versicolor_half1 = df[df['species'] == 'versicolor'].iloc[:len(df[df['species'] == 'versicolor']) // 2]
versicolor_half2 = df[df['species'] == 'versicolor'].iloc[len(df[df['species'] == 'versicolor']) // 2:]

X = np.concatenate((setosa_half1.iloc[:,:4].values,versicolor_half1.iloc[:,:4].values),axis=0)
y = np.concatenate((setosa_half1.iloc[:,4].replace({"setosa":1}).values,versicolor_half1.iloc[:,4].replace({"versicolor":-1}).values),axis=0)

comparacao = np.concatenate((setosa_half2.iloc[:,4].replace({"setosa":1}).values,versicolor_half2.iloc[:,4].replace({"versicolor":-1}).values),axis=0)
predict = np.concatenate((setosa_half2.iloc[:,:4].values,versicolor_half2.iloc[:,:4].values),axis=0)



perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)
# Testando com os próprios dados de treinamento
predictions = perceptron.predict(predict)
contador = 0
for valor in range(len(comparacao)):
    if comparacao[valor] == predictions[valor]:
        contador+=1
resultado = (contador * 100)/ 50
print(F"Taxa de acerto:{resultado:.2f}%")