# Dados de entrada
import numpy as np
from perceptron import Perceptron
import pandas as pd
import time

def iris():
    tempo_inicial = (time.time()) # em segundos
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

    perceptron = Perceptron(learning_rate=0.1, n_iterations=5)
    perceptron.fit(X, y)
    # Testando com os próprios dados de treinamento
    predictions = perceptron.predict(predict)
    contador = 0
    for valor in range(len(comparacao)):
        if comparacao[valor] == predictions[valor]:
            contador+=1
    resultado = (contador * 100)/ 50
    print(F"Taxa de acerto:{resultado:.2f}%")
    tempo_final = (time.time()) # em segundos
    print(f"{tempo_final - tempo_inicial:.2f} segundos")

def glass():
    tempo_inicial = (time.time()) # em segundos
    df = pd.read_csv('glass.csv', header=None)

    X = np.concatenate((df.iloc[:35,1:10].values,df.iloc[70:105,1:10].values),axis=0)
    y = np.concatenate((df.iloc[:35,10].replace({"1":1}).values,df.iloc[70:105,10].replace({"2":-1}).values),axis=0)

    comparacao = np.concatenate((df.iloc[35:70,10].replace({"1":1}).values,df.iloc[105:140,10].replace({"2":-1}).values),axis=0)

    predict = np.concatenate((df.iloc[35:70,1:10].values,df.iloc[105:140,1:10].values),axis=0)

    perceptron = Perceptron(learning_rate=0.01, n_iterations=10000)
    perceptron.fit(X, y)
    # Testando com os próprios dados de treinamento
    predictions = perceptron.predict(predict)
    contador = 0
    for valor in range(len(comparacao)):
        if comparacao[valor] == predictions[valor]:
            contador+=1
    resultado = (contador * 100)/ 70
    print(F"Taxa de acerto:{resultado:.2f}%")
    tempo_final = (time.time()) # em segundos
    print(f"{tempo_final - tempo_inicial:.2f} segundos")
#glass()
iris()