#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@authors: Aydano Machado <aydano.machado@gmail.com>
          Vanessa Vieira <vsv@ic.ufal.br>
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import requests
from sklearn.metrics import accuracy_score

# Lê o arquivo com o dataset
def read_data():
    print('\n - Lendo o arquivo com o dataset sobre diabetes')
    data = pd.read_csv('diabetes_dataset.csv')
    data_app = pd.read_csv('diabetes_app.csv')
    feature_cols = ['Glucose', 'BMI', 'Age']
    X = data[feature_cols]
    X_app = data_app[feature_cols]
    y = data.Outcome

    return data, X_app, X, y, feature_cols

# Separa os dados entre treinamento e teste utilizando validação cruzada
def create_cross_validation(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test

# Função para adicionar valores em casos de valores faltantes
def imputer(X_train, X_test, X_app):
    imp = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    # Aplica o fit transform no X_train
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)
    data_app = imp.transform(X_app)

    return X_train, X_test, X_app

# Função para normalizar os dados de acordo com seu tipo max abs
def normalization(X_train, X_test, X_app, type):

    if type == "min_max":
        normalizer = MinMaxScaler(feature_range=(0, 1))

    elif type == "normalizer":
        normalizer = Normalizer().fit(X_train)

    elif type == "max_abs":
        normalizer = MaxAbsScaler()

    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    data_app = normalizer.transform(X_app)

    return X_train, X_test, X_app

# Criando o modelo preditivo knn para base de treino
def knn_train(X_train,y_train, X_test, y_test):

    print(' - Criando modelo preditivo')
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Faz o treinamento com as bases de treinamento
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(" * Train accuracy: " + str(accuracy))

    return knn_model

# Função para enviar as predições feitas ao servidor
def send_predictions(X_train, y_train, X_test, y_test, X_app):
    print(' - Aplicando modelo e enviando para o servidor')

    model =  knn_train(X_train, y_train, X_test, y_test)
    app_predictions = model.predict(X_app)

    URL = "http://aydanomachado.com/mlclass/01_Preprocessing.php"

    DEV_KEY = "AV"

    # json para ser enviado para o servidor
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(app_predictions).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    print(" - Resposta do servidor:\n", r.text, "\n")

data, X_app, X, y, feature_cols = read_data()

X_train, X_test, y_train, y_test = create_cross_validation(X, y, 0.05)
X_train, X_test, X_app = imputer(X_train, X_test, X_app)
X_train, X_test, X_app = normalization(X_train, X_test, X_app, type="max_abs")

send_predictions(X_train, y_train, X_test, y_test, X_app)

