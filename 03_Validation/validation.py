import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import itertools
import matplotlib.pyplot as plt
import requests

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# fixa o random seed por motivo de reproducibilidade
seed = 7
np.random.seed(seed)

# Lê o arquivo com o dataset
def read_data():
    print('\n - Reading Abalone dataset')
    data = pd.read_csv('abalone_dataset.csv')
    data_app = pd.read_csv('abalone_app.csv')

    feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                    'shell_weight']
    class_names = ['class 01', 'class 02', 'class 03']

    X = data[feature_cols]
    data_app = data_app[feature_cols]
    y = data.type

    return X, y, data_app, class_names, feature_cols

# Altera os dados categóricos para numéricos a fim de aplicar os modelos
def categorical_to_numerical_data(data, data_app):
    sex_col_numerical = {"sex": {"M": 0, "F": 1, "I": 2}}
    data.replace(sex_col_numerical, inplace=True)
    data_app.replace(sex_col_numerical, inplace=True)

    return data, data_app

# Separa os dados entre treinamento e teste utilizando validação cruzada
def create_cross_validation(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test

# Função para normalizar os dados de acordo com seu tipo
def normalization(X_train, X_test, type):

    if type == "min_max":
        normalizer = MinMaxScaler(feature_range=(0, 1))

    elif type == "normalizer":
        normalizer = Normalizer().fit(X_train)

    elif type == "max_abs":
        normalizer = MaxAbsScaler()

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)

    return X_train, X_test

# Função para rankear a importância das para a classificação do problema
def importanceRanking(X, X_train, y_train):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("\n\n - Criando o rank das features com base na importância para classificação\n")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Função para criar modelo de matriz de confusão
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Função para plotar a matriz de confusão de acordo com a função anterior
def plot(confusion_matrix):
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                             title='Confusion matrix')
    plt.show()

# Modelo base para utilização do modelo de redes neurais
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=8, activation="tanh"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Função para criar os modelos e as predições de acordo com seus tipos
def create_model_and_predict_test(X_train, y_train, X_test, y_test, type):
    if type == "knn":
        print(' - Criando modelo preditivo KNN')
        model = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)

    elif type == "dtree":
        print(' - Criando modelo preditivo Decision Tree')
        model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

    elif type == "gnb":
        print(' - Criando modelo preditivo GaussianNB Naive Bayes')
        model = GaussianNB().fit(X_train, y_train)

    elif type == "svm":
        print('- Criando modelo preditivo SVM')
        model = SVC(C=3.4, kernel='linear', tol=0.01).fit(X_train, y_train)

    elif type == "neuralnet":
        print(' - Criando modelo preditivo de Redes Neurais ')

        # baseline_model = baseline_model()
        model = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    return model, predictions, cm

# Função para computar métricas de avaliação
def compute_and_print_metrics(model, predictions, cm):
    accuracy = model.accuracy_score(X_test, y_test)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    fscore = f1_score(y_test, predictions, average='macro')

    print("Accuracy: " + str(accuracy) + " Precision: " + str(precision)
          + " Recall: " + str(recall) + " F1 Score: " + str(fscore))
    print(cm)
    plot(cm)

# Função para enviar as predições feitas ao servidor
def send_predictions(X_train, y_train, X_test, y_test, data_app):
    # mudar o tipo para testar com cada modelo
    model, predictions, cm =  create_model_and_predict_test(X_train, y_train, X_test, y_test, type= "neuralnet")

    print(' - Aplicando modelo e enviando para o servidor')
    app_predictions = model.predict(data_app)
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"

    DEV_KEY = "AV"

    # json para ser enviado para o servidor
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(app_predictions).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    print(" - Resposta do servidor:\n", r.text, "\n")


X, y, data_app, class_names, feature_cols = read_data()
X, data_app = categorical_to_numerical_data(X, data_app)
X_train, X_test, y_train, y_test = create_cross_validation(X, y, 0.5)
send_predictions(X_train, y_train, X_test, y_test, data_app)
