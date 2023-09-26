'''
Fish market
1.Загрузить этот датасет в программу через Pandas
2.Построить на основании датасета модель классификации с использованием метода KNeighboursCalassifier
3.Определить качество модели посредством методов meanили score (попытаться повысить качество модели)
4.Научиться генерить новую рыбу (рандомным образом создавать объект со случайным набором параметров)
5.Определять тип рыбы по новым параметром.
'''
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def getFish():
    result = []

    weight = random.randint(0, 1700)
    length1 = random.uniform(6.0, 59.0)
    length2 = random.uniform(8.4, 63.4)
    length3 = random.uniform(8.8, 68.0)
    height = random.uniform(1.5, 20.0)
    width = random.uniform(1.0, 9.0)

    result.append(weight)
    result.append(length1)
    result.append(length2)
    result.append(length3)
    result.append(height)
    result.append(width)

    return result

# загрузка датасета
dataset = pd.read_csv('Fish.csv', delimiter=',')
knn = KNeighborsClassifier(n_neighbors=1)

# вспомогательный вывод
#print(dataset.shape)
#print(dataset.head())
#print(dataset.describe())
categorical_columns = [c for c in dataset.columns if dataset[c].dtype.name == 'object']
numerical_columns   = [c for c in dataset.columns if dataset[c].dtype.name != 'object']
#print(categorical_columns)
#print(numerical_columns)

data_describe = dataset.describe(include=[object])

# общая информация по категориальным признакам:
#print(dataset.describe(include=[object]))


# разбивает на 2 части \ перемешиваем датасет
X = dataset.drop('Species', axis=1)
y = dataset['Species']

#print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

print(X_train.shape, X_test.shape)

knn.fit(X_train, y_train)

fish = getFish()
X_new = np.array([fish])

prediction = knn.predict(X_new)
print("Свойства рандомной рыбы: {}".format(fish))
print("Прогноз: {}".format(prediction))

y_pred = knn.predict(X_test)
#print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

