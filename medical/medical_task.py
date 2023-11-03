# Импортируем Pandas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Считываем содержимое файла в переменную data:
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("insurance.csv")

# Проверяем, что данные прочитаны:
print(data.head())

knn = KNeighborsClassifier(n_neighbors=1)

# разбивает на 2 части \ перемешиваем датасет
X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

print(X_train.shape, X_test.shape)

knn.fit(X_train, y_train)


# logreg001 = LogisticRegression().fit(X_train, y_train)
# print("Правильность на обучающем наборе: {:.3f}".format(logreg001.score(X_train, y_train)))
# print("Правильность на тестовом наборе: {:.3f}".format(logreg001.score(X_test, y_test)))
# logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# print("Правильность на обучающем наборе: {:.3f}".format(logreg100.score(X_train, y_train)))
# print("Правильность на тестовом наборе: {:.3f}".format(logreg100.score(X_test, y_test)))
