from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:193] + "\n...")
#print("Названия ответов: {}".format(iris_dataset['target_names']))
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
#numpy -- data massive
#target - класс цветка
#print("Тип массива data: {}".format(type(iris_dataset['data'])))
#print("Форма массива data: {}".format(iris_dataset['data'].shape))
#print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
#print("Первые 10 строк массива data:\n{}".format(iris_dataset['target'][:10]))
#print("Первые 10 строк массива data:\n{}".format(iris_dataset['target']))

# разбивает на 2 части \ перемешивает датасет


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#print("форма массива X_train: {}".format(X_train.shape))
#print("форма массива y_train: {}".format(y_train.shape))
#print("форма массива X_test: {}".format(X_test.shape))
#print("форма массива y_test: {}".format(y_test.shape))

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

