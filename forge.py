import mglearn as mglearn
import pandas as pd
import matplotlib.pyplot as plt

# Генерация данных на основе датасета Forge
# генерируем набор данных
X, y = mglearn.datasets.make_forge()
# строим график для набора данных
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
plt.show()
print("форма массива X: {}".format(X.shape))

