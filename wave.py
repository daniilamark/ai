import mglearn as mglearn
import matplotlib.pyplot as plt

# регрессия
# Генерация данных на основе датасета Wave,

# Генерация данных на основе датасета Wave, регрессия
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")


mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=2)
mglearn.plots.plot_knn_regression(n_neighbors=3)

plt.show()