import mglearn as mglearn

# Использование сокращенного Бостона
from sklearn.datasets import load_boston

boston = load_boston()
print("форма массива data для набора boston: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("форма массива X: {}".format(X.shape))


