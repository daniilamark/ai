# импорт библиотеки pandas
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

result = [4026110106, 200000,	200000,	1950,	1,	100,	37,	99.2,	103.2,	45.4,
          7.8,	0.2,	3243162.2,	37728.7,	5234449.5,	1534129.2,	1556499.9,	636.3,	172.3,	464.1]
#print(len(result))

params = {'n_estimators': 200,
          'max_depth': 8,
          'criterion': 'squared_error',
          'learning_rate': 0.03,
          'min_samples_leaf': 20,
          'min_samples_split': 20
          }
# params_c = {'criterion': 'friedman_mse',
#             'init': None,
#             'learning_rate': 0.1,
#             'loss': 'deviance',
#             'max_depth': 3,
#             'max_features': None,
#             'max_leaf_nodes': None,
#             'min_impurity_decrease': 0.0,
#             'min_impurity_split': None,
#             'min_samples_leaf': 1,
#             'min_samples_split': 2,
#             'min_weight_fraction_leaf': 0.0,
#             'n_estimators': 100,
#             'presort': 'auto',
#             'random_state': None,
#             'subsample': 1.0,
#             'verbose': 0,
#             'warm_start': False
# }

# {'n_estimators': 300,
#           'max_depth': 15,
#           'criterion': 'squared_error',
#           'learning_rate': 0.05,
#           'min_samples_leaf': 10,
#           'min_samples_split': 20
#           }
file = 'nyc_benchmarking.xlsx'

# Загружаем spreadsheet в объект pandas
xl = pd.ExcelFile(file)

# Загрузить лист в DataFrame по его имени: df1
df1 = xl.parse('Information and Metrics')

#knn = KNeighborsClassifier(n_neighbors=1)

# data_describe = df1.describe(include=[object])

X = df1.drop('ENERGY STAR Score', axis=1)
y = df1['ENERGY STAR Score']
GBC = GradientBoostingClassifier()
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)

### Тренируем
#gbr = GradientBoostingRegressor(**params)
#gbr = GradientBoostingClassifier(**params_c)

df1 = df1.reset_index()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(X_train.shape, X_test.shape)
#knn.fit(X_train, y_train)

accuracy_scores = cross_val_score(GBC, X_train, y_train, cv=10, scoring='accuracy')
GBC.fit(X_train, y_train)

print('accuracy_scores')
print(accuracy_scores)
print('--------')
print(np.mean(accuracy_scores))

X_new = np.array([result])

prediction = GBC.predict(X_new)

print("Свойства: {}".format(result))
print("Прогноз: {}".format(prediction))

y_pred = GBC.predict(X_test)

print("Прогнозы для тестового набора:\n {}".format(y_pred))
#print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(GBC.score(X_test, y_test)))
