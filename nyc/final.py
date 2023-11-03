import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

result = [4026110106, 200000,	200000,	1950,	1,	100,	37,	99.2,	103.2,	45.4,
          7.8,	0.2,	3243162.2,	37728.7,	5234449.5,	1534129.2,	1556499.9,	636.3,	172.3,	464.1]
print(len(result))

params = {'n_estimators': 300,
           'max_depth': 16,
           'criterion': 'squared_error',
           'learning_rate': 0.03,
           'min_samples_leaf': 15,
           'min_samples_split': 20
          }

file = 'nyc_benchmarking.xlsx'

# Загружаем spreadsheet в объект pandas
xl = pd.ExcelFile(file)

# Загрузить лист в DataFrame по его имени: df1
df1 = xl.parse('Information and Metrics')

knn = KNeighborsClassifier(n_neighbors=1)

X = df1.drop('ENERGY STAR Score', axis=1)
y = df1['ENERGY STAR Score']

### Тренируем
gbr = GradientBoostingRegressor(**params)

df1 = df1.reset_index()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(X_train.shape, X_test.shape)
#knn.fit(X_train, y_train)
gbr.fit(X_train, y_train)

X_new = np.array([result])

prediction = gbr.predict(X_new)

print("Свойства: {}".format(result))
print("Прогноз: {}".format(prediction))

y_pred = gbr.predict(X_test)

print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(gbr.score(X_test, y_test)))