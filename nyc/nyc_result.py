# импорт библиотеки pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

result = [4026110106, 200000,	200000,	1950,	1,	100,	37,	99.2,	103.2,	45.4,
          7.8,	0.2,	3243162.2,	37728.7,	5234449.5,	1534129.2,	1556499.9,	636.3,	172.3,	464.1]
print(len(result))

file = 'nyc_benchmarking.xlsx'

# Загружаем spreadsheet в объект pandas
xl = pd.ExcelFile(file)

# Печатаем название листов в данном файле
# print(xl.sheet_names)

# Загрузить лист в DataFrame по его имени: df1
df1 = xl.parse('Information and Metrics')
print(df1)
# df1.replace([np.inf, -np.inf], np.nan, inplace=True)

knn = KNeighborsClassifier(n_neighbors=1)

# data_describe = df1.describe(include=[object])

X = df1.drop('ENERGY STAR Score', axis=1)
y = df1['ENERGY STAR Score']

df1 = df1.reset_index()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
print(X_train.shape, X_test.shape)
knn.fit(X_train, y_train)

X_new = np.array([result])

prediction = knn.predict(X_new)

print("Свойства: {}".format(result))
print("Прогноз: {}".format(prediction))

y_pred = knn.predict(X_test)

print("Прогнозы для тестового набора:\n {}".format(y_pred))
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))
