import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

dataset = pd.read_csv("insurance.csv")
gender_list = {'male': 1, 'female': 2}
smoker_list = {'yes': 1, 'no': 2}
region_list = {'southeast': 1, 'northwest': 2, 'southwest': 3, 'northeast': 4}


dataset.sex = [gender_list[item] for item in dataset.sex]
dataset.smoker = [smoker_list[item] for item in dataset.smoker]
dataset.region = [region_list[item] for item in dataset.region]

print(dataset.shape)
# Проверяем, что данные прочитаны:
print(dataset.head())
print(dataset.describe())

X = dataset[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
#X = dataset[['age', 'bmi', 'children']]
y = dataset['charges']

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Построение модели OLS

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Правильность на обучающем наборе: {:.2f}".format(lr.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(lr.score(X_test, y_test)))
print("-------------------------------------------------")


coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)


y_pred = lr.predict(X_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("----------------------------")
print(df)

print("----------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Построение Ridge
print("------------Ri----------------")
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(ridge.score(X_test, y_test)))


