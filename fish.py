'''
Fish market
Database of common fish species for fish market (файл прикреплен снизу)

1.Загрузить этот датасет в программу через Pandas
2.Построить на основании датасета модель классификации с использованием метода KNeighboursCalassifier
3.Определить качество модели посредством методов meanили score (попытаться повысить качество модели)
4.Научиться генерить новую рыбу (рандомным образом создавать объект со случайным набором параметров)
5.Определять тип рыбы по новым параметром.
import pandas as pd
df = pd.read_csv('data.csv') (изменено)
'''

import numpy as np
import pandas as pd


df = pd.read_csv('Fish.csv', delimiter=',')

#pd.read_csv('pandas_tutorial_read.csv', delimiter=';',
#	    names=['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])

print(df)

