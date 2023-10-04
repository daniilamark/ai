# импорт библиотеки pandas
import pandas as pd

# Загружаем ваш файл в переменную `file` / вместо 'example' укажите название свого файла из текущей директории
file = 'nyc_benchmarking.xlsx'

# Загружаем spreadsheet в объект pandas
xl = pd.ExcelFile(file)

# Печатаем название листов в данном файле
print(xl.sheet_names)

# Загрузить лист в DataFrame по его имени: df1
df1 = xl.parse('Information and Metrics')
print(df1)