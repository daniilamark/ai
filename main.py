import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import sklearn

print("версия Python: {}".format(sys.version))

# парсить данные
print("версия pandas: {}".format(pd.__version__))

# для визуализации данных
print("версия matplotlib: {}".format(matplotlib.__version__))

# для машинных вычислений
print("версия NumPy: {}".format(np.__version__))
print("версия SciPy: {}".format(sp.__version__))

# машинное обучение, описание и расчеты
print("версия scikit-learn: {}".format(sklearn.__version__))
