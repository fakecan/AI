'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/test0822.csv', encoding='UTF-8',sep=',')

df.columns.drop('date')
print(df)   # (5479, 9)
# df = df.dropna(how='any')
# print(df.shape)   # (5474, 9)
'''


    
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
test = pd.read_csv("./data/test0822.csv")
test_columns = test.columns.drop("date")
print(test_columns)
print(test)