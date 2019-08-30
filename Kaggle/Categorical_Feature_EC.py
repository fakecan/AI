import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/cat-in-the-dat/train.csv')
df_test = pd.read_csv('./data/cat-in-the-dat/test.csv')

# print(df_train.info())
# for col in df_train.columns:
#     print('{}\n'.format(df_train[col].head()))

# print(df_train.shape)   # (300000, 25)
# print(df_train.head(), df_test.head())
# id,bin_0,bin_1,bin_2,ord_0,day,month,target -> int64
# bin_3,bin_4,nom_0,nom_1,nom_2,nom_3,nom_4,nom_5,nom_6,nom_7,nom_8,nom_9,  -> object
# ord_1,ord_2,ord_3,ord_4,ord_5  -> object

# print(df_test.shape)    # (200000, 24)
# print(df_test.head(), df_test.head())

x_data = df_train.loc[:, ['id','bin_0','bin_1','bin_2','bin_3','bin_4','nom_0',
                            'nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7',
                            'nom_8','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4',
                            'ord_5','day','month']]
y_data = df_train.loc[:, 'target']
# print(x_train.shape, y_train.shape)   # (300000, 24) (300000,)
# print(x_train)
# print(y_train)

# x_test = df_test.loc[:, :]
# print(x_test.shape) # (200000, 24)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, train_size=0.8, random_state=66
)
print(x_train.shape, x_test.shape)      # (240000, 24) (60000, 24)   
# print(y_train.shape, y_test.shape)    # (240000,) (60000,)
# print(y_train)
# print(y_test)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(x_train[:, ])

# le = LabelEncoder()
# x_train = le.fit(x_train)
# x_train = le.transform(x_train)

# onehotencoder = OneHotEncoder()