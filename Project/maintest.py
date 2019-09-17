import pandas as pd

df = pd.read_csv('./Project/data/train_intent.csv')
print(df.shape) # (3918, 2)
print(df.isnull().sum())    # 결측값 없음



