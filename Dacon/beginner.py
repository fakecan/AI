# import pandas as pd
# import numpy as np

# # import pandas_profiling
# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import matplotlib.pyplot as plt # 데이터 시각화
import itertools
from datetime import datetime, timedelta # 시간 데이터 처리
from statsmodels.tsa.arima_model import ARIMA

train = pd.read_csv("./Dacon/train.csv")
test = pd.read_csv("./Dacon/test.csv")
submission = pd.read_csv("./Dacon/submission_1002.csv")

# print("train shape : ", train.shape)            # (16909, 1301)
# print("test shape : ", test.shape)              # (8760, 201)
# print("submission shape : ", submission.shape)  # (200, 40)

test["Time"] = pd.to_datetime(test["Time"])
test = test.set_index("Time")
print(test.head())

place_id, time, target = [], [], []
for i in test.columns:
    for j in range(len(test)):
        place_id.append(i) # place_id에 미터 ID를 정리합니다.
        time.append(test.index[j]) # time에 시간대를 정리합니다.
        target.append(test[i].iloc[j]) # target에 전력량을 정리합니다.

new_df=pd.DataFrame({'place_id':place_id,'time':time,'target':target})
new_df=new_df.dropna() # 결측치를 제거합니다.
new_df=new_df.set_index('time') # time을 인덱스로 저장합니다.
print(new_df.head())