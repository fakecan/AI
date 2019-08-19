import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten, Conv1D
from sklearn.model_selection import train_test_split

# kospi = pd.read_csv("kospi.csv")
kospi = pd.read_csv("D:/GitRepository/ML/ml/kospi.csv", encoding="UTF-8")
kospi

kospi["price_band"] = kospi["day_high"] - kospi["day_low"]
kospi.head()

# Var_Corr = kospi[["volume","price_band"]].corr()
# sns.heatmap(Var_Corr, annot=True)

# plt.figure(figsize=(15,15))
# sns.heatmap(data = kospi.corr(), annot=True, 
# fmt = '.4f', linewidths=.5, cmap='Blues')





# kospi=kospi.sort_index(ascending=False)



# stand = StandardScaler()
volume_minmax = MinMaxScaler(feature_range=(0, 3))
won_exchange_minmax = MinMaxScaler(feature_range=(0,5))
# stand.fit(x_train)
# x_train = stand.transform(x_train)
# x_test = stand.transform(x_test)
volume_minmax_list = np.array(kospi["volume"])
volume_minmax_list= volume_minmax_list.reshape((-1,1))
volume_minmax.fit(volume_minmax_list)
volume_minmax_list = volume_minmax.transform(volume_minmax_list)



won_exchange_minmax_list = np.array(kospi["won_exchange"])
won_exchange_minmax_list= won_exchange_minmax_list.reshape((-1,1))
won_exchange_minmax.fit(won_exchange_minmax_list)
won_exchange_minmax_list = won_exchange_minmax.transform(won_exchange_minmax_list)

kospi["volume"]=volume_minmax_list
kospi["won_exchange"]=won_exchange_minmax_list



kospi_label = kospi[["day", "day_end"]]

stand = MinMaxScaler(feature_range=(0, 8))

open_price_list = np.array(kospi["open_price"])
day_high_list = np.array(kospi["day_high"])
day_low_list = np.array(kospi["day_low"])
day_end_list = np.array(kospi["day_end"])

open_price_list = open_price_list.reshape((-1,1))
day_high_list = day_high_list.reshape((-1,1))
day_low_list = day_low_list.reshape((-1,1))
day_end_list = day_end_list.reshape((-1,1))

stand.fit(open_price_list)
open_price_list = stand.transform(open_price_list)
day_high_list = stand.transform(day_high_list)
day_low_list = stand.transform(day_low_list)
day_end_list = stand.transform(day_end_list)

kospi["open_price"],kospi["day_high"],kospi["day_low"], kospi["day_end"] = open_price_list, day_high_list, day_low_list, day_end_list


kospi= kospi.drop(["day"], axis=1)
kospi_label = kospi_label.drop(["day"], axis=1)
label = np.array(kospi_label)

label


kospi.head()

band = MinMaxScaler(feature_range=(0, 5))

price_band_list = np.array(kospi["price_band"])
price_band_list = price_band_list.reshape((-1,1))

band.fit(price_band_list)
price_band_list = band.transform(price_band_list)

kospi["price_band"] = price_band_list
kospi = kospi.drop(["open_price","won_exchange"], axis=1)
kospi_list = np.array(kospi)

kospi_list.shape


###예측에 쓰이는 기간
size = 5
pre_day = 10

##TRAIN 데이터와 LABEL데이터를 나누는 곳
def split_7(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)

def split_label(seq, size,pre_day):
    lab = []
    lab = np.array(lab)
    lab = seq[size:-pre_day+1]
    print(lab.shape)
    
    for i in range(1,pre_day):
        
        # print("%d"%i,seq[size+i:-pre_day+(i+1)].shape)
        if (pre_day-1)!= i:
            lab = np.c_[lab[:], seq[size+i:-pre_day+(i+1)]]
        else:
            lab = np.c_[lab[:], seq[size+i:]]
        print(lab.shape)
    return lab



x_train=split_7(kospi_list, size)
print("==================")
# print(x_train[0])
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],1)
x_train = x_train.reshape(x_train.shape[0],-1,1)

# print(x_train[0])
# x_train.shape
x_map = x_train[-90:]
y_map = label[-90+size:]

x_pre = x_train[-pre_day:]
x_train = x_train[:-pre_day]
# print(x_pre)

label.shape
y_train = split_label(label, size,pre_day)
# print(y_train)
# print(label[:10])
# print(y_train[:10])
# print(x_train[0])
# print(x_train.shape)
# print(y_train.shape)
# print(x_pre.shape)
# x_pre = x_pre.reshape((-1,x_pre.shape[0],1))
# print(x_pre.shape)

# print(x_train.shape)
# print(y_train.shape)

print(kospi_list.shape)
print(x_train.shape)
print(y_train.shape)
x_tr, x_t, y_tr, y_t = train_test_split(x_train, y_train, random_state=66, test_size = 0.3)
# x_t, x_val, y_t, y_val = train_test_split(x_t, y_t, random_state=66, test_size = 0.5)
###x_train shape (593, 7, 6, 1)


##모델

model = Sequential()
# model.add(LSTM(36, input_shape=(x_tr.shape[1],1),return_sequences=True,activation="relu"))
model.add(Conv1D(kernel_size=3,filters=50,padding='same'))
model.add(Conv1D(kernel_size=3,filters=50,padding='same'))


model.add(Conv1D(kernel_size=4,filters=70,padding='same'))
model.add(Conv1D(kernel_size=4,filters=70,padding='same'))
model.add(Conv1D(kernel_size=4,filters=200))
model.add(Conv1D(kernel_size=4,filters=200))
# model.add(Conv1D(kernel_size=4,filters=70,padding='same'))






# model.add(BatchNormalization())

# model.add(Dropout(0.3))

# model.add(Dropout(0.3))
model.add(Dense(30))
model.add(Flatten())

model.add(Dense(10))
import keras

model.compile(loss = "mse",optimizer="adam", metrics=['mae'])
early = keras.callbacks.EarlyStopping(monitor='loss',mode="auto",patience=5)
model.fit(x_tr,y_tr,epochs=1000,batch_size=50, verbose=2)

model.summary()


#모델검증
loss, acc = model.evaluate(x_t, y_t)
y_ = model.predict(x_t)
for i in range(len(y_)):
    print("pre: ",y_[i], "\nori: ",y_t[i])
print(loss, acc)
y_pre = model.predict(x_pre)
print(y_pre)
for i in range(len(y_pre[-1])):
    print("8 월 %d회 :"%(i+1),y_pre[-1][i])
from sklearn.metrics import r2_score
print("r2:",r2_score(y_t,y_))

y_pre_map = model.predict(x_map)
# print(y_pre_map)
plt.plot(y_map)
plt.plot(y_pre_map[:,0])

plt.title("model kospi")
plt.ylabel("end_price")

plt.xlabel("day")
plt.legend(["real_end","pre_end"], loc="upper left")
plt.show()
plt.show()