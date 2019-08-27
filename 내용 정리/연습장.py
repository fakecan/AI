import numpy as np

x = np.array(range(10))
print(x)

arraySet = np.array(range(1, 101))
print(arraySet)
size = 7
def split_n(seq, size):
    newList = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        newList.append([item for item in subset])
    # print(type(newList))    # list
    return np.array(newList)

dataset = split_n(arraySet, size)
print(dataset)
x_data = dataset[:, 0:6] #(6)
y_data = dataset[:, -1] #(6, )
print(x_data)
print(y_data)
cc