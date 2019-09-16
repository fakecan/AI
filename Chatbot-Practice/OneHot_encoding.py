from konlpy.tag import Okt

okt = Okt()
token = okt.morphs("나는 자연어 처리를 배운다")
# print(token)    # ['나', '는', '자연어', '처리', '를', '배운다']

word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
        # print(word2index, len(word2index))
# print(word2index)   # {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}

# def onehot_encoding(word, word2index):
#     onehot_vector = [0]*(len(word2index))
#     index = word2index[word]
#     onehot_vector[index] = 1
#     return onehot_vector

# print(onehot_encoding("자연어", word2index))    # [0, 0, 1, 0, 0, 0]