import pandas as pd
import numpy as np

from konlpy.tag import Okt
from gensim.models import FastText

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

def main():
    print('──────────────────────main Start──────────────────────')
    # entitiy 훈련 데이터 불러오기
    df = pd.read_csv('./GBJ/data/train_entity.csv', encoding='UTF-8', sep=' ')
    print(df.shape) # (1362, 2)
    print(df.isnull().sum()) # word    0 tag      0
    
    words, tags = df['word'], df['tag'] # x, y 데이터

    # x,y 중복된 데이터 제거.
    words = list(set(words))
    tags = list(set(tags))
    print(len(words)) # 474
    print(tags) # ['WORD', 'O']
    #──────────────────────────────────────────────────────────────────────────────────────────
    # words 데이터 word2vec(FastText)
    vector_size = 15
    words = np.array(words)
    words = words.reshape(1, len(words)) # fasttext가 단어별로 적용되로독 차원 크기 변경. (1, n)

    word2vec_model = FastText(size = vector_size, window=3, min_count= 1)
    print('Fasttext build compile')
    word2vec_model.build_vocab(sentences=words)
    print('Fasttext trian')
    word2vec_model.train(sentences = words, total_examples= word2vec_model.corpus_count, epochs= 10)

    print('Fasttext complete')
    w2c_index = word2vec_model.wv.index2word # fasttext가 적용된 단어 목록들
    print(w2c_index, len(w2c_index)) # 474

    tags_mapping = {'O': 0, 'WORD': 1} # tags 맵핑 데이터 생성
    print(tags_mapping) # {'O': 0, 'WORD': 1}
    #──────────────────────────────────────────────────────────────────────────────────────────
    # x, y train 생성
    words_data, y_train = df['word'], df['tag']
    y_train = y_train.map(tags_mapping)

    x_train = []
    for x_raw in words_data:
        x_raw = word2vec_model[x_raw]
        x_train.append(x_raw)
    x_train = np.array(x_train)
    print(x_train.shape) # (1362, 15)
    #──────────────────────────────────────────────────────────────────────────────────────────
    # keras modeling
    model = Sequential()
    model.add(Dense(256, input_dim = 15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=48, epochs=100)
    #──────────────────────────────────────────────────────────────────────────────────────────
    # 입력
    okt = Okt()
    tags_mapping = {0: 'O', 1: 'WORD'} # tags 맵핑 데이터 생성

    while True:
        print('User : ', end='')
        text = okt.morphs(input())

        result = []
        for word in text:
            vec = word2vec_model[word]
            vec = vec.reshape(1, len(vec))
            r = model.predict(vec)
            r = np.round(r)
            result.append(r[0][0])
        result = pd.Series(result)
        result = result.map(tags_mapping)
        result = np.array(result)

        print('>>')
        for i in range(len(text)):
            print(text[i], '{', result[i],'}', end=' ')
        print('')

main()