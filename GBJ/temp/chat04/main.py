import numpy as np
import pandas as pd
import os

from gensim.models import FastText
from konlpy.tag import Okt

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from keras.utils.np_utils import to_categorical

path_name = 'GBJ\\temp\chat04\\'
okt = Okt()
vector_size = 15
#──────────────────────────────────────────────────────────────────────────────────────────
def mapping_dict_making(arr): # 고유한 요소별로 매핑되는 int 값을 생성한다.
    index = np.arange(len(arr))
    mapping_dict = zip(arr, index) # zip [1,2,3], [4,5,6] >> [(1,4), (2,5), (3,6)]
    return dict(mapping_dict)
#──────────────────────────────────────────────────────────────────────────────────────────
def w2vTrain_model(arr, save_name, vector_size = 15, epochs_count = 10, train = False): # str로된 list를 실수로된 벡터로 변환 한다.
    save_local, file_name = path_name + save_name, '\\w2v'

    if train: # 모델을 훈련 시킨다.
        # w2v 모델링
        print('▶ Fast Text Training :', save_name)
        model = FastText(size = vector_size, window=3, min_count= 1)
        model.build_vocab(sentences=arr)
        model.train(sentences = arr, total_examples= model.corpus_count, epochs= epochs_count)
        wv_index = model.wv.index2word

        # 모델링 저장
        print('▶ Save Fast Text Train :', save_name)
        if not os.path.exists(save_local): os.makedirs(save_local) # 풀더가 없을 경우 풀더를 만든다.
        save_local = save_local + file_name
        model.save(save_local)
        return model

    else: # 훈련 시킨 모델을 불러 온다.
        print('▶ Load Fast Text Train :', save_name)
        return FastText.load(save_local + file_name)
#──────────────────────────────────────────────────────────────────────────────────────────
del_josa = ['이구나', '이네', '이야','은', '는', '이', '가', '을', '를','로', '으로', '이야', '야', '냐', '니']
def stopword_elimination(sentence): # 불용어를 제거하고 형태소별로 분리 시킨다.
    pos = okt.pos(sentence) # 문장의 각 형태소마다 품사를 붙힌다.
    word_bag = []
    for word, tag in pos: # (단어, 품사)
        if tag == 'Josa' and word in del_josa: continue # 불필요한 조사를 패스
        elif tag == 'Punctuation': continue # 구두점 패스
        else: word_bag.append(word)
    return ' '.join(word_bag) # list를 str로 형변환
#──────────────────────────────────────────────────────────────────────────────────────────
def pred_intent_data(text_raw, intent_size, w2c_index, vector_size): # intent 데이터 딥러닝 모델에 적용할 수 있도록 전처리한다.
    text_raw = okt.morphs(text_raw) # 형태소별로 분리
    test_raw = list(
                map(lambda word_index: text_raw[word_index] if word_index < len(text_raw) else '$', range(intent_size))
                )
    # 입력 데이터의 차원의 수를 조정
    test_raw = list(
                map(lambda word: w2c_index[word] if word in w2c_index else np.zeros(vector_size, dtype=float), test_raw)
                )
    # str로된 데이터를 전부 수치화 시킴
    return np.array(test_raw)
#──────────────────────────────────────────────────────────────────────────────────────────