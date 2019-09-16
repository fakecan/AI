import pandas as pd
import numpy as np

from gensim.models import FastText
from konlpy.tag import Okt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

def main():
    print('──────────────────────main Start──────────────────────')
    # intent 훈련 데이터 불러오기
    df = pd.read_csv('./GBJ/data/train_intent.csv')
    print(df.shape) # (3918, 2)
    print(df.isnull().sum()) # question    0 intent      0

    #──────────────────────────────────────────────────────────────────────────────────────────
    # 형태소 추출및 word2vec(FastText)
    vector_size = 15
    okt = Okt()
    word2vec_model = FastText(size = vector_size, window=3, min_count= 1)

    question = df['question'] # 질문 문장 데이터
    joinStr = ' '.join(question) # list > str로 형 변환
    morphs = okt.morphs(joinStr) # 형태소 추출 > list
    morphs = np.array(list(set(morphs))) # 중복된 단어 제거
    morphs = morphs.reshape(1, len(morphs)) # fasttext가 단어별로 적용되로독 차원 크기 변경. (1, n)

    print('Fasttext build compile')
    word2vec_model.build_vocab(sentences=morphs)
    print('Fasttext trian')
    word2vec_model.train(sentences = morphs, total_examples= word2vec_model.corpus_count, epochs= 10)

    print('Fasttext complete')
    w2c_index = word2vec_model.wv.index2word # fasttext가 적용된 단어 목록들

    #──────────────────────────────────────────────────────────────────────────────────────────
    # intent 값 분류
    intent = df['intent'] # 의도 값
    intent = list(set(intent)) # 중복된 데이터 제거
    print(intent)
    # ['위키', '달력', '날씨', '시간', '인물', '맛집', '번역', '명언', '이슈', '먼지', '음악', '뉴스']

    # intent_mapping 생성
    idx = 0
    intent_mapping = {}
    for i in intent:
        intent_mapping[i] = idx
        idx += 1

    print(intent_mapping)
    # {'뉴스': 0, '달력': 1, '음악': 2, '먼지': 3, '명언': 4, '이슈': 5, '날씨': 6, '번역': 7, '맛집': 8, '시간': 9, '인물': 10, '위키': 11}

    # y_data 생성
    y_data = df['intent'] # 의도 값
    y_data = y_data.map(intent_mapping)
    y_data = to_categorical(y_data) # onehot encoding
    print(y_data.shape) # (3918, 12)
    #──────────────────────────────────────────────────────────────────────────────────────────
    # x_data 생성
    encode_length = 10
    x_data = []
    for q_raw in question:
        q_raw = okt.morphs(q_raw) # 문장 형태소별로 분리(단어 분리). str > list
        q_raw = list(map(lambda x: q_raw[x] if x < len(q_raw) else '@', range(encode_length)))
        # x가 단어의 수보다 작을 경우 단어(q_raw[x]) 그대로 리스트에 삽입하고 아닐 경우 @를 삽입한다.

        q_raw = list(map(lambda x: word2vec_model[x] if x in w2c_index else np.zeros(vector_size, dtype=float), q_raw))
        q_raw = np.array(q_raw)
        x_data.append(q_raw)
    x_data = np.array(x_data)
    print(x_data.shape) # (3918, 10, 15)
    #──────────────────────────────────────────────────────────────────────────────────────────
    # keras modeling
    x_data = x_data.reshape(len(x_data), encode_length * vector_size)
    print('Keras Start',x_data.shape, y_data.shape)

    model = Sequential()
    model.add(Dense(256, input_dim = 150, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(x_data, y_data, batch_size=128, epochs=100)
    #──────────────────────────────────────────────────────────────────────────────────────────

    # 입력 데이터 중 불용어 제거
    del_josa = [
    '이구나', '이네', '이야',
    '은', '는', '이', '가', '을', '를',
    '로', '으로', '이야', '야', '냐', '니']

    def tokenize(sentence):
        word_bag = []
        pos = okt.pos(sentence) # 형태소에 품사를 추가한다.

        for word, tag in pos: # 단어와 품사
            if (tag == 'Josa' and word in del_josa) or tag == 'Punctuation':
                # 불 필요한 조사와 구두점을 제거
                continue
            else:
                word_bag.append(word) # 단어를 리스트에 추가한다.
        result = ' '.join(word_bag)

        return result

    #  입력 데이터(문장)를 벡터화 한다. (데이터 전처리)
    def pred(text):
        q_raw = okt.morphs(text)
        q_raw = list(map(lambda x: q_raw[x] if x < len(q_raw) else '@', range(encode_length)))
        q_raw = list(map(lambda x: word2vec_model[x] if x in w2c_index else np.zeros(vector_size, dtype=float), q_raw))
        q_raw = np.array(q_raw)
        print(q_raw)
        q_raw = q_raw.reshape(1,150)
        return q_raw

    # 작동.
    while True:
        print('User : ', end='')
        speech = tokenize(input())
        print('tokenize : ',speech)
        speech = pred(speech)

        # 결과
        y_intent = model.predict(speech)
        y_intent = np.argmax(y_intent)

        for result, num in intent_mapping.items():
            if y_intent == num:
                print('Intent : ',result, y_intent)
                break
        
    print('──────────────────────main END──────────────────────')


print('──────────────────────Start──────────────────────')

main()
print('asqwdqdd')