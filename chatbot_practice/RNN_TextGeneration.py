from keras_preprocessing.text import Tokenizer

text="""경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

# 토큰화 및 정수 인코딩
t = Tokenizer()
t.fit_on_texts([text])
encoded = t.texts_to_sequences([text])[0]

vocab_size = len(t.word_index) + 1
# print('단어 집합의 크기 : %d' % vocab_size) # 단어 집합의 크기 : 12
# print(t.word_index)
# {'말이': 1, '경마장에': 2, '있는': 3, '뛰고': 4, '있다': 5, '그의': 6,
#  '법이다': 7, '가는': 8, '고와야': 9, '오는': 10, '곱다': 11}

sequences = list()
for line in text.split('\n'): # Wn을 기준으로 문장 토큰화
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

# print('학습에 사용할 샘플의 개수: %d' % len(sequences)) # 11
# print(sequences)

# print(max(len(l) for l in sequences))

from keras.preprocessing.sequence import pad_sequences
max_len=6
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
# print(sequences)

import numpy as np
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
# print(X)
# print(y)

from keras.utils import to_categorical
y = to_categorical(y, num_classes=vocab_size)
# print(y)

from keras.layers import Embedding, Dense, SimpleRNN
from keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=5, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, '경마장에', 4))
# '경마장에' 라는 단어 뒤에는 총 4개의 단어가 있으므로 4번 예측

print(sentence_generation(model, t, '그의', 2)) # 2번 예측

print(sentence_generation(model, t, '가는', 5)) # 5번 예측