'''
#!/usr/bin/env python
# coding: utf-8

# ## (1) 형태소분석

# <br>

# < 파이썬 한국어 자연어처리 패키지인 KoNLPy 설치 >

# In[1]:


# get_ipython().system('pip3 install konlpy')


# <br>

# < 꼬꼬마로 형태소분석 ><br><br>
# nouns -> 명사만 추출<br>
# morphs -> 형태소 추출<br>
# pos -> 형태소와 품사 추출<br>

# In[2]:


from konlpy.tag import Kkma
kkma = Kkma()

print(kkma.nouns(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))
print(kkma.morphs(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))
print(kkma.pos(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))


# <br>

# < 코모란으로 형태소분석 >

# In[3]:


from konlpy.tag import Komoran
komoran = Komoran()

print(komoran.nouns(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))
print(komoran.morphs(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))
print(komoran.pos(u'자연어처리는 컴퓨터가 인간의 언어를 처리하도록 하는 인공지능입니다.'))


#   <br>
#   <br>
#   <br>
#   <br>
#   <br>

# ## (2) RNN으로 캐릭터 글자 예측

# <br>

# < 원본 코드 ><br>
# -> https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/02%20-%20Autocomplete.py<br>
# <br><br><br>
# < 프로그램 설명 ><br>
# <br>
# 4개의 글자를 가진 단어를 학습<br>
# 3글자가 주어지면 마지막 글자를 예측<br>
# <br>
# -----------------<br>
# wor -> d<br>
# lov -> e<br>
# -----------------<br>
# <br>
# (에러 발생시 '런타임->모든 런타임 재설정' 후 처음부터 실행)
# <br><br><br>
'''
# In[4]:


import tensorflow as tf
import numpy as np



# 캐릭터 설정
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# One-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만듬
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k': 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 4자로 된 단어 집합
# wor -> X, d -> Y
# woo -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 
            'cool', 'load', 'love', 'kiss', 'kind']



#----------------------------------------
# 데이터 배치 생성
#----------------------------------------
def make_batch(seq_data):

    input_batch = []
    target_batch = []

    for seq in seq_data:
        # input_batch와 target_batch는 알파벳 배열의 인덱스 번호
        # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
        input = [num_dic[n] for n in seq[:-1]]
        
        # 3, 3, 15, 4, 3 ...
        target = num_dic[seq[-1]]
        
        # One-hot 인코딩
        # if input is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])
        
        # 비용함수의 label 값
        # softmax_cross_entropy_with_logits -> one-hot 인코딩
        # sparse_softmax_cross_entropy_with_logits -> index 숫자
        target_batch.append(target)

    return input_batch, target_batch


  
#----------------------------------------
# 첫번째 배치 출력
# input : w, o, r
# target : d
#----------------------------------------
input_batch, target_batch = make_batch(seq_data)

print(input_batch[0])
print('\n')
print(target_batch[0])


# <br>
# 
# 

# In[5]:


#----------------------------------------
# 옵션 설정
#----------------------------------------
learning_rate = 0.01
n_hidden = 128
total_epoch = 30

# 타입 스텝: [1 2 3] => 3
# RNN을 구성하는 시퀀스의 갯수
n_step = 3

# 입력값과 출력값의 크기
# 알파벳에 대한 one-hot 인코딩이므로 26개
# 예) c => [0 0 1 0 0 0 0 0 0 0 0 ... 0]
n_input = n_class = dic_len



#----------------------------------------
# 신경망 모델 구성
#----------------------------------------
X = tf.placeholder(tf.float32, [None, n_step, n_input]) # (3, 26)
Y = tf.placeholder(tf.int32, [None])    # (26,)
W = tf.Variable(tf.random_normal([n_hidden, n_class]))  # (128, 26) 
b = tf.Variable(tf.random_normal([n_class]))    # (26,)

# RNN 셀
cell1 = tf.nn.rnn_cell.LSTMCell(n_hidden)   # ~(output) 여기서는 128

# 과적합 방지를 위한 Dropout
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)

# 여러개의 셀을 조합해서 사용하기 위해 셀을 추가로 생성
cell2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
# cell2 = tf.nn.rnn_cell.BasicRNNCell(n_hidden)


# 여러개의 셀을 조합한 RNN 셀을 생성
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

# 순환 신경망을 생성
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# outputs 결과에서 마지막 타임 스텝만 구함
# outputs의 형태를 변경
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

# 128개 히든 레이어의 output를 26개 알파벳의 one-hot 인코딩 형식으로 변경
model = tf.matmul(outputs, W) + b

# 비용함수
cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=Y))

# 옵티마이저
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



#----------------------------------------
# 신경망 모델 학습
#----------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')



#----------------------------------------
# 테스트
#----------------------------------------

# 레이블값이 정수이므로 예측값도 정수로 변경
prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# one-hot 인코딩이 아니므로 입력값을 그대로 비교
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, 
                                            Y: target_batch})



#----------------------------------------
# 테스트 결과 출력
#----------------------------------------
predict_words = []

for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)
    
print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)


# <br>
# <br>
# <br>
# < transpose 추가 설명 >
'''
# In[6]:


# outputs 결과에서 마지막 타임 스텝만 구함
# outputs의 형태를 변경
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
#
# batch_size -> 2
# n_step -> 3
# n_hidden -> 2
outputs = [[[1,2], [3,4], [5,6]], [[11,12], [13,14], [15,16]]]

# 행렬의 차원을 [0, 1, 2]에서 [1, 0, 2]로 변경
outputs = np.transpose(outputs, [1, 0, 2])
print(outputs)

# n_step을 삭제하고 마지막 타임 스텝만 구함
outputs = outputs[-1]
print('\n')
print(outputs)


# <br>
# <br>
# <br>
# <br>
# <br>

# ## (3) Seq2Seq로 단어 번역

# <br>

# < 원본 코드 ><br>
# -> https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/03%20-%20Seq2Seq.py<br>
# <br><br><br>
# < 프로그램 설명 ><br>
# <br>
# 영어 단어를 한글로 번역<br>
# <br>
# --------------------------<br>
# wood -> 나무<br>
# game -> 놀이<br>
# --------------------------<br>
# <br><br><br>

# In[7]:


import tensorflow as tf
import numpy as np



# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]



#----------------------------------------
# 데이터 배치 생성
#----------------------------------------
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만듬.
        input = [num_dic[n] for n in seq[0]]
        
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여줌.
        output = [num_dic[n] for n in ('S' + seq[1])]
        
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙임.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        # One-hot 인코딩으로 배치 추가
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        
        # 출력값만 one-hot 인코딩이 아님(sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch



#----------------------------------------
# 첫번째 배치 출력
# input : w, o, r, d
# output : S, 단, 어
# target : 단, 어, E
#----------------------------------------
input_batch, output_batch, target_batch = make_batch(seq_data)

print(input_batch[0])
print('\n')
print(output_batch[0])
print('\n')
print(target_batch[0])


# <br>

# In[8]:


#----------------------------------------
# 옵션 설정
#----------------------------------------
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같음
n_class = n_input = dic_len



#----------------------------------------
# 신경망 모델 구성
#----------------------------------------

# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같음
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])

# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, 
                                             output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, 
                                             output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심
    # 인코더 셀과 달리 initial_state에 enc_states를 설정
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

# 이전 RNN으로 캐릭터 글자 예측과 달리 타입 스텝을 삭제하지 않음
# 각 타임 스텝마다 단어 출력
model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



#----------------------------------------
# 신경망 모델 학습
#----------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')



#----------------------------------------
# 단어를 입력받아 번역 단어를 예측하고 디코딩
#----------------------------------------
def translate(word):

    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로 디코더의 입출력값을 의미 없는 값인 P 값으로 채움
    # ['word', 'PPPP']
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax로 취해 가장 확률이 높은 글자를 예측 값으로 만듬
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]
    
    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated



print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))

'''