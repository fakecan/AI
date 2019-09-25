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


# ------------------------------------------------------------
# 데이터 배치 생성
# ------------------------------------------------------------
def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

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


# ------------------------------------------------------------
# 첫번째 배치 출력
# input : w, o, r, d
# output : S, 단, 어
# target : 단, 어, E
# ------------------------------------------------------------
input_batch, output_batch, target_batch = make_batch(seq_data)

print(input_batch[0])
print('\n')
print(output_batch[0])
print('\n')
print(target_batch[0])


# ------------------------------------------------------------
# 옵션 설정
# ------------------------------------------------------------
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

# 입력과 출력의 형태가 원핫 인코딩으로 같으므로 크기도 같음
n_class = n_input = dic_len


# ------------------------------------------------------------
# 신경망 모델 구성
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 신경망 모델 학습
# ------------------------------------------------------------
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