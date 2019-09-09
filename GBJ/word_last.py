# from konlpy.tag import Okt
# okt = Okt() # Kkma, Komoran etc..
# print(okt.nouns('나는 자연어 처리를 공부한다.'))   # 명사
# print(okt.morphs('나는 자연어 처리를 공부한다.'))  # 형태소
# print(okt.pos('나는 자연어 처리를 공부한다.'))     # 품사

# ◆◆◆◆◆◆◆◆◆◆ RNN 캐릭터 글자 예측 ◆◆◆◆◆◆◆◆◆◆
import tensorflow as tf
import numpy as np

# 캐릭터 설정
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# {'a':0, 'b':1, 'c':2 ..., 'x':23, 'y':24, 'z':25}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)  # 26

# ex) 'word':  wor -> X, d-> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 
            'cool', 'load', 'love', 'kiss', 'kind']


# ------------------------------------------------------------
# 데이터 배치 생성
# ------------------------------------------------------------
def make_batch(seq_data):
    
    input_batch, target_batch = [], []
    # target_batch = []

    for seq in seq_data:
        # input_batch, target_batch: 알파벳 배열의 인덱스 번호
        input = [num_dic[n] for n in seq[:-1]]  # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
        target = num_dic[seq[-1]]   # 3, 3, 15, 4, 3 ...
        
        # One-Hot encoding
        # if input is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])
        
        # 비용함수의 label 값
        # softmax_cross_entropy_with_logits -> One-Hot encoding
        # sparse_softmax_cross_entropy_with_logits -> index 숫자
        target_batch.append(target)

    return input_batch, target_batch

input_batch, target_batch = make_batch(seq_data)
print(input_batch[0])
print('\n')
print(target_batch[0])

learning_rate = 0.01
total_epoch = 30
batch_size = 32 # 옵션

n_step = 3  # RNN을 구성하는 시퀀스의 개수
n_input = 26    # = dic_len
n_class = 26
n_hidden = 128

# ------------------------------------------------------------
# 신경망 모델 구성
# ------------------------------------------------------------
# 기본 식 설정
X = tf.placeholder(tf.float32, [None, n_step, n_input]) # (?, 3, 26)
Y = tf.placeholder(tf.int32, [None])                    # (?,)
W = tf.Variable(tf.random_normal([n_hidden, n_class]))  # (128, 26)
b = tf.Variable(tf.random_normal([n_class]))            # (26,)

# RNN Cell
# 멀티 셀
cell1 = tf.nn.rnn_cell.LSTMCell(n_hidden)
cell2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

# 순환 신경망 생성
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# print(outputs)  # shape=(?, 3, 128), dtype=float32 -> (?, n_step, n_hidden)

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# Y : [batch_size, n_class]
# outputs의 형태를 이에 맞춰 변경해야 합니다.
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

# 128개 히든 레이어의 output을 26개 알파벳의 원핫 인코딩 형식으로 변경
model = tf.matmul(outputs, W) + b

# 비용 함수
cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model, labels=Y))

# 옵티마이저
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# ------------------------------------------------------------
# 신경망 모델 학습
# ------------------------------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'Cost= ', '{:.6f}'.format(loss)) 

print('최적화 완료!')                      

# ------------------------------------------------------------
# 테스트
# ------------------------------------------------------------
# labels 값(Y)이 정수이므로 예측값도 정수로 변경
prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# 원핫 인코딩이 아니므로 입력값을 그대로 비교
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch,
                                            Y: target_batch})

# ------------------------------------------------------------
# 테스트 결과 출력
# ------------------------------------------------------------
predict_words = []

for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n======== 예측 결과 ========')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)