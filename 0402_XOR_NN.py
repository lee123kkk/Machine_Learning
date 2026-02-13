# Lab 9 XOR-NN
import tensorflow as tf
import numpy as np

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()

# 은닉층: 뉴런 10개 (충분한 뇌 용량)
# 활성화 함수: 시그모이드 대신 'relu'를 쓰면 더 좋지만, 
# 지금은 Adam의 위력을 보기 위해 sigmoid를 유지하겠습니다.
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=2, activation='sigmoid'))

# 출력층: 최종 결과 1개
tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# [핵심 수정] SGD -> Adam
# learning_rate도 0.01 ~ 0.1 정도로 설정하면 아주 잘 됩니다.
tf.model.compile(loss='binary_crossentropy', 
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), 
                 metrics=['accuracy'])

tf.model.summary()

# 학습 수행
history = tf.model.fit(x_data, y_data, epochs=2000, verbose=0)

# 결과 확인
predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])
#=====================================================================
# 다층 퍼셉트론으로 XOR문제 해결
# 이론상으로 문제가 해결되어야 하지만, 해결되지 않았다.
# 은닉층의 뉴런 개수(unit=2)가 너무 적거나, SGD가 길을 못찾고 있기 때문이다.
# 은닉층의 뉴런 개수를 늘려도 해결이 되지 않았는데, SGD를 Adam으로 바꾸니까 정상적으로 결과가 나왔다.  
# 입력층과 출력 층 사이에 은닉층을 추가함으로써 인공지능이 비선형 문제를 해결할 수 있다.

