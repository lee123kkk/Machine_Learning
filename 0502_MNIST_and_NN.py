# Lab 10-2 MNIST and NN

'''
신경망(Neural Network)의 구조와 작동 원리 이해

이미지 분류 문제에 딥러닝 모델을 적용하는 방법 학습

텐서플로우(TensorFlow)를 활용한 실전 프로젝트 구현

성능 향상을 위한 하이퍼파라미터 튜닝 실습
'''

import numpy as np
import random
import tensorflow as tf

# 랜덤 시드 고정 (재현성 확보)
random.seed(777)
tf.random.set_seed(777) # Tensorflow 내부 시드도 고정하는 것이 좋습니다.

# [하이퍼파라미터]
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# [데이터 로드]
# 이번 코드의 특이점: / 255.0 정규화(Normalization) 과정이 빠져 있습니다.
# (정규화를 안 하면 초기 손실값이 매우 크게 튈 수 있지만, Adam이 똑똑해서 학습은 됩니다.)
(x_train, y_train), (x_test2, y_test) = tf.keras.datasets.mnist.load_data()

# [데이터 전처리] 
# 2차원 이미지(28x28)를 1차원(784)으로 펴주기
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test2.reshape(x_test2.shape[0], 28 * 28)

# 원-핫 인코딩 (숫자 5 -> [0,0,0,0,0,1,0,0,0,0])
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# [모델 구성] 심층 신경망 (Deep Neural Network)
tf.model = tf.keras.Sequential()

# 은닉층 1: 784개 입력을 받아 256개 특징으로 변환, 활성화 함수는 ReLU
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=256, activation='relu'))

# 은닉층 2: 256개 특징을 다시 256개로 가공 (더 깊은 추론), 활성화 함수는 ReLU
tf.model.add(tf.keras.layers.Dense(units=256, activation='relu'))

# 출력층: 최종적으로 10개의 숫자 확률로 압축 (Softmax)
tf.model.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))

# [컴파일] Adam 최적화 도구 사용
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])
tf.model.summary()

# [학습 수행]
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# [랜덤 테스트] 테스트 데이터 중 무작위로 10개를 뽑아 예측 결과 확인
y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),    # 실제 정답
          "predicted y: ", np.argmax(y_predicted[random_index])) # AI 예측

# [최종 평가] 테스트 데이터 전체에 대한 정확도 검증
evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
#===================================================================
# (Deep Nerual Network)심층 신경망을 구현
# 입력층과 출력층 사이의 은닉층 추가: 곡선이나 복잡한 면을 그려서 숫자 구분 가능
# ReLU 활성화 함수 도입: softmax대신 relu를 사용해서 신호가 약하면 끄고 강하면 내보낸다. 

# 결과 분석
# 초기 loss값이 116.531로 크다. 코드에 정규화 과정이 빠졌기 때문
# 학습 정확도: 98.51%, 테스트 정확도: 96.65%로 softmax사용때보다 정확도가 상승

# 단일층의 한계를 넘어서 은닉층과 활성화 함수(ReLU)를 추가하여 신경망을 깊게 쌓으면 
# 이미지 분류 정확도가 획기적으로 상승한다.

