# Lab 10-5 MNIST and NN dropout

'''
딥러닝 기본 구조 이해: Dense Layer, 활성화 함수(ReLU), 드롭아웃, 소프트맥스의 역할을 명확히 이해할 수 있습니다.

MNIST를 통한 실습 경험: 대표적인 이미지 분류 데이터셋으로 딥러닝 실습 능력을 키울 수 있습니다.

하이퍼파라미터 튜닝의 중요성 인식: 드롭아웃, 학습률, batch size, epoch 등이 성능에 어떤 영향을 주는지 체험합니다.

실생활 문제 적용: 숫자 인식 외 다양한 분야에 딥러닝을 어떻게 적용할 수 있는지 감을 잡을 수 있습니다.
'''

import numpy as np
import random
import tensorflow as tf

random.seed(777)
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# [핵심] 드롭아웃 비율 설정 (0.3 = 30%)
# 학습 과정에서 전체 뉴런 중 30%를 무작위로 쉬게 합니다.
drop_rate = 0.3

# [데이터 로드]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# [데이터 전처리]
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

tf.model = tf.keras.Sequential()

# Layer 1: Dense -> Dropout
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate)) # 512개 중 30%를 끕니다.

# Layer 2: Dense -> Dropout
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate)) # 또 30%를 끕니다.

# Layer 3: Dense -> Dropout
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))

# Layer 4: Dense -> Dropout
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))

# Output Layer: (출력층에는 보통 드롭아웃을 쓰지 않습니다)
tf.model.add(tf.keras.layers.Dense(units=nb_classes, 
                                   kernel_initializer='glorot_normal', activation='softmax'))

# [컴파일]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()
# Summary를 보면 Dense 층 사이에 Dropout 층이 끼어있는 것을 볼 수 있습니다.

# [학습 수행]
# 주의: 드롭아웃은 '학습(fit)'할 때만 작동하고, '평가(evaluate/predict)'할 때는 자동으로 꺼집니다.
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# [예측]
y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

# [평가]
evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])

#====================================================================
# 과적합 문제를 해결하기 위해 드롭 아웃 기법 사용
# 모델을 깊고 넓게 만들었더니 학습 데이터를 너무 완벽하게 외워 버려서 새로운 문제에 약해질 위험이 있었다.
# 드롭아웃: 학습할때 일부 뉴런을 랜덤하게 꺼버리는 기법

# 학습 정확도: 98.87%에ㅐ서 96.16%로 떨어짐 
# 실전 정확도: 97.96%에서 97.41%로 높아짐. 

# 드롭아웃을 사용하면 과적합을 방해하고 테스트에 더 강하고 유연한 모델을 만든다.
