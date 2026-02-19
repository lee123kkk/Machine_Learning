# Lab 10-4 MNIST and NN deep

'''
신경망(Neural Network)의 기본 구조와 학습 과정을 이해할 수 있습니다.

딥러닝 모델 설계 및 활성화 함수, 초기화, 최적화 기법에 대한 실습을 통해 직관을 얻을 수 있습니다.

실제 손글씨 숫자(MNIST) 데이터셋을 활용하여 모델을 학습하고 예측하는 전 과정을 체험합니다.

파라미터 튜닝을 통한 성능 향상 경험과, 그로 인한 실제 응용 가능성을 확인합니다.
'''

import numpy as np
import random
import tensorflow as tf

random.seed(777)
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# [데이터 로드]
(x_train, y_train), (x_test2, y_test) = tf.keras.datasets.mnist.load_data()

# [데이터 전처리]
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test2.reshape(x_test2.shape[0], 28 * 28)

y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# [모델 구성] Deep & Wide Neural Network
tf.model = tf.keras.Sequential()

# Layer 1: 입력(784) -> 출력(512), Xavier 초기화
# 뉴런 수를 512개로 늘려서 정보 처리량을 대폭 확대했습니다.
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))

# Layer 2: 512 -> 512
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))

# Layer 3: 512 -> 512 (새로 추가됨!)
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))

# Layer 4: 512 -> 512 (새로 추가됨!)
# 층이 깊어질수록 추상적인 특징(직선 -> 곡선 -> 숫자 모양)을 더 잘 잡아냅니다.
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))

# Output Layer: 512 -> 10 (Softmax)
tf.model.add(tf.keras.layers.Dense(units=nb_classes, 
                                   kernel_initializer='glorot_normal', activation='softmax'))

# [컴파일]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()
# summary를 보면 파라미터(Weight) 개수가 100만 개(1.2M)에 육박함을 알 수 있습니다.

# [학습]
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

#================================================================
# 더 깊고 더 넓은 신경망 구축
# 은닉층을 2개에서 4개로 늘렸다
# 각 층 뉴런의 개수를 256개에서 512개로 2배 늘렸다

# 학습 정확도는 이전의 98.3%에서 이번 98.87%로 더 커진것을 확인할 수 있다.
# 테스트 정확도도 이전의 96.43%에서 이번의 97.06%로 더 커졌다. 

# 신경망을 더 깊고 넓게 설계하고 올바른 초기화 모델을 사용하면, 
# 모델의 표현력이 극대화되어 정확도를 한계까지 끌어 올릴 수 있다.  
