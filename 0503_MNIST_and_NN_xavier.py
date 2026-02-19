# Lab 10-3 MNIST and NN xavier

'''
이 강의를 통해 여러분은 다음과 같은 실질적인 능력을 기르게 됩니다:

신경망에서 Xavier (Glorot) 초기화가 중요한 이유를 이해하게 됩니다.

MNIST 데이터셋을 통해 이미지 분류 모델을 설계, 학습, 평가하는 전 과정을 실습하게 됩니다.

TensorFlow를 활용한 모델 구현, 하이퍼파라미터 튜닝, 성능 향상 기법을 배웁니다.

실제 생활과 연관된 예제를 통해, AI 모델이 우리의 일상에 어떻게 적용되는지를 체감하게 됩니다.
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
# 이번에도 정규화(/ 255.0) 코드는 빠져있습니다. (Xavier의 위력을 테스트하기 위함)
(x_train, y_train), (x_test2, y_test) = tf.keras.datasets.mnist.load_data()

# [데이터 전처리] 2차원(28x28) -> 1차원(784)
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test2.reshape(x_test2.shape[0], 28 * 28)

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

tf.model = tf.keras.Sequential()

# [핵심 변경 사항] Xavier 초기화 (Glorot Normal)
# kernel_initializer='glorot_normal': 가중치(W)의 초기값을 Xavier 방식으로 설정합니다.
# 이전에는 이 옵션이 없어서 기본값(작은 랜덤값)을 썼지만, 이제는 입력(784)과 출력(256) 비율에 맞춰 똑똑하게 결정합니다.
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=256, 
                                   kernel_initializer='glorot_normal', activation='relu'))

tf.model.add(tf.keras.layers.Dense(units=256, 
                                   kernel_initializer='glorot_normal', activation='relu'))

tf.model.add(tf.keras.layers.Dense(units=nb_classes, 
                                   kernel_initializer='glorot_normal', activation='softmax'))

# [컴파일]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()

# [학습]
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# [예측 테스트]
y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

# [최종 평가]
evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
#===================================================================
# 심층 신경망(NN) 학습을 시작학때 가중치의 첫번 째 숫자를 어떻게 정해야 할까?
# 이전 예제에서는 가중치를 랜덤으로 설정하고 학습을 시작했지만,
# 이번 예제에서는 Xavier Initialization 기법을 사용했다.
# Input과 Output의 개수를 맞춰서 가장 적절한 범위의 난수를 자동으로 계산해서 초기값으로 활용한다.
# 신호가 너무 강해지거나 사라지는 현상을 막아줘서 학습이 훨신 더 안정적이고 빨라진다.


# 이전 예제에서는 Epoch 1 loss가 118.53이었는데, 이번에는 2.09로 오차가 50배 이상 줄었다.
# 최종 정확도는 96.43%로 이전과 비슷하지만, 학습 초반에 훨씬 더 빠르게 정답에 수렴했다.

# 가중치 초기화를 잘 설정한느 것만으로도 학습 초기의 엄청난 오차를 잡고, 
# 모델을 훨씬 안정적이고 빠르게 학습시킬 수 있다.
