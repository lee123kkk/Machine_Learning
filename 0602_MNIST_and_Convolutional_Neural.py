# Lab 11 MNIST and Convolutional Neural Network

'''
📚 CNN(합성곱 신경망)의 구조와 작동 원리를 이해합니다.

🧠 이미지 분류 문제에 CNN을 어떻게 적용하는지 실습합니다.

🧪 하이퍼파라미터 튜닝의 효과와 중요성을 배웁니다.

🚀 실제 응용 가능한 예제들을 통해 문제 해결 역량과 응용력을 기릅니다.
'''
import numpy as np
import tensorflow as tf
import random

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)
random.seed(777)

mnist = tf.keras.datasets.mnist

# 1️⃣ [데이터 로드 및 정규화]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_train = x_train / 255.0

# ⭐ [핵심 차이점] 2차원 형태 유지!
# 이전: (60000, 784)로 평탄화
# CNN: (60000, 28, 28, 1)로 변환. (데이터수, 가로, 세로, 채널수)
# 흑백 이미지이므로 채널(색상)은 1입니다. (컬러면 3)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 하이퍼파라미터 설정
learning_rate = 0.001
training_epochs = 12
batch_size = 128

tf.model = tf.keras.Sequential()

# 2️⃣ [Layer 1: 특성 추출기]
# Conv2D: 3x3 크기의 돋보기(필터) 16개를 사용해 이미지를 훑으며 특징을 찾습니다.
tf.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
# MaxPooling2D: 2x2 크기로 묶어서 가장 큰 값만 남겨 이미지 크기를 절반으로 줄입니다.
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 3️⃣ [Layer 2: 더 깊은 특성 추출기]
# 이전 단계의 결과를 바탕으로 다시 3x3 필터 32개를 사용해 더 복잡한 특징(동그라미, 꺾인 선 등)을 찾습니다.
tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 4️⃣ [Layer 3: 분류기 (Fully Connected)]
# Flatten: 추출이 끝난 특징들을 최종 판단하기 위해 1차원으로 쫙 폅니다.
tf.model.add(tf.keras.layers.Flatten())
# Dense: 최종적으로 어떤 숫자인지 10개로 분류합니다. (출력층)
tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

# 5️⃣ [컴파일 및 학습]
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()

tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# 6️⃣ [예측 및 평가]
y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
#==================================================================
# CNN(합성곱 신경망)
# 기존의 일반 신경망이 2차원 이미지를 1차원으로 펴 버린것과 달리, 
# CNN은 합성곱과 풀링을 통해서 공간 정보를 그대로 살린다.

# 이전 예제에서는 파라미터가 100만개에서 230만개였지만, CNN모델에서는 12810개이다.
# 최종 실전 테스트의 정확도는 98.76%이고 학습 정확도는 99.17%로 학습 속도와 안정성 모두 더 커졌다.

# CNN을 통해서 공간 정보를 보존하며 특징만 효율적으로 추출하는 것이
# 연산량과 정확도 면에서 더 우수하다.

