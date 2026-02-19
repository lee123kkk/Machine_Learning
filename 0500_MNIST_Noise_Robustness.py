#MNIST_Noise_Robustness

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt # 노이즈 이미지를 눈으로 보기 위해 추가

# 1. [시드 고정]
random.seed(777)
tf.random.set_seed(777)

# 2. [하이퍼파라미터 설정]
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10
drop_rate = 0.3 # 드롭아웃은 노이즈 제거에도 큰 도움을 줍니다.

# 3. [데이터 로드]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 4. [데이터 전처리]
# (1) 평탄화 및 정규화
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32') / 255.0

# (2) 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# =================================================================
# [⭐ 핵심 추가] 이미지에 노이즈(잡음) 강제로 주입하기
# =================================================================
noise_factor = 0.5 # 노이즈 강도 (0.0 ~ 1.0, 클수록 지저분해짐)

print(f"🌫️ 데이터에 노이즈를 주입합니다 (강도: {noise_factor})...")

# np.random.normal: 정규분포를 따르는 무작위 잡음을 생성해 원본에 더함
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# np.clip: 노이즈를 더하다 보면 픽셀값이 1.0을 넘거나 0.0보다 작아질 수 있음.
# 이를 0.0 ~ 1.0 사이로 강제로 잘라냄 (유효 범위 유지)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# [👀 시각화] 노이즈가 낀 이미지가 어떻게 생겼는지 확인해봅시다.
plt.figure(figsize=(10, 4))
for i in range(5):
    # 원본 이미지
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 노이즈 이미지
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(x_train_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')
plt.show()
# =================================================================

# 5. [모델 구성] Lab 10-5와 동일 (Deep + Wide + Dropout)
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=512, kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))
tf.model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))
tf.model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))
tf.model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dropout(drop_rate))
tf.model.add(tf.keras.layers.Dense(units=nb_classes, kernel_initializer='glorot_normal', activation='softmax'))

# 6. [컴파일]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])

# 7. [학습 수행] *중요: x_train 대신 x_train_noisy를 사용합니다!
print("\n🔥 노이즈가 섞인 이미지로 학습을 시작합니다...")
history = tf.model.fit(x_train_noisy, y_train, batch_size=batch_size, epochs=training_epochs)

# 8. [예측 테스트]
y_predicted = tf.model.predict(x_test_noisy)

print("\n" + "="*50)
print("🔍 노이즈 이미지 예측 결과 확인")
print("="*50)
for x in range(0, 10):
    random_index = random.randint(0, x_test_noisy.shape[0]-1)
    
    actual_val = np.argmax(y_test[random_index])
    pred_val = np.argmax(y_predicted[random_index])
    
    result = "✅ 정답" if actual_val == pred_val else "❌ 오답"
    print(f"Index: {random_index:<5} | 정답: {actual_val} vs 예측: {pred_val} -> {result}")

# 9. [최종 평가]
evaluation = tf.model.evaluate(x_test_noisy, y_test)
print("="*50)
print(f"노이즈 환경 최종 정확도: {evaluation[1]*100:.2f}%")
#===============================================================
# 드롭아웃 적용 모델을 기반으로 노이즈가 잔뜩 낀 악조건 속에서 숫자를 맞춰내는 AI
# 현실 세계에서의 데이터처러 잡티가 많은 데이터를 섞어서 학습

# 노이즈를 더하면 0에서 1사이의 범위를 벗어난 값들이 생기는데 이 값들을 제거하는 클리핑을 수행한다.

# 첫번째 에포크는 74.55%로 깨끗한 데이터를 썼을 때 보다 낮게 출발했다.
# 에포크가 지날수록 정확도가 상승해서 96.64%에 도달했다.
# 최종 정확도는 92.43%로 노이즈가 없을 때보다 떨어졌지만, 좋은 결과를 냈다.

# 학습 데이터에 일부러 노이즈를 섞어 훈련하면 악조건 속에서도 작동하는 AI를 만들 수 있다.
