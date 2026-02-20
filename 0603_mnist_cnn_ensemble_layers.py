# Lab 11-2-mnist_cnn_ensemble_layers

'''
CNN(합성곱 신경망)의 구조와 학습 흐름을 명확히 이해할 수 있습니다.

앙상블 학습이 왜 강력한지, 어떻게 정확도를 높이는지 체감할 수 있습니다.

실제로 모델을 5개 생성하고 평균 예측을 통해 정확도를 개선하는 법을 배웁니다.

하이퍼파라미터 튜닝이 모델 성능에 어떤 영향을 주는지도 실험합니다.

실생활 데이터를 활용한 문제 해결 능력을 향상시킬 수 있습니다.
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# [데이터 로드 및 전처리]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

# CNN에 넣기 위해 채널 차원(1) 추가 -> (N, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)  
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 하이퍼파라미터
learning_rate = 0.001
training_epochs = 20
batch_size = 100
num_models = 5 # ⭐ 5개의 모델(전문가)을 만들 것입니다.

# [모델 생성 함수]
# 똑같은 구조의 모델을 여러 개 찍어내기 위해 함수로 정의합니다.
def build_model():
    model = models.Sequential()
    # 깊고 넓은 CNN 구조 + 드롭아웃(과적합 방지)
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(625, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ⭐ [앙상블 모델 생성] 리스트 안에 5개의 똑같은 모델을 담습니다.
models_list = [build_model() for _ in range(num_models)]

# [학습 수행]
print('Learning Started!')
for epoch in range(training_epochs):
    # 5개 모델의 오차를 각각 저장할 배열
    avg_cost_list = np.zeros(num_models)
    total_batch = int(x_train.shape[0] / batch_size)

    for i in range(total_batch):
        batch_xs = x_train[i*batch_size:(i+1)*batch_size]
        batch_ys = y_train[i*batch_size:(i+1)*batch_size]

        # ⭐ 핵심: 데이터를 한 번 가져올 때마다 5개의 모델을 번갈아가며 모두 학습시킵니다.
        for m_idx, model in enumerate(models_list):
            history = model.train_on_batch(batch_xs, batch_ys)
            avg_cost_list[m_idx] += history[0] / total_batch

    print(f"Epoch: {epoch+1:04d}, cost = {avg_cost_list}")

print('Learning Finished!')

# [예측 및 평가]
predictions = np.zeros((x_test.shape[0], 10)) # 최종 투표함 (10000개 데이터, 10개 클래스)

for m_idx, model in enumerate(models_list):
    # 개별 모델의 정확도를 평가합니다.
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model {m_idx} Accuracy: {acc:.4f}")
    
    # ⭐ 앙상블 투표: 5개 모델이 예측한 확률값(0~1)을 모두 더합니다.
    predictions += model.predict(x_test)

# 모든 투표를 더한 값 중 가장 큰 확률을 가진 인덱스(0~9)를 최종 정답으로 채택합니다.
ensemble_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# 최종 앙상블 정확도 계산
ensemble_accuracy = np.mean(ensemble_pred == y_true)
print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")

#====================================================================
# CNN에 앙상블 기법 적용
# 앙상블: 여러 개의 모델을 만들어 각자의 예측값을 합산하여 최종 결정을 내리는 기법
# 한 모델이 실수하더라도 다른 모델들이 정답을 맞춰서 약점을 상호 보완한다.

# 에포크마다 5개의 숫자가 담긴 배열이 출력 
# -> 5개의 개별 모델이 각각 자신만의 방식으로 동시에 학습하여 오차를 줄인다.

# 개별 평가를 보면 각각 99.23%에서 99.46%의 훌률한 정확도를 보여준다
# 최종 앙상블 정확도는 99.49%이다.
# 개별 모델 중 가장 성능이 좋은 모델보다 높은 점수이다.

# 여러 개의 신경망을 병령로 훈련시키고 합치는 앙상블 기법을 활용하면
# 개별 모델들의 오판을 상호 보완하여 한계 정확도를 돌파할 수 있다.
