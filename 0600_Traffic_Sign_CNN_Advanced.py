

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ [데이터 로드] 변경된 부분
print("🚀 CIFAR-10 사물 인식 데이터를 다운로드합니다...")
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',  # 👈 'gtsrb'를 'cifar10'으로 변경합니다!
    split=['train', 'test'],
    with_info=True,
    as_supervised=True,
    data_dir='./my_tf_data',
    download=True 
)

# 총 43종류의 교통 표지판이 있습니다. (속도 제한, 진입 금지, 멈춤 등)
nb_classes = ds_info.features['label'].num_classes
IMG_SIZE = 32 # 모든 이미지를 32x32 크기로 통일

# 2️⃣ [데이터 전처리 함수]
# 실전 데이터는 크기가 다 다르므로 강제로 리사이즈해야 모델에 넣을 수 있습니다.
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) # 32x32로 크기 통일
    image = image / 255.0                                # 0~1 정규화
    label = tf.one_hot(label, depth=nb_classes)          # 원-핫 인코딩
    return image, label

# 데이터 파이프라인 구축 (메모리 폭발을 막기 위해 덩어리 단위로 처리)
BATCH_SIZE = 64
train_data = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3️⃣ [모델 구성] 실전형 CNN (Batch Normalization + Dropout 추가)
model = tf.keras.Sequential()

# Layer 1: 특성 추출 + 배치 정규화
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
# ⭐ 배치 정규화: 각 층에 들어가는 데이터를 가지런히 정돈하여 학습 속도와 안정성을 대폭 끌어올립니다.
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Layer 2: 깊은 특성 추출
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Layer 3: 분류기 (Fully Connected)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

# ⭐ 드롭아웃: 너무 똑같이 외우지 않도록 뉴런의 50%를 끕니다 (과적합 방지).
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

# 4️⃣ [컴파일] Optimizer 비교 (Adam vs SGD)
# SGD: 천천히 꼼꼼하게 산을 내려갑니다. 설정(학습률 등)을 기가 막히게 맞추면 최고 성능을 낼 수도 있지만 튜닝이 매우 어렵습니다.
# Adam: 상황에 맞춰 보폭을 영리하게 조절합니다. 실무에서 가장 기본적이고 강력하게 쓰입니다. (여기서는 Adam 사용)
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              metrics=['accuracy'])
model.summary()

# 5️⃣ [학습 수행]
training_epochs = 15
print("\n🚀 자율주행 교통 표지판 인식 모델 학습 시작...")
history = model.fit(train_data, epochs=training_epochs, validation_data=test_data)

# 6️⃣ [최종 평가]
print("\n" + "="*50)
evaluation = model.evaluate(test_data)
print(f"최종 Loss: {evaluation[0]:.4f}")
print(f"최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")
#===================================================================
# CNN에 배치 정규화와 드롭 아웃을 결합하여 교통 표지판 인식기 구축

# 사용하려던 GTSRB 데이터 셋 원본 서버에서 다운로드를 차단하는 문제가 발생했다.
# CIRAF-10 데이터 셋으로 대체해서 사용했다.

# 총 파라미터 수: 31.6만개로 이전 MNIST모델(1.2만개)보다 무겁지만 
# 컬러 이미지의 특징을 담아내기에 효율적인 규모
# 배치 정규화: 학습 속도를 높여주고, 안정적으로 오차를 줄여줌
# 드롭아웃: 과적합 방지

# MLP(단순 신경망)에서는 CIFAR-10에서 46.19%의 정확도였지만 CNN에서는 70.71%를 달성했다

# GPU의 사욜량을 보면 텐서플로우에 정상적으로 할당되었음을 확인할 수 있다.
# 에포크 당 약 2초 내외로 학습이 완료되었다.

# CNN 모델을 통해서 CIFAR-10 데이터 셋에서 70%이상의 높은 성능을 낼 수 있다.