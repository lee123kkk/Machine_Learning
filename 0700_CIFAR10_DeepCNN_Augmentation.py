# 0700_CIFAR10_DeepCNN_Augmentation

import tensorflow as tf
import os
import datetime

# 1️⃣ [데이터 로드 및 전처리] TFDS 대신 기본 Keras 데이터셋 사용 (빠른 로드를 위해)
print("🚀 CIFAR-10 데이터 로드 중...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 0~1 정규화 및 float32 변환 (GPU 연산 최적화)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 원-핫 인코딩
nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# 2️⃣ [데이터 증강 (Data Augmentation)]
# Keras 레이어로 만들어 모델 내부에 삽입하면 GPU를 사용해 매우 빠르게 처리됩니다.
data_augmentation = tf.keras.Sequential([
    # 이미지를 무작위로 좌우 반전
    tf.keras.layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    # 이미지를 최대 10% 무작위 회전
    tf.keras.layers.RandomRotation(0.1),
    # 이미지를 무작위로 확대/축소
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# 3️⃣ [모델 구성] 더 깊은 CNN (VGG 스타일: Conv-Conv-Pool 구조)
model = tf.keras.Sequential()

# 증강 레이어를 모델의 맨 처음에 배치
model.add(data_augmentation)

# [Block 1] 얕은 특징(선, 색감 등) 추출
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2)) # 낮은 층은 조금만 드롭아웃

# [Block 2] 중간 특징(형태, 질감 등) 추출 (채널을 64개로 늘림)
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

# [Block 3] 깊은 특징(고차원 패턴) 추출 (채널을 128개로 늘림)
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.4))

# [분류기 (Classifier)]
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5)) # 깊은 층은 많이 드롭아웃
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

# 4️⃣ [컴파일]
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

# 5️⃣ [⭐핵심: 콜백 함수 설정]
# (1) TensorBoard: 학습 과정을 시각화하여 저장
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# (2) ModelCheckpoint: 학습 중 검증 정확도(val_accuracy)가 가장 높았던 최고 모델만 저장
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_cifar10_cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# (3) EarlyStopping: 성능이 더 이상 안 오르면 10분 내에 끝내기 위해 조기 종료 설정
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=8, # 8번 연속으로 성능이 안 오르면 학습 중단
    restore_best_weights=True # 가장 좋았던 가중치로 복구
)

# 6️⃣ [학습 수행] RTX 4060의 성능을 테스트!
training_epochs = 40 # 40번 돌려도 4060이면 금방 끝납니다.
batch_size = 128

print("\n🚀 데이터 증강 및 VGG 스타일 CNN 학습 시작...")
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=training_epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb])

# 7️⃣ [최종 평가]
print("\n" + "="*50)
evaluation = model.evaluate(x_test, y_test)
print(f"최종 Loss: {evaluation[0]:.4f}")
print(f"최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")

# 데이터 증강 + 더 깊은 CNN 설계

# 학습 초기의 정확도 하락 현상: 
# 데이터 증강 레이어 때문에 훈련 데이터가 뒤틀리고 있기 때문에 초기에는 이전 모델보다 낮아졌다.
# 꾸준한 우상향 곡선:
# 실전 정확도가 꾸준히 오르는 모습을 보이며 과적합을 억제하고 있다.

# 초반 정확도는 29.6%로 낮게 출발했다.
# 파란선과 주황선이 함꼐 우상향하고 있다. 과적합을 잘 방지하고 있다.
# 8번 연속으로 성능이 오르지 않자 조기 종료되었다. 이포크 26에서 학습을 멈췄고, 
# 최종 정확도 76.33%를 기록했다.

