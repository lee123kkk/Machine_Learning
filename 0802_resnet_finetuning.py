#0802_resnet_finetuning

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import datasets, Input

# =========================================================
# [이전 단계] 데이터 준비 및 1차 학습 (특징 추출기 고정)
# =========================================================
print("--- 데이터 로드 및 1차 학습 준비 ---")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
input_y = tf.keras.utils.to_categorical(train_labels, 10)
test_y = tf.keras.utils.to_categorical(test_labels, 10)

# ResNet50 베이스 모델 로드 및 가중치 동결
base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, pooling='avg', input_shape=(32,32,3), weights='imagenet'
)
base_model.trainable = False 

# 모델 조립
inputs = Input(shape=(32,32,3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)
x = Flatten()(x)                                                       
outputs = Dense(10, activation='softmax')(x)  
model_res = tf.keras.Model(inputs, outputs)  

# 1차 학습 컴파일 및 실행 (빠른 테스트를 위해 epoch를 3으로 조정함)
model_res.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print("--- 1차 전이 학습 시작 ---")
model_res.fit(train_images, input_y, epochs=3, validation_data=(test_images, test_y), batch_size=256)


# =========================================================
# [새로운 단계] STEP 6 ~ STEP 10
# =========================================================

# STEP 6: 미세조정 (Fine-tuning)
print("\n--- STEP 6: 미세조정(Fine-tuning) 준비 ---")
base_model.trainable = True # 전체 레이어 동결 해제

# 마지막 30개 레이어만 학습을 허용하고, 그 앞단은 다시 동결
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 기존 지식이 파괴되지 않도록 매우 작은 학습률(1e-5)로 재컴파일
model_res.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# STEP 7: 정확도/손실 시각화 (중복 코드를 합쳐 history 변수에 바로 저장)
print("\n--- STEP 7: 미세조정 학습 및 시각화 ---")
history = model_res.fit(train_images, input_y, epochs=5, validation_data=(test_images, test_y), batch_size=256)

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy per Epoch (Fine-tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# STEP 8: 실시간 예측 및 확인
print("\n--- STEP 8: 실시간 예측 확인 ---")
sample = test_images[0:1] # (1, 32, 32, 3) 형태로 데이터 한 개 추출
pred = model_res.predict(sample) # 수정됨: model -> model_res
class_idx = np.argmax(pred)

# CIFAR-10 클래스 이름 (시각화용)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(3, 3))
# 수정됨: test_images는 이미 numpy 배열이므로 .numpy() 메서드가 불필요하여 제거함
plt.imshow(sample[0]) 
plt.title(f"Pred: {class_idx} ({class_names[class_idx]})")
plt.axis('off')
plt.show()

# ==========================================
# 기존 코드의 .h5 부분을 모두 .keras로 바꿉니다.
# ==========================================

# STEP 9: 모델 저장/로드 및 재사용
print("\n--- STEP 9: 모델 저장 및 로드 테스트 ---")
model_res.save('resnet_transfer_cifar10.keras') # 수정됨 (.keras)
print("모델 저장 완료: 'resnet_transfer_cifar10.keras'")

from tensorflow.keras.models import load_model
loaded_model = load_model('resnet_transfer_cifar10.keras') # 수정됨 (.keras)
loaded_pred = loaded_model.predict(sample)
print(f"불러온 모델로 다시 예측한 결과: {np.argmax(loaded_pred)} ({class_names[np.argmax(loaded_pred)]})")


# STEP 10: 다양한 모델 실험 준비 (MobileNetV2)
print("\n--- STEP 10: 다른 모델(MobileNetV2) 로드 ---")
from tensorflow.keras.applications import MobileNetV2

# ResNet50 대신 훨씬 가볍고 모바일에 최적화된 MobileNetV2를 불러옵니다.
base_model_mobile = MobileNetV2(include_top=False, pooling='avg', input_shape=(32,32,3), weights='imagenet')
print(f"MobileNetV2 베이스 모델 로드 성공! (총 파라미터 수: {base_model_mobile.count_params()})")

#===============================================
# 전이학습 파이프라인 조정
# 1차 학습 -> 미세 조정 -> 그래프 출력 -> 예측 결과 출력 -> 모델 저장/로드 테스트-> MobileNetV2 준비

# 미세 조정 결과를 시각화 하였다. 훈련정확도는 80% 부근까지 가파르게 상승하였고, 검증 정확도는 65%까지 완만하게 성장하였다.
# 고양이 사진의 상단을 보면 고양이 이미지를 정확하게 맞춘것을 확인할 수 있다.

# 전이학습과 미세조정 기법을 결합하여 
# 최소한의 연산 자원마능로 최적화된 이미지 분류기를 빠르게 구축하고 
# 서비스 배표용으로 저장하는 엔드 투 엔드 파이프라인을 보여준다. 



