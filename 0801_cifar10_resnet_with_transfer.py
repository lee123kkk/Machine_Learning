#cifar10_resnet_with_transfer.ipynb
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import datasets, Input

# 1. 데이터 불러오기: CIFAR-10 (10가지 클래스를 가진 32x32 크기의 이미지 데이터셋)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. 정답(Label) 데이터를 원-핫 인코딩(One-Hot Encoding)으로 변환
# 예: [6] -> [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.] 형태로 변환 (분류 문제의 필수 작업)
input_y = tf.keras.utils.to_categorical(train_labels, 10)
test_y = tf.keras.utils.to_categorical(test_labels, 10)

# 3. 사전 학습된 ResNet50 모델 불러오기 (전이 학습의 핵심)
# include_top=False: 원래 ResNet의 분류기(1000개 분류용)는 빼고 특징 추출기만 가져옴
# weights='imagenet': ImageNet 데이터로 학습된 가중치를 그대로 사용
base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, pooling='avg', input_shape=(32,32,3), weights='imagenet'
)

# ★ 4. 가중치 동결 (Freezing)
# 가져온 ResNet50의 지식(가중치)이 파괴되지 않도록 학습되지 않게 잠금 처리
base_model.trainable = False

# 5. 새로운 분류기 모델 만들기
inputs = Input(shape=(32,32,3))
# ResNet50에 맞는 입력 형태로 데이터 전처리
x = tf.keras.applications.resnet50.preprocess_input(inputs) 
x = base_model(x, training=False) # 잠가둔 ResNet50 통과 (특징 추출)
x = Flatten()(x)                  # 1차원 배열로 펼치기                                                
# 우리가 원하는 10개 클래스(CIFAR-10)를 분류하기 위한 최종 출력층(Dense) 추가
outputs = Dense(10, activation='softmax')(x)  

# 입력과 출력을 연결하여 최종 모델 완성
model_res = tf.keras.Model(inputs, outputs)  

# 6. 모델 컴파일 (학습 방법 설정)
model_res.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 7. 모델 학습 실행
model_res.fit(train_images, input_y, epochs=10, validation_data=(test_images, test_y), batch_size=256)
#================================================================
# 전이 학습
# 미리 학습된 모델(ResNet50)을 가져와서 새로운 문제에 맞게 뒷부분만 살짝 고쳐서 재활용한다.
# 총 파라미터 개수는 2360만 개이고 학습 가능한 파라미커는 20490개이다. 나머지는 학습시키지 않고, 그대로 사용해서 학습 속도를 향상시킨다.
# 첫번째 데이터의 사진을 보면 상단의 [6]은 이 이미지의 정답을 의미한다. 인공지능 모델에게 이 사진의 정답을 학습시키는 역할을 한다.
# 에포크당 2~3초로 빠르게 작업이 수행되었다.
# 10번의 에포크가 끝난 후에 훈련 정확도는 72.2%, 테스트 정확도는 65.6%를 기록했다.

# 방대한 데이터로 사전 학습된 거대 모델을 활용하면 적은 연산량과 짧은 시간만으로 새로운 문제에 준수한 성능을 얻을 수 있다.

