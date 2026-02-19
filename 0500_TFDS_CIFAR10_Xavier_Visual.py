#TFDS_CIFAR10_Xavier_Visual

'''
Xavier초기화를 유지하면서 데이터 셋을 TFDS 라이브러리를 사용해 교체
CIRAR-10 데이터 사용 (32X32 컬러, 사물 10종)
'''

import tensorflow as tf
import tensorflow_datasets as tfds  # TFDS 라이브러리
import numpy as np
import matplotlib.pyplot as plt     # 시각화(이미지 그리기)용

# 1. [시드 고정]
tf.random.set_seed(777)

# 2. [하이퍼파라미터 설정]
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# CIFAR-10의 정답 이름표 (0~9)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 3. [데이터 로드 - TFDS 활용]
# batch_size=-1로 설정하면 데이터셋 전체를 한 번에 Numpy 형태로 가져옵니다.
print("📦 TFDS에서 CIFAR-10 데이터를 다운로드 중입니다...")
ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], 
                              batch_size=-1, as_supervised=True)

# TFDS는 텐서(Tensor) 형태로 데이터를 줍니다. 이를 우리가 익숙한 Numpy로 바꿉니다.
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)

print(f"학습 데이터 형태: {x_train.shape}") # (50000, 32, 32, 3) -> 컬러 이미지라 3채널!

# 4. [데이터 전처리]
# (1) 2차원(32x32x3) -> 1차원(3072) 평탄화
# 컬러 이미지는 픽셀 수가 훨씬 많습니다. (32 * 32 * 3 = 3072)
input_shape = 32 * 32 * 3
x_train_flat = x_train.reshape(x_train.shape[0], input_shape)
x_test_flat = x_test.reshape(x_test.shape[0], input_shape)

# (2) 정규화 (Normalization) - 0~255 값을 0~1로
x_train_flat = x_train_flat / 255.0
x_test_flat = x_test_flat / 255.0

# (3) 원-핫 인코딩
y_train_onehot = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, nb_classes)

# 5. [모델 구성] Xavier(Glorot) 초기화 적용
tf.model = tf.keras.Sequential()

# 입력이 3072개로 늘어났으므로, 은닉층 뉴런도 512개로 늘려줍니다.
tf.model.add(tf.keras.layers.Dense(input_dim=input_shape, units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=512, 
                                   kernel_initializer='glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=nb_classes, 
                                   kernel_initializer='glorot_normal', activation='softmax'))

# 6. [컴파일 및 학습]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])

print("\n🚀 학습을 시작합니다 (컬러 이미지라 시간이 조금 더 걸립니다)...")
history = tf.model.fit(x_train_flat, y_train_onehot, batch_size=batch_size, epochs=training_epochs)

# 7. [결과 시각화] 실제 이미지와 예측 결과 보여주기
print("\n🎨 결과 이미지를 생성 중입니다...")

# 테스트 데이터 중 랜덤으로 15개 뽑기
indices = np.random.choice(len(x_test), 15, replace=False)
predictions = tf.model.predict(x_test_flat[indices])

plt.figure(figsize=(15, 6)) # 창 크기 조절

for i, idx in enumerate(indices):
    plt.subplot(3, 5, i + 1) # 3줄 5칸으로 나누기
    
    # 평탄화된 데이터를 다시 32x32x3 이미지 형태로 되돌려야 그림이 그려집니다.
    img = x_test[idx] 
    plt.imshow(img)
    
    # 정답과 예측값 비교
    actual_idx = y_test[idx] if len(y_test.shape) == 1 else np.argmax(y_test[idx]) # 원본 라벨
    pred_idx = np.argmax(predictions[i])
    
    label_actual = class_names[actual_idx]
    label_pred = class_names[pred_idx]
    
    # 맞으면 초록색, 틀리면 빨간색 글씨
    color = 'green' if actual_idx == pred_idx else 'red'
    
    plt.title(f"A: {label_actual}\nP: {label_pred}", color=color)
    plt.axis('off') # 축 없애기

plt.tight_layout()
plt.show()

# 8. [최종 정확도 평가]
score = tf.model.evaluate(x_test_flat, y_test_onehot, verbose=0)
print(f"\n최종 정확도: {score[1]*100:.2f}%")
#==========================================================
# NN_xavior를 TFDS 데이터에 적용시킨 결과 
# MNIST는 (28, 28, 1)이었지만 CIFAR-10은 (32, 32, 3)으로 데어터가 4배 정도 늘어남
# 최종 정확도: 46.19% 학습을 하긴 했지만, 단순 신경망으로는 컬러 이미지를 분석하는데 한계가 있다.
# 학습 점수와 실전 점수에 46.2% 오차가 발생해서 과적합의 조짐이 보인다.
# 개와 고양이를 혼동하거나 트럭과 승용차관련 문제가 발생했음.
# 평탄화 과정에서 위치 정보의 파괴와 배경색에 의존하는 문제가 발생했다

# TFDS를 이용해 더 복잡한 컬리 이미지에 Xavior 초기화를 적용시켜 학습시켰지만,
# 아직 충분한 성능을 발휘하지 못한다.
