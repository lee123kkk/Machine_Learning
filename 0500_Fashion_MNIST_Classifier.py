# Fashion_MNIST_Classifier

# 티셔츠, 운동화, 등등 패션 아이템 분류

import numpy as np
import random
import tensorflow as tf

# 1. [시드 고정] 재현성을 위해 랜덤 시드 고정
random.seed(777)
tf.random.set_seed(777)

# 2. [하이퍼파라미터 튜닝]
# 학습률을 낮춰서(0.0005) 더 정밀하게 학습
learning_rate = 0.0005  
batch_size = 100
training_epochs = 15
nb_classes = 10
# 은닉층의 뉴런 수를 512로 늘려서 복잡한 패턴 수용 능력 확대
hidden_units = 512      

# Fashion MNIST 클래스 이름 (출력 확인용)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. [데이터 로드] Fashion MNIST 데이터셋 사용
# 인터넷에서 자동으로 다운로드 받습니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 4. [데이터 전처리]
# (1) 2차원(28x28) -> 1차원(784) 평탄화
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# (2) 정규화 (Normalization) - [중요]
# 지난번 예제에서 초기 loss가 폭발했던 것을 방지하기 위해 0~1 사이 값으로 변환
x_train = x_train / 255.0
x_test = x_test / 255.0

# (3) 원-핫 인코딩 (One-hot Encoding)
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# 5. [모델 구성] 심층 신경망 (Deep & Wide)
tf.model = tf.keras.Sequential()

# 은닉층 1: 뉴런 512개 (용량 증가)
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=hidden_units, activation='relu'))

# 은닉층 2: 뉴런 512개 (깊이 유지)
tf.model.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))

# 출력층: 10개 분류
tf.model.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))

# 6. [컴파일]
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])
tf.model.summary()

# 7. [학습 수행]
print("👗 패션 아이템 분류 학습을 시작합니다...")
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# 8. [랜덤 예측 테스트]
y_predicted = tf.model.predict(x_test)
print("\n" + "="*50)
print("🔍 랜덤 예측 결과 확인")
print("="*50)

for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    
    actual_label = np.argmax(y_test[random_index])
    predicted_label = np.argmax(y_predicted[random_index])
    
    # 결과 출력 (숫자 대신 옷 이름으로 출력)
    print(f"Index: {random_index:<5} | "
          f"정답: {class_names[actual_label]:<12} vs "
          f"예측: {class_names[predicted_label]:<12} -> "
          f"{'✅ 정답' if actual_label == predicted_label else '❌ 오답'}")

# 9. [최종 평가]
evaluation = tf.model.evaluate(x_test, y_test, verbose=0)
print("="*50)
print(f"최종 Loss: {evaluation[0]:.4f}")
print(f"최종 Accuracy: {evaluation[1]*100:.2f}%")
#===================================================================
# 단순한 손글씨에서 패션 아이템으로 대상을 확장
# 하이퍼 파라미터 튜닝: unit의 개수(뉴런 수) 2배 증가, 학습률 절반으로 감소

# 첫번째 에포크의 loss가 0.4904로 시작 -> 데이터 전처리가 잘 이루어졌다.
# 마지막 에포크에서 학습 정확도가 94.64%까지 도달
# 실전 테스트 점수 87.2%로 학습 점수와의 오차가 발생 (과적합 가능성)

# 복잡한 데이터를 상태로 준수한 성적을 거두었으나 학습 데이터와의 격차를 줄여야한다.
