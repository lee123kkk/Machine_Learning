# 0700_2_CIFAR10_Transfer_Learning

import tensorflow as tf
import datetime

# 1️⃣ [데이터 로드 및 전처리]
print("🚀 CIFAR-10 데이터 로드 중...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# MobileNetV2 전용 전처리 (0~1 정규화 대신 -1~1 사이로 값을 변환하여 모델에 최적화시킴)
x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test.astype('float32'))

nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# 2️⃣ [일타 강사 영입: 사전 학습된 모델 로드]
# weights='imagenet': 수백만 장의 이미지를 보고 깨우친 '시각적 규칙'을 그대로 가져옵니다.
# include_top=False: 원래 모델의 마지막 분류기(1000개 분류)는 떼어내고 '눈(특성 추출기)'만 가져옵니다.
print("🧠 사전 학습된 MobileNetV2 모델을 다운로드합니다...")
base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), 
                                               include_top=False, 
                                               weights='imagenet')

# ⭐ 핵심: 가져온 뇌의 기억이 지워지지 않도록 가중치를 얼려버립니다(Freeze). 
# 이렇게 하면 학습해야 할 파라미터가 확 줄어들어 RTX 4060에서 순식간에 학습이 끝납니다.
base_model.trainable = False 

# 3️⃣ [우리의 목적에 맞는 새로운 모델 조립]
model = tf.keras.Sequential([
    # 명시적인 입력층 (그래프 추적을 깔끔하게 만들기 위함)
    tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
    
    # 팁: MobileNetV2는 32x32 이미지보다 큰 이미지를 더 잘 봅니다. 
    # 해상도를 3배(96x96)로 강제로 키워서 넣어주면 정확도가 훨씬 올라갑니다.
    tf.keras.layers.UpSampling2D(size=(3, 3)),
    
    # 떼어온 일타 강사의 뇌(특성 추출기) 부착
    base_model,
    
    # 추출된 수많은 특징들을 평균 내어 1차원으로 압축 (Flatten보다 파라미터가 훨씬 적고 효율적임)
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # 우리가 직접 학습시킬 꼬리 부분 (새로운 분류기)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(nb_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

# 4️⃣ [⭐수동 그래프 추적 (Manual Graph Tracing)]
# Keras 콜백의 버그(Malformed GraphDef)를 피하기 위해, 직접 모델 구조를 녹화해서 텐서보드에 저장합니다.
log_dir = "logs/transfer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

print("\n📹 텐서보드용 모델 그래프 구조를 녹화합니다...")
tf.summary.trace_on(graph=True, profiler=False)
dummy_input = tf.zeros((1, 32, 32, 3)) # 가짜 데이터 1장
_ = model(dummy_input)                 # 모델에 통과시키며 흐름 녹화

with writer.as_default():
    tf.summary.trace_export(name="transfer_learning_graph", step=0)
print("✅ 그래프 녹화 완료!")

# 5️⃣ [콜백 함수 설정]
# 이미 수동으로 그래프를 저장했으므로, write_graph=False로 설정하여 충돌을 막습니다.
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, histogram_freq=1)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_transfer_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1
)

# 6️⃣ [학습 수행]
# 얼어있는 모델은 학습하지 않으므로, 에포크당 학습 속도가 매우 빠릅니다. 10 에포크만 돌려봅니다.
training_epochs = 10 
batch_size = 128

print("\n🚀 전이 학습 시작! (분류기 부분만 학습하므로 매우 빠릅니다)...")
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=training_epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_cb, checkpoint_cb])

# 7️⃣ [최종 평가]
print("\n" + "="*50)
evaluation = model.evaluate(x_test, y_test)
print(f"최종 Loss: {evaluation[0]:.4f}")
print(f"최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")

#=========================================================
# 전이 학습
# MobileNetV2를 활용한 전이 학습
# 원래 학습되어 있는 내용을 바타응로 분류기 부분만 새로 학습

# 전체 파라미터는 242만개이지만 훈련 가능한 파라미터는 새로 추가한 16만개 뿐이다.
# MolieNetV2를 freeze 했기 때문에 에포크당 소요 시간이 7에서 8초 밖에 되지 않는다.
# 정확도 그ㅍ래프가 과적합없이 우상향하고 있다.
# 최종 정확도가 80.22%가 나왔다.