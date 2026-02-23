# 700_4_CIFAR10_ResNet50V2_Ultimate

import tensorflow as tf
import datetime

# =========================================================================
# 1️⃣ [데이터 로드 및 전처리] - ResNet50V2 전용 전처리
# =========================================================================
print("🚀 CIFAR-10 데이터 로드 중...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# ResNet50V2 모델이 좋아하는 방식(-1 ~ 1 범위)으로 픽셀 값을 정규화합니다.
x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.resnet_v2.preprocess_input(x_test.astype('float32'))

nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# =========================================================================
# 2️⃣ [스마트 기법 1: 강화된 데이터 증강 (Advanced Augmentation)]
# =========================================================================
# 기존 회전, 줌에 더해 상하좌우 이동(Translation)까지 추가하여 모델을 혹독하게 훈련시킵니다.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name="data_augmentation")

# =========================================================================
# 3️⃣ [전략 D & C: 초일타 강사 영입 및 해상도 대폭 확대]
# =========================================================================
print("🧠 초일타 강사 'ResNet50V2'를 모셔옵니다 (파라미터 약 2,300만 개)...")
base_model = tf.keras.applications.ResNet50V2(input_shape=(160, 160, 3), # 해상도를 160x160으로 고정
                                              include_top=False, 
                                              weights='imagenet')
base_model.trainable = False # 1단계 워밍업을 위해 얼려둠

inputs = tf.keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)

# ⭐ 전략 C: 32x32 이미지를 가로세로 5배씩 늘려 160x160으로 뻥튀기합니다! (UpSampling)
# 이미지의 여백이 많아져서 ResNet이 특징을 훨씬 정밀하게 잡아냅니다.
x = tf.keras.layers.UpSampling2D(size=(5, 5))(x)

# 뼈대 모델 통과 (배치 정규화 파괴 방지를 위해 training=False)
x = base_model(x, training=False)

# 새로운 분류기 부착 (용량이 큰 모델이므로 은닉층 뉴런 수도 256개로 늘림)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# =========================================================================
# 4️⃣ [수동 그래프 추적 및 텐서보드 설정]
# =========================================================================
log_dir = "logs/ultimate/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

print("\n📹 텐서보드용 모델 그래프 구조를 녹화합니다...")
tf.summary.trace_on(graph=True, profiler=False)
_ = model(tf.zeros((1, 32, 32, 3))) # 가짜 데이터로 흐름 생성
with writer.as_default():
    tf.summary.trace_export(name="ultimate_graph", step=0)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)

# =========================================================================
# 🎯 [1단계 학습: 분류기 워밍업 (5 Epochs)]
# =========================================================================
print("\n" + "="*50)
print("🚀 [1단계] 분류기 워밍업을 시작합니다 (빠르게 진행됩니다)...")
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# RTX 4060의 VRAM(8GB)을 고려하여 Batch Size를 64로 설정합니다. (메모리 초과 시 32로 줄이세요)
BATCH_SIZE = 64 

history_phase1 = model.fit(x_train, y_train,
                           batch_size=BATCH_SIZE,
                           epochs=5,
                           validation_data=(x_test, y_test),
                           callbacks=[tensorboard_cb])

# =========================================================================
# 🎯 [2단계 학습: 미세 조정 및 스마트 학습 스케줄러 적용]
# =========================================================================
print("\n" + "="*50)
print("🔥 [2단계] 일타 강사의 봉인을 해제하고 극한의 훈련에 돌입합니다...")

base_model.trainable = True
# ResNet50V2의 깊은 층(약 100번째 층 이후)만 녹여서 학습시킵니다.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# =========================================================================
# 🧠 [스마트 기법 2 & 3: 콜백(Callbacks) 설정]
# =========================================================================
# 1. ModelCheckpoint: 가장 최고 점수를 기록한 순간의 가중치만 저장
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='ultimate_resnet_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1
)

# 2. EarlyStopping: 10번 연속 갱신이 없으면 무의미한 훈련으로 판단하고 자동 종료
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
)

# 3. ⭐ ReduceLROnPlateau (동적 학습률 스케줄링): 
# 학습이 정체기(Plateau)에 빠지면 보폭(Learning Rate)을 절반(0.5)으로 확 줄여서 
# 마치 현미경으로 보듯 미세하게 튜닝합니다. 이게 90% 돌파의 핵심 키입니다.
lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), # Phase 1보다 훨씬 작게 시작
              metrics=['accuracy'])
model.summary()

# 훈련 시작 (시간이 오래 걸리니 커피 한 잔 드시고 오셔도 좋습니다!)
print("\n시간이 꽤 소요됩니다. 텐서보드를 켜두고 실시간으로 그래프가 오르는 것을 감상하세요!")
history_phase2 = model.fit(x_train, y_train,
                           batch_size=BATCH_SIZE,
                           epochs=50, # 최대 50번 추가 진행 (총 55회)
                           initial_epoch=history_phase1.epoch[-1] + 1,
                           validation_data=(x_test, y_test),
                           callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb, lr_scheduler_cb])

# =========================================================================
# 4️⃣ [최종 평가]
# =========================================================================
print("\n" + "="*50)
evaluation = model.evaluate(x_test, y_test)
print(f"🎉 최종 Loss: {evaluation[0]:.4f}")
print(f"🏆 최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")


