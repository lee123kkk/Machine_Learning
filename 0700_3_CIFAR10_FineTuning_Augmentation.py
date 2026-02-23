# 0700_3_CIFAR10_FineTuning_Augmentation

import tensorflow as tf
import datetime

# 1️⃣ [데이터 로드 및 전처리]
print("🚀 CIFAR-10 데이터 로드 중...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test.astype('float32'))

nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# 2️⃣ [전략 B: 데이터 증강 (Data Augmentation)]
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# 3️⃣ [일타 강사 영입 및 모델 조립]
print("🧠 사전 학습된 MobileNetV2 모델을 다운로드합니다...")
base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), 
                                               include_top=False, 
                                               weights='imagenet')
# 1단계 학습을 위해 일단 꽁꽁 얼려둡니다.
base_model.trainable = False 

# 이번에는 조금 더 세밀한 컨트롤을 위해 함수형 API(Functional API) 방식으로 조립합니다.
inputs = tf.keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)

# ⭐ 중요: training=False로 설정하여 base_model 내부의 배치 정규화(BatchNorm) 층이 
# 미세 조정(Fine-Tuning) 중에도 망가지지 않도록 보호합니다.
x = base_model(x, training=False) 

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# [수동 그래프 추적 (Manual Graph Tracing)]
log_dir = "logs/finetuning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=False)
_ = model(tf.zeros((1, 32, 32, 3))) 
with writer.as_default():
    tf.summary.trace_export(name="fine_tuning_graph", step=0)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)

# =========================================================================
# 🎯 [1단계 학습: 워밍업 (분류기만 학습)]
# =========================================================================
print("\n" + "="*50)
print("🚀 [1단계] 새로운 분류기 워밍업 학습을 시작합니다 (Base Model Frozen)...")
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 기본 학습률
              metrics=['accuracy'])

history_phase1 = model.fit(x_train, y_train,
                           batch_size=128,
                           epochs=5, # 워밍업은 5번만 짧게 진행
                           validation_data=(x_test, y_test),
                           callbacks=[tensorboard_cb])

# =========================================================================
# 🎯 [2단계 학습: 미세 조정 (Fine-Tuning)]
# =========================================================================
print("\n" + "="*50)
print("🔥 [2단계] 전략 A 적용: 미세 조정(Fine-Tuning)을 시작합니다...")

# 일타 강사의 뇌(전체 층)의 얼음을 모두 녹입니다.
base_model.trainable = True

# 하지만 너무 기초적인 지식(앞쪽 층)까지 건드리면 역효과가 나므로, 
# 100번째 층 이전은 다시 얼려두고, 깊고 복잡한 특징을 잡는 100번째 층 이후만 학습시킵니다.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# ⭐ 핵심: 이미 똑똑한 상태이므로, 기존 지식이 파괴되지 않게 보폭(학습률)을 1/100 수준으로 아주 작게 줄입니다!
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # 1e-5 (매우 낮음)
              metrics=['accuracy'])
model.summary()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_finetuned_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1
)

# 1단계에서 5번 학습했으므로, 이어서 추가로 10번(총 15 에포크) 더 학습시킵니다.
history_phase2 = model.fit(x_train, y_train,
                           batch_size=128,
                           epochs=15, 
                           initial_epoch=history_phase1.epoch[-1] + 1, # 5번부터 이어서 시작
                           validation_data=(x_test, y_test),
                           callbacks=[tensorboard_cb, checkpoint_cb])

# =========================================================================
# 4️⃣ [최종 평가]
print("\n" + "="*50)
evaluation = model.evaluate(x_test, y_test)
print(f"최종 Loss: {evaluation[0]:.4f}")
print(f"최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")



#===========================================================================
# 전이 학습 미세 조정 + 데이터 증강 결합
# 1단계: 새로 붙인 데이터 증강 필터와 분류기만 먼저 CIFAR-10에 적응시킨다.
# 2단계: 분류기가 어느 정도 똑똑해지면 뇌의 깊은 층을 살짝 녹이고 낮은 학습률롷 학습시킨다.

# 에포크당 14~17초 정도의 속도로 학습이 진행되었다.
# 5번쨰 에포크에서 검증 정확도가 72.54%까지 올라갔다.

# 미세 조정을 통해서 훈련 가능한 파라미터가 16만개에서 202만개로 12배이상 증가하였다.
# 에포크당 소요 시간이 26에서 33초로 늘어났다.
# 최종 테스트 결과 82.16%로 올랐다.

# 학습 횟수가 부족하다. 증강 필터가 켜져 있드면 최소 30에서 50번의 에포크가 필요하다.
# MobileNetV2는 경령화 모델이므로 한계가 명확하다.
