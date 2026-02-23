# 700_5_CIFAR10_ResNet50V2_MixUp

import tensorflow as tf
import datetime

# =========================================================================
# 1️⃣ [데이터 로드 및 전처리] 
# =========================================================================
print("🚀 CIFAR-10 데이터 로드 중...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.resnet_v2.preprocess_input(x_test.astype('float32'))

nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# =========================================================================
# 2️⃣ [스마트 기법 1: MixUp (최신 데이터 증강)]
# =========================================================================
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1 = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2 = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1 / (gamma_1 + gamma_2)

def mix_up(images, labels):
    # ⭐ 오류 수정: 입력된 데이터를 명시적으로 float32로 변환하여 타입 에러 원천 차단
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)

    batch_size = tf.shape(images)[0]
    alphas = sample_beta_distribution(batch_size)
    alphas_image = tf.reshape(alphas, [batch_size, 1, 1, 1])
    alphas_label = tf.reshape(alphas, [batch_size, 1])

    # 배치 내에서 순서 섞기
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # 이미지와 라벨 합성
    mixed_images = images * alphas_image + shuffled_images * (1 - alphas_image)
    mixed_labels = labels * alphas_label + shuffled_labels * (1 - alphas_label)
    return mixed_images, mixed_labels

BATCH_SIZE = 64

# tf.data.Dataset을 활용해 CPU에서 MixUp을 병렬 처리
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
train_ds = train_ds.map(mix_up, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 기본 물리적 증강 (이동, 회전, 줌 등)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name="data_augmentation")

# =========================================================================
# 3️⃣ [모델 조립: ResNet50V2 + UpSampling]
# =========================================================================
print("🧠 초일타 강사 'ResNet50V2'를 모셔옵니다...")
base_model = tf.keras.applications.ResNet50V2(input_shape=(160, 160, 3), 
                                              include_top=False, 
                                              weights='imagenet')
base_model.trainable = False 

inputs = tf.keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.UpSampling2D(size=(5, 5))(x)
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# =========================================================================
# 4️⃣ [수동 그래프 추적 및 텐서보드 설정]
# =========================================================================
log_dir = "logs/mixup/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

print("\n📹 텐서보드용 모델 그래프 구조를 녹화합니다...")
tf.summary.trace_on(graph=True, profiler=False)
_ = model(tf.zeros((1, 32, 32, 3))) 
with writer.as_default():
    tf.summary.trace_export(name="mixup_graph", step=0)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)

# =========================================================================
# 🧠 [스마트 기법 2 & 3: AdamW 옵티마이저 & Label Smoothing]
# =========================================================================
# Label Smoothing (0.1)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# AdamW
optimizer_phase1 = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
optimizer_phase2 = tf.keras.optimizers.AdamW(learning_rate=0.00005, weight_decay=1e-4)

# =========================================================================
# 🎯 [1단계 학습: 분류기 워밍업 (5 Epochs)]
# =========================================================================
print("\n" + "="*50)
print("🚀 [1단계] 분류기 워밍업 시작 (MixUp 적용)...")
model.compile(loss=loss_fn, optimizer=optimizer_phase1, metrics=['accuracy'])

history_phase1 = model.fit(train_ds, 
                           epochs=5,
                           validation_data=val_ds,
                           callbacks=[tensorboard_cb])

# =========================================================================
# 🎯 [2단계 학습: 미세 조정 (Fine-Tuning)]
# =========================================================================
print("\n" + "="*50)
print("🔥 [2단계] 미세 조정 돌입 (ResNet 봉인 해제)...")

base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='ultimate_mixup_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
)
lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
)

model.compile(loss=loss_fn, optimizer=optimizer_phase2, metrics=['accuracy'])
model.summary()

print("\n시간 제약을 맞추기 위해 최대 40 에포크만 진행합니다. MixUp의 힘을 믿어보세요!")
history_phase2 = model.fit(train_ds,
                           epochs=40,
                           initial_epoch=history_phase1.epoch[-1] + 1,
                           validation_data=val_ds,
                           callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb, lr_scheduler_cb])

# =========================================================================
# 5️⃣ [최종 평가]
# =========================================================================
print("\n" + "="*50)
evaluation = model.evaluate(val_ds)
print(f"🎉 최종 Loss: {evaluation[0]:.4f}")
print(f"🏆 최종 실전 정확도(Accuracy): {evaluation[1]*100:.2f}%")

