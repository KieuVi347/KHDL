# =====================================================
# TRAIN MOBILENETV2 - SIGN LANGUAGE RECOGNITION
# Thay thế ensemble 4 models cũ bằng 1 model hiện đại
# =====================================================

import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# =====================================================
# CONFIG
# =====================================================

BASE_DIR   = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataSet"
MODELS_DIR  = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_DIR = DATASET_DIR / "trainingData"
TEST_DIR  = DATASET_DIR / "testingData"

IMG_SIZE   = 128       # giữ nguyên như model cũ
BATCH_SIZE = 32
EPOCHS     = 30
NUM_CLASSES = 27       # blank(0) + A-Z = 27 classes

# =====================================================
# DATA GENERATORS
# Chuyển grayscale -> RGB bằng cách lặp 3 kênh
# MobileNetV2 yêu cầu 3-channel input
# =====================================================

def grayscale_to_rgb_preprocess(x):
    """Chuyển ảnh threshold grayscale thành RGB cho MobileNetV2."""
    # x shape: (H, W, 3) — ImageDataGenerator đã load RGB
    # Chuyển về grayscale rồi normalize
    gray = tf.image.rgb_to_grayscale(x)           # (H, W, 1)
    rgb  = tf.repeat(gray, repeats=3, axis=-1)     # (H, W, 3)
    rgb  = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
    return rgb

train_datagen = ImageDataGenerator(
    preprocessing_function=grayscale_to_rgb_preprocess,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,   # Ký hiệu tay phân biệt trái/phải
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=grayscale_to_rgb_preprocess
)

print("Đang load training data...")
train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Đang load testing data...")
test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Lưu class indices để dùng trong app
import json
class_indices = train_generator.class_indices
idx_to_class  = {v: k for k, v in class_indices.items()}

with open(str(MODELS_DIR / "class_indices.json"), "w") as f:
    json.dump(idx_to_class, f, indent=2)

print(f"Classes: {class_indices}")
print(f"Train samples  : {train_generator.samples}")
print(f"Val samples    : {val_generator.samples}")
print(f"Test samples   : {test_generator.samples}")

# =====================================================
# BUILD MODEL - MobileNetV2 + Custom Head
# =====================================================

def build_mobilenet_model(num_classes: int) -> Model:
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base trước, chỉ train custom head
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


model, base_model = build_mobilenet_model(NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# PHASE 1: Train chỉ custom head (base frozen)
# =====================================================

print("\n=== PHASE 1: Training custom head ===")

callbacks_phase1 = [
    ModelCheckpoint(
        str(MODELS_DIR / "mobilenet_best_phase1.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks_phase1,
    verbose=1
)

# =====================================================
# PHASE 2: Fine-tune — mở thêm 30 layers cuối của base
# =====================================================

print("\n=== PHASE 2: Fine-tuning ===")

base_model.trainable = True

# Chỉ train 30 layers cuối
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    ModelCheckpoint(
        str(MODELS_DIR / "mobilenet_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks_phase2,
    verbose=1
)

# =====================================================
# EVALUATE
# =====================================================

print("\n=== EVALUATION ON TEST SET ===")
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss    : {loss:.4f}")

print(f"\nModel đã lưu tại: {MODELS_DIR / 'mobilenet_model.h5'}")
print(f"Class indices  : {MODELS_DIR / 'class_indices.json'}")
