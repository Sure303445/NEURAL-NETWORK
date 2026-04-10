import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ============================================================
BASE      = r"C:\Users\Welcome\OneDrive\NNDL PROJECT.CAREAI"
DATA_DIR  = os.path.join(BASE, "dataset_split")
MODEL_DIR = os.path.join(BASE, "models")
PLOT_DIR  = os.path.join(BASE, "outputs", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)
# ============================================================

# ── Step 1: Remove corrupted images first ─────────────────
print("Checking and removing corrupted images...")
removed = 0
for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception:
                os.remove(fpath)
                removed += 1
print("Removed " + str(removed) + " corrupted images")
print("")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
NUM_CLASSES = 3

print("Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Classes: " + str(train_gen.class_indices))

# Handle class imbalance
total = 12297 + 96 + 463
class_weight = {
    0: total / (3 * 463),
    1: total / (3 * 12297),
    2: total / (3 * 96)
}
print("Class weights set!")

# ── Build Model ────────────────────────────────────────────
print("Building model...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = layers.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = Model(inputs, outputs)

model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model ready!")

callbacks = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ── Phase 1 Training ───────────────────────────────────────
print("")
print("Phase 1: Training classifier head...")
h1 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks
)

# ── Phase 2 Fine tuning ────────────────────────────────────
print("")
print("Phase 2: Fine tuning...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

h2 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks
)

print("")
print("Training complete!")
print("Model saved to: " + os.path.join(MODEL_DIR, 'best_model.h5'))

# ── Plot curves ────────────────────────────────────────────
acc   = h1.history['accuracy']     + h2.history['accuracy']
val   = h1.history['val_accuracy'] + h2.history['val_accuracy']
loss  = h1.history['loss']         + h2.history['loss']
vloss = h1.history['val_loss']     + h2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(acc,  label='Train Accuracy')
ax1.plot(val,  label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(loss,  label='Train Loss')
ax2.plot(vloss, label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'training_curves.png')
plt.savefig(plot_path)
plt.show()
print("Plot saved!")
print("")
print("NEXT STEP: Run python step3_evaluate.py")
