import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

BASE      = r"C:\Users\Welcome\OneDrive\NNDL PROJECT.CAREAI"
DATA_DIR  = os.path.join(BASE, "dataset_split")
MODEL_DIR = os.path.join(BASE, "models")
PLOT_DIR  = os.path.join(BASE, "outputs", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16

print("Loading saved model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.h5'))
print("Model loaded! Starting Phase 2...")

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

total = 12297 + 96 + 463
class_weight = {
    0: total / (3 * 463),
    1: total / (3 * 12297),
    2: total / (3 * 96)
}

print("Unfreezing layers...")
for layer in model.layers:
    layer.trainable = True

base = model.layers[1]
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model_finetuned.h5'),
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

print("Phase 2 Fine tuning started...")
h2 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks
)

print("Fine tuning complete!")
print("Model saved to models/best_model_finetuned.h5")

acc   = h2.history['accuracy']
val   = h2.history['val_accuracy']
loss  = h2.history['loss']
vloss = h2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(acc,  label='Train Accuracy')
ax1.plot(val,  label='Val Accuracy')
ax1.set_title('Fine Tuning Accuracy')
ax1.legend()

ax2.plot(loss,  label='Train Loss')
ax2.plot(vloss, label='Val Loss')
ax2.set_title('Fine Tuning Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'finetuning_curves.png'))
plt.show()
print("Done! Next run: python step3_evaluate.py")
