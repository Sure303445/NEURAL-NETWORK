import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE      = r"C:\Users\Welcome\OneDrive\NNDL PROJECT.CAREAI"
DATA_DIR  = os.path.join(BASE, "dataset_split")
MODEL_DIR = os.path.join(BASE, "models")
PLOT_DIR  = os.path.join(BASE, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model_finetuned.h5'))
print("Model loaded!")

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
print("Classes: " + str(class_names))

print("Running predictions...")
preds  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("")
print("="*55)
print("CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
save_path = os.path.join(PLOT_DIR, 'confusion_matrix.png')
plt.savefig(save_path)
plt.show()
print("Confusion matrix saved to: " + save_path)
print("")
print("NEXT STEP: Run python step4_gradcam.py")
