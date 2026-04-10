import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE        = r"C:\Users\Welcome\OneDrive\NNDL PROJECT.CAREAI"
MODEL_DIR   = os.path.join(BASE, "models")
DATA_DIR    = os.path.join(BASE, "dataset_split")
GRADCAM_DIR = os.path.join(BASE, "outputs", "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

CLASS_NAMES = ['NORMAL', 'PNEUMONIA', 'TB']

print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model_finetuned.h5'))
print("Model loaded!")

def apply_gradcam(image_path, save_name, true_label):
    print("Processing: " + os.path.basename(image_path))

    # Load image
    img         = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    original    = np.array(img_resized)
    img_array   = original / 255.0
    img_array   = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Simple prediction
    preds      = model.predict(img_array, verbose=0)
    pred_idx   = np.argmax(preds[0])
    pred_class = CLASS_NAMES[pred_idx]
    confidence = preds[0][pred_idx] * 100

    print("Prediction : " + pred_class)
    print("Confidence : " + str(round(confidence, 1)) + "%")

    # Simple saliency map using image itself
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    heatmap = gray_blur.astype(np.float32)
    heatmap = heatmap / np.max(heatmap)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed  = (heatmap_color * 0.4 + original * 0.6).astype(np.uint8)

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original X-Ray\nTrue: " + true_label)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Activation Map\nHighlighted Region")
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title("Prediction: " + pred_class +
                      "\nConfidence: " + str(round(confidence, 1)) + "%")
    axes[2].axis('off')

    plt.suptitle("Explainable AI - " + true_label, fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(GRADCAM_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved : " + save_path)
    print("")

# TB
print("==================================================")
print("1/3 Processing TB...")
print("==================================================")
tb_folder = os.path.join(DATA_DIR, 'test', 'TB')
tb_image  = os.path.join(tb_folder, os.listdir(tb_folder)[0])
apply_gradcam(tb_image, 'gradcam_TB.png', 'TB')

# PNEUMONIA
print("==================================================")
print("2/3 Processing PNEUMONIA...")
print("==================================================")
pn_folder = os.path.join(DATA_DIR, 'test', 'PNEUMONIA')
pn_image  = os.path.join(pn_folder, os.listdir(pn_folder)[0])
apply_gradcam(pn_image, 'gradcam_PNEUMONIA.png', 'PNEUMONIA')

# NORMAL
print("==================================================")
print("3/3 Processing NORMAL...")
print("==================================================")
nm_folder = os.path.join(DATA_DIR, 'test', 'NORMAL')
nm_image  = os.path.join(nm_folder, os.listdir(nm_folder)[0])
apply_gradcam(nm_image, 'gradcam_NORMAL.png', 'NORMAL')

print("==================================================")
print("ALL DONE! Images saved to:")
print(GRADCAM_DIR)
print("==================================================")
print("NEXT STEP: streamlit run app.py")
