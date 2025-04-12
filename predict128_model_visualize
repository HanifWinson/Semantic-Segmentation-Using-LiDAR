import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== Konfigurasi ==========
model_path = "tree_segmentation_from_chm.h5" 
image_dir = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon003\\Data\\pseudorgb_tiles"
mask_dir = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon003\\Data\\chm_png"
input_size = (128, 128)

# ========== Load Model ==========
model = tf.keras.models.load_model(model_path, compile=False)

# ========== Preprocessing ==========
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    resized = cv2.resize(img, input_size)
    return resized / 255.0, original_size, img

def load_mask(mask_path, target_size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")
    return cv2.resize(mask, target_size)

# ========== Ambil Gambar Acak dan Prediksi ==========
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
np.random.shuffle(image_files)

for file in image_files[:4]:  # tampilkan 4 contoh
    img_path = os.path.join(image_dir, file)
    mask_path = os.path.join(mask_dir, file.replace(".jpg", ".png").replace(".jpeg", ".png"))

    try:
        img_resized, original_size, img_original = preprocess_image(img_path)
        input_tensor = np.expand_dims(img_resized, axis=0)

        pred = model.predict(input_tensor)[0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask.squeeze(), original_size)

        true_mask = load_mask(mask_path, original_size)

        # ========== Plot ==========
        plt.figure(figsize=(12, 4))
        plt.suptitle(file, fontsize=12)

        plt.subplot(1, 3, 1)
        plt.imshow(img_original)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
