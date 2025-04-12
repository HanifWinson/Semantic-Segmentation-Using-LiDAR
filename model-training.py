import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== Konfigurasi ==========
model_path = "tree_segmentation_from_chm.h5"
image_dirs = [
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon001\\Data\\pseudorgb_tiles",
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Data\\pseudorgb_tiles",
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon003\\Data\\pseudorgb_tiles"
]
mask_dirs = [
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon001\\Data\\chm_png",
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Data\\chm_png",
    "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon003\\Data\\chm_png"
]
input_size = (128, 128)

# ========== Load Model ==========
model = tf.keras.models.load_model(model_path, compile=False)

# ========== Fungsi Preprocessing ==========
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

# ========== Kumpulkan Semua Gambar dari Banyak Folder ==========
def collect_files_recursively_from_dirs(dir_list, valid_exts=('.jpg', '.jpeg', '.png')):
    all_files = []
    for root_dir in dir_list:
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(valid_exts):
                    full_path = os.path.join(dirpath, f)
                    all_files.append(full_path)
    return all_files

image_files = collect_files_recursively_from_dirs(image_dirs)
np.random.shuffle(image_files)

# ========== Prediksi dan Visualisasi ==========
for img_path in image_files[:4]:
    filename = os.path.basename(img_path)
    mask_path = None

    # Cari mask di semua direktori mask_dirs
    for mask_root in mask_dirs:
        for dirpath, _, filenames in os.walk(mask_root):
            target_mask_name = filename.replace(".jpg", ".png").replace(".jpeg", ".png")
            if target_mask_name in filenames:
                mask_path = os.path.join(dirpath, target_mask_name)
                break
        if mask_path: break

    if mask_path is None:
        print(f"Mask untuk {filename} tidak ditemukan.")
        continue

    try:
        img_resized, original_size, img_original = preprocess_image(img_path)
        input_tensor = np.expand_dims(img_resized, axis=0)

        pred = model.predict(input_tensor)[0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask.squeeze(), original_size)

        true_mask = load_mask(mask_path, original_size)

        # ========== Plot ==========
        plt.figure(figsize=(12, 4))
        plt.suptitle(filename, fontsize=12)

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
