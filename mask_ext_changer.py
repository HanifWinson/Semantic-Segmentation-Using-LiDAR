import os
import rasterio
import numpy as np
import cv2

tif_folder = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Data\\chm_tiles"
output_folder = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Data\\chm_png"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(tif_folder):
    if filename.endswith(".tif"):
        tif_path = os.path.join(tif_folder, filename)
        png_path = os.path.join(output_folder, filename.replace(".tif", ".png"))

        try:
            with rasterio.open(tif_path) as src:
                chm = src.read(1)
                chm = np.nan_to_num(chm)  # kalau ada NaN
                mask = (chm > 2).astype(np.uint8) * 255  # Thresholding jika mau
                cv2.imwrite(png_path, mask)
                print(f"Saved: {png_path}")
        except Exception as e:
            print(f"Gagal konversi {filename}: {e}")
