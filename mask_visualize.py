import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Folder yang berisi file mask
mask_folder = "C:\\Users\\joy\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon001\\Data\\mask_tiles"

# Ambil semua file mask (nama file berakhiran _mask.tif)
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('_mask.tif')]

# Tentukan ukuran grid visualisasi
cols = 3
rows = (len(mask_files) + cols - 1) // cols  # hitung jumlah baris yang dibutuhkan

plt.figure(figsize=(15, 5 * rows))

for i, mask_name in enumerate(mask_files):
    mask_path = os.path.join(mask_folder, mask_name)
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(mask, cmap='gray')
    plt.title(mask_name)
    plt.axis('off')

plt.tight_layout()
plt.show()
