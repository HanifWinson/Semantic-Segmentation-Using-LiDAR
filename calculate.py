import cv2
import numpy as np
import rasterio

# Path ke file
chm_path = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon001\\Data\\chm_tiles\\Polygon_001_utm_50S_0_chm.tif"
mask_path = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon001\\Data\\pseudorgb_tiles\\Polygon_001_utm_50S_2_chm.jpg"
# 1. Baca mask dan binarisasi
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 2. Baca CHM
with rasterio.open(chm_path) as src:
    chm = src.read(1)
    chm[chm < 0] = 0  # bersihin noise
    pixel_size = src.res[0]
    transform = src.transform
    chm_shape = chm.shape

# Resize mask ke ukuran CHM
mask_resized = cv2.resize(mask, (chm_shape[1], chm_shape[0]), interpolation=cv2.INTER_NEAREST)
mask_bin = (mask_resized > 127).astype(np.uint8)

# 3. Hitung metrik
tree_pixels = np.sum(mask_bin)
area = tree_pixels * (pixel_size ** 2)

tree_heights = chm[mask_bin == 1]
avg_height = np.mean(tree_heights)
max_height = np.max(tree_heights)

# Estimasi biomassa dan karbon
volume = area * avg_height  # m³
wood_density = 0.6  # ton/m³
biomass = volume * wood_density
carbon = biomass * 0.47
co2 = carbon * 3.67

# 4. Output hasil
print("===== Tree Carbon Estimation =====")
print(f"Luas Tajuk:       {area:.2f} m²")
print(f"Tinggi Rata-rata: {avg_height:.2f} m")
print(f"Volume Kanopi:    {volume:.2f} m³")
print(f"Biomassa:         {biomass:.2f} ton")
print(f"Kandungan Karbon: {carbon:.2f} ton C")
print(f"Setara CO₂:       {co2:.2f} ton CO₂")
print("===================================")