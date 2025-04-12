import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Path file
chm_path = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Polygon_002_utm_50N_chm.tif"
output_mask_path = chm_path.replace("_chm.tif", "_mask.tif")

# Load CHM
with rasterio.open(chm_path) as src:
    chm = src.read(1)
    profile = src.profile

# Thresholding: pohon kalau tinggi > 2 meter
mask_tree = (chm > 2).astype(np.uint8)

# Update profile
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw'
)
if 'nodata' in profile:
    del profile['nodata']

# Simpan mask
with rasterio.open(output_mask_path, 'w', **profile) as dst:
    dst.write(mask_tree, 1)

print(f"✅ Mask disimpan di: {output_mask_path}")

# Visualisasi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Canopy Height Model")
plt.imshow(chm, cmap="viridis")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Tree Mask (CHM > 2m)")
plt.imshow(mask_tree, cmap="gray")
plt.show()

