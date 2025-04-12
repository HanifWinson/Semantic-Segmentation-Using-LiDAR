import os
import laspy
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

# Path utama
laz_dir = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Data"
chm_path = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Polygon_002_utm_50N_chm.tif"
mask_path = "C:\\Users\\joy\\Documents\\Python\\LiDAR Kalimantan Databases\\Polygon002\\Polygon_002_utm_50N_mask.tif"  # Opsional

# Output folder
output_chm_dir = os.path.join(laz_dir, "chm_tiles")
output_mask_dir = os.path.join(laz_dir, "mask_tiles")
os.makedirs(output_chm_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Loop semua file LAZ
for filename in os.listdir(laz_dir):
    if filename.endswith(".laz"):
        laz_path = os.path.join(laz_dir, filename)
        print(f"Processing {filename}...")

        # 1. Ambil bounding box dari file .laz
        las = laspy.read(laz_path)
        min_x, max_x = las.x.min(), las.x.max()
        min_y, max_y = las.y.min(), las.y.max()

        # 2. Buka CHM dan potong window
        with rasterio.open(chm_path) as chm_src:
            chm_window = from_bounds(min_x, min_y, max_x, max_y, chm_src.transform)
            chm_data = chm_src.read(1, window=chm_window)
            chm_transform = chm_src.window_transform(chm_window)

            # Simpan potongan CHM
            chm_tile_path = os.path.join(output_chm_dir, filename.replace(".laz", "_chm.tif"))
            with rasterio.open(
                chm_tile_path,
                "w",
                driver="GTiff",
                height=chm_data.shape[0],
                width=chm_data.shape[1],
                count=1,
                dtype=chm_data.dtype,
                crs=chm_src.crs,
                transform=chm_transform,
            ) as dst:
                dst.write(chm_data, 1)

        # 3. (Opsional) Potong mask juga
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as mask_src:
                mask_window = from_bounds(min_x, min_y, max_x, max_y, mask_src.transform)
                mask_data = mask_src.read(1, window=mask_window)
                mask_transform = mask_src.window_transform(mask_window)

                # Simpan potongan mask
                mask_tile_path = os.path.join(output_mask_dir, filename.replace(".laz", "_mask.tif"))
                with rasterio.open(
                    mask_tile_path,
                    "w",
                    driver="GTiff",
                    height=mask_data.shape[0],
                    width=mask_data.shape[1],
                    count=1,
                    dtype=mask_data.dtype,
                    crs=mask_src.crs,
                    transform=mask_transform,
                ) as dst:
                    dst.write(mask_data, 1)

print("✅ CHM dan mask dipotong sesuai bounding box dari file LAZ.")
