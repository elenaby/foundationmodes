import os
import openslide
import numpy as np
import math
from histoplus.extract import extract
from histoplus.helpers.segmentor import CellViTSegmentor
from openslide.deepzoom import DeepZoomGenerator

MPP = 0.5
INFERENCE_IMAGE_SIZE = 784

slide = openslide.open_slide("/home/..")


def generate_coordinates_in_circle(center_x, center_y, radius, num_points, tile_size=224):
    """
    Generate tile coordinates within a circular area at the highest DeepZoom level.
    """
    # Convert center and radius from pixels to tiles at highest level
    center_x_tiles = center_x // tile_size
    center_y_tiles = center_y // tile_size
    radius_tiles = radius // tile_size
    
    coords = []
    
    for i in range(num_points):
        angle = 2 * math.pi * np.random.random()
        distance = radius_tiles * math.sqrt(np.random.random())
        
        x = int(center_x_tiles + distance * math.cos(angle))
        y = int(center_y_tiles + distance * math.sin(angle))
        
        coords.append([x, y])
    
    return np.array(coords)

# TMA-41 parameters at level 0 (pixel coordinates)
tma_center_x = 35395
tma_center_y = 29292
tma_radius = 2346
num_coordinates = 5



# Generate tile coordinates within the TMA-41 area at highest DeepZoom level
coords_tiles = generate_coordinates_in_circle(
    tma_center_x, tma_center_y, tma_radius, num_coordinates
)

# coords_tiles = np.array([
#     (4084.35, 12350.67),
#     (2542.91, 11008.26),
#     (2910.34, 11243.42),
#     (1978.78, 12071.86),
#     (4666.83, 11904.06)
# ])

print('coords_tiles (tile coordinates at highest level):', coords_tiles)
print(type(coords_tiles))



# Create DeepZoom generator to find the highest level
dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)
highest_level = dz.level_count - 1

print(f"Using DeepZoom level {highest_level} (highest resolution)")
print(f"Level {highest_level} dimensions: {dz.level_dimensions[highest_level]} tiles")

# Verify coordinates are within bounds
level_dims = dz.level_dimensions[highest_level]
valid_coords = []
for coord in coords_tiles:
    if (0 <= coord[0] < level_dims[0] and 0 <= coord[1] < level_dims[1]):
        valid_coords.append(coord)
    else:
        print(f"Coordinate {coord} is out of bounds for level {highest_level}")

coords_to_use = np.array(valid_coords)
print(f"Valid coordinates: {coords_to_use}")

if len(coords_to_use) == 0:
    print("No valid coordinates found. Generating new ones within bounds...")
    # Generate coordinates within the valid range
    coords_to_use = np.random.randint(0, min(level_dims), size=(num_coordinates, 2))
    print(f"New coordinates: {coords_to_use}")



print(f"\nInitializing segmentor...")
segmentor = CellViTSegmentor.from_histoplus(
    mpp=MPP,
    mixed_precision=True,
    inference_image_size=INFERENCE_IMAGE_SIZE,
)

# Extract using the highest DeepZoom level
results = extract(
    slide=slide,
    coords=coords_to_use,
    deepzoom_level=highest_level,  # Use highest level
    segmentor=segmentor,
    tile_size=224,
    batch_size=1
)

print("✅ Extraction completed successfully!")
os.makedirs("output", exist_ok=True)
results.save("output/results.json")
print("✅ Results saved to output/results.json")
    
# Add this after your extraction code
import json

print("FULL RESULTS.JSON CONTENT (pretty printed):")
print("=" * 60)

try:
    with open("output/results.json", "r") as f:
        results_data = json.load(f)
    
    # Pretty print the entire JSON
    print(json.dumps(results_data, indent=2, default=str))
    
except Exception as e:
    print(f"Error: {e}")

#epi700
# import os
# import openslide
# import numpy as np
# import math
# from histoplus.extract import extract
# from histoplus.helpers.segmentor import CellViTSegmentor
# from openslide.deepzoom import DeepZoomGenerator

# MPP = 0.5
# INFERENCE_IMAGE_SIZE = 784

# # Open your slide
# slide = openslide.open_slide("/home/es2122/ELENA/EPI-700 Feb24 TMA 3A TUMOUR S24-054 HE .ndpi")

# # Your coordinate points (pixel coordinates)
# coords_tiles = np.array([
#     (10731, 9531),
#     (8975, 8975),
#     (10553, 10233),
#     (9600, 10000),
#     (9806, 10146),
#     (10355, 10355),
#     (11054, 10355)
# ])


# print('coords_tiles (pixel coordinates):', coords_tiles)
# print(type(coords_tiles))

# # Create DeepZoom generator to find the highest level
# dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)
# highest_level = dz.level_count - 1

# print(f"Using DeepZoom level {highest_level} (highest resolution)")
# level_dims = dz.level_dimensions[highest_level]
# print(f"Level {highest_level} dimensions (tiles): {level_dims}")

# # Convert pixel coordinates to tile indices
# tile_size = 224
# valid_coords_indices = []

# for coord in coords_tiles:
#     tile_x = int(coord[0] // tile_size)
#     tile_y = int(coord[1] // tile_size)
#     # Check if within bounds
#     if (0 <= tile_x < level_dims[0]) and (0 <= tile_y < level_dims[1]):
#         valid_coords_indices.append([tile_x, tile_y])
#     else:
#         print(f"Coordinate {coord} (tile {tile_x},{tile_y}) is out of bounds for level {highest_level}")

# # Convert to numpy array
# coords_to_use = np.array(valid_coords_indices)

# if len(coords_to_use) == 0:
#     print("No valid coordinates found within bounds.")
# else:
#     print(f"Valid tile indices: {coords_to_use}")

# # Initialize segmentor
# print(f"\nInitializing segmentor...")
# segmentor = CellViTSegmentor.from_histoplus(
#     mpp=MPP,
#     mixed_precision=True,
#     inference_image_size=INFERENCE_IMAGE_SIZE,
# )

# # Perform extraction using tile indices
# results = extract(
#     slide=slide,
#     coords=coords_to_use,
#     deepzoom_level=highest_level,
#     segmentor=segmentor,
#     tile_size=224,
#     batch_size=1
# )

# print("✅ Extraction completed successfully!")
# os.makedirs("output_epi700", exist_ok=True)
# results.save("output_epi700/results.json")
# print("✅ Results saved to output_epi700/results.json")

# # Load and pretty-print results JSON
# import json

# print("FULL RESULTS.JSON CONTENT (pretty printed):")
# print("=" * 60)

# try:
#     with open("output_epi700/results.json", "r") as f:
#         results_data = json.load(f)
#     print(json.dumps(results_data, indent=2, default=str))
# except Exception as e:
#     print(f"Error reading results: {e}")



