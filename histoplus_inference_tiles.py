import os
import json
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load your WSI and results
# slide_path = "/home/es2122/ELENA/EPI-700 Feb24 TMA 3A TUMOUR S24-054 HE .ndpi"
slide_path = "/home/es2122/SLIDE_047_1.0.1_R000_20X_VHE_F.tif"
slide = openslide.open_slide(slide_path)

print(f"Slide dimensions at level 0: {slide.dimensions}")
print(f"Slide level count: {slide.level_count}")
print(f"Slide level dimensions: {slide.level_dimensions}")
print(f"Slide level downsamples: {slide.level_downsamples}")

with open("output/results.json", "r") as f:
# with open("output_epi700/results.json", "r") as f:
    results = json.load(f)

print(f"\nLoaded results JSON keys: {list(results.keys())}")
print(f"Number of cell masks (tiles): {len(results.get('cell_masks', []))}")

# Create output directory
# os.makedirs("output_epi700/comparison_visualizations", exist_ok=True)
os.makedirs("output/comparison_visualizations", exist_ok=True)

# Get cell type information from results
cell_types = set()
for tile_data in results.get("cell_masks", []):
    for cell in tile_data.get('masks', []):
        cell_type = cell.get('cell_type', 'Unknown')
        cell_types.add(cell_type)

print(f"Found cell types: {cell_types}")

# Create color mapping for cell types based on your actual distribution
cell_type_colors = {
    'Cancer cell': 'red',
    'Apoptotic Body': 'blue', 
    'Red blood cell': 'green',
    'Plasmocytes': 'orange',
    'Neutrophils': 'purple',
    'Unknown': 'gray'
}

# Assign colors to actual cell types found
available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
color_mapping = {}
for i, cell_type in enumerate(sorted(cell_types)):
    if cell_type in cell_type_colors:
        color_mapping[cell_type] = cell_type_colors[cell_type]
    else:
        color_mapping[cell_type] = available_colors[i % len(available_colors)]

print(f"Cell type color mapping: {color_mapping}")

# Process each tile in cell_masks
for tile_idx, tile_data in enumerate(results.get("cell_masks", [])):
    print(f"\n{'='*50}")
    print(f"Processing tile {tile_idx + 1}...")
    
    # Get tile coordinates from JSON
    tile_x_virtual = tile_data["x"]
    tile_y_virtual = tile_data["y"]
    tile_width_virtual = tile_data["width"]
    tile_height_virtual = tile_data["height"]
    
    print(f"JSON data (virtual level):")
    print(f"  Position: ({tile_x_virtual}, {tile_y_virtual})")
    print(f"  Size: {tile_width_virtual}x{tile_height_virtual}")
    print(f"  Cells: {len(tile_data.get('masks', []))}")
    
    # Convert virtual coordinates to level 0
    tile_x_0 = int(tile_x_virtual * 224)
    tile_y_0 = int(tile_y_virtual * 224)
    tile_width_0 = 224
    tile_height_0 = 224
    
    print(f"Converted to level 0:")
    print(f"  Position: ({tile_x_0}, {tile_y_0})")
    print(f"  Size: {tile_width_0}x{tile_height_0}")
    
    # Check if coordinates are within slide bounds
    slide_width, slide_height = slide.dimensions
    print(f"Slide bounds: 0-{slide_width}, 0-{slide_height}")
    
    if (tile_x_0 < 0 or tile_y_0 < 0 or 
        tile_x_0 + tile_width_0 > slide_width or 
        tile_y_0 + tile_height_0 > slide_height):
        print(f"❌ WARNING: Tile coordinates are outside slide bounds!")
        print(f"   Would read from ({tile_x_0}, {tile_y_0}) to ({tile_x_0 + tile_width_0}, {tile_y_0 + tile_height_0})")
        
        # Use alternative positions for out-of-bounds tiles
        tma_center_x = 8899
        tma_center_y = 4086
        positions = [
            (tma_center_x, tma_center_y),
            (tma_center_x + 1000, tma_center_y),
            (tma_center_x, tma_center_y + 1000),
            (tma_center_x - 1000, tma_center_y),
            (tma_center_x, tma_center_y - 1000)
        ]
        
        if tile_idx < len(positions):
            tile_x_0, tile_y_0 = positions[tile_idx]
            print(f"Using alternative position: ({tile_x_0}, {tile_y_0})")
        else:
            print(f"Skipping tile {tile_idx + 1} due to out-of-bounds coordinates")
            continue
    
    try:
        # Extract original patch from WSI at level 0
        print(f"Reading region from slide at ({tile_x_0}, {tile_y_0}) with size ({tile_width_0}, {tile_height_0})")
        original_patch = slide.read_region(
            (tile_x_0, tile_y_0), 
            0, 
            (tile_width_0, tile_height_0)
        ).convert('RGB')
        original_array = np.array(original_patch)
        print(f"✅ Successfully read patch of shape {original_array.shape}")
        
        # Create figure with side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Original patch
        ax1.imshow(original_array)
        ax1.set_title(f"Original Tile {tile_idx + 1}\nPosition: ({tile_x_0}, {tile_y_0})", fontsize=14, weight='bold')
        ax1.axis('off')
        
        # Right: Patch with cell masks
        ax2.imshow(original_array)
        
        # Draw cell masks with type-based coloring
        masks = tile_data.get('masks', [])
        cell_count_by_type = {}
        
        for cell_idx, cell in enumerate(masks):
            cell_type = cell.get('cell_type', 'Unknown')
            color = color_mapping.get(cell_type, 'purple')
            
            # Count cells by type
            if cell_type not in cell_count_by_type:
                cell_count_by_type[cell_type] = 0
            cell_count_by_type[cell_type] += 1
            
            # Convert cell coordinates
            coordinates_virtual = cell.get('coordinates', [])
            if coordinates_virtual:
                coordinates_relative = []
                for point in coordinates_virtual:
                    if len(point) >= 2:
                        x_rel = point[0]
                        y_rel = point[1]
                        x_rel = max(0, min(223, x_rel))
                        y_rel = max(0, min(223, y_rel))
                        coordinates_relative.append([x_rel, y_rel])
                
                if len(coordinates_relative) > 2:
                    coordinates_array = np.array(coordinates_relative)
                    # Draw filled polygon with transparency (REMOVED letter labels)
                    polygon = patches.Polygon(coordinates_array, linewidth=2, 
                                            edgecolor=color, facecolor=color, alpha=0.3)
                    ax2.add_patch(polygon)
        
        # Set title for right panel
        total_cells = len(masks)
        title_parts = [f"Tile {tile_idx + 1} with Cell Masks", f"Total cells: {total_cells}"]
        ax2.set_title("\n".join(title_parts), fontsize=14, weight='bold')
        ax2.axis('off')
        
        # Create legend for cell types
        legend_elements = []
        for cell_type, color in color_mapping.items():
            if cell_type in cell_count_by_type:
                legend_elements.append(
                    patches.Patch(facecolor=color, alpha=0.7, 
                                label=f'{cell_type} ({cell_count_by_type.get(cell_type, 0)})')
                )
        
        if legend_elements:
            # Position legend below the images
            ax2.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, -0.1), fontsize=12,
                     title="Cell Types", title_fontsize=13, ncol=2)
        
        # Add overall information
        fig.suptitle(f"Tile {tile_idx + 1} Analysis - {total_cells} Cells Detected", 
                    fontsize=16, weight='bold', y=1.00)
        
        plt.tight_layout()
        
        # Save the comparison visualization
        # output_path = f"output_epi700/comparison_visualizations/tile_{tile_idx + 1}_comparison.png"
        output_path = f"output/comparison_visualizations/tile_{tile_idx + 1}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved comparison: {output_path}")
        print(f"✅ Cell counts: {cell_count_by_type}")
        
    except Exception as e:
        print(f"❌ ERROR processing tile {tile_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*50}")
print("Comparison visualization generation complete!")
# print(f"Saved {len(results.get('cell_masks', []))} comparison images to output_epi700/comparison_visualizations/")
print(f"Saved {len(results.get('cell_masks', []))} comparison images to output/comparison_visualizations/")
# Print final summary
print(f"\n{'='*50}")
print("FINAL SUMMARY")
print(f"{'='*50}")
print(f"Total tiles processed: {len(results.get('cell_masks', []))}")

total_cells_all = 0
final_cell_counts = {}
for tile_data in results.get("cell_masks", []):
    for cell in tile_data.get('masks', []):
        cell_type = cell.get('cell_type', 'Unknown')
        final_cell_counts[cell_type] = final_cell_counts.get(cell_type, 0) + 1
        total_cells_all += 1

print(f"Total cells detected: {total_cells_all}")
print("\nCell type distribution across all tiles:")
for cell_type, count in sorted(final_cell_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_cells_all) * 100
    print(f"  {cell_type}: {count} ({percentage:.1f}%)")

