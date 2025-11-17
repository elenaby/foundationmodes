import os
import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

def create_wsi_thumbnail_with_masks(wsi_path, geojson_path, output_path, thumbnail_size=2000):
    print("Reading WSI...")
    try:
        img = tifffile.imread(wsi_path, aszarr=True)
        img_data = zarr.open(img, 'r')
        
        if hasattr(img_data, '__len__') and len(img_data) > 0:
            base_level = img_data[0]
            h, w = base_level.shape[0], base_level.shape[1]
            print(f"Base level dimensions: {w} x {h}")
            
            thumbnail = np.array(base_level)
            
            if len(thumbnail.shape) == 3 and thumbnail.shape[2] > 3:
                thumbnail = thumbnail[:, :, :3]
            elif len(thumbnail.shape) == 2:
                thumbnail = np.stack([thumbnail] * 3, axis=-1)
                
        else:
            print("Warning: No pyramid levels found, reading full resolution (this may be slow)")
            full_img = tifffile.imread(wsi_path)
            h, w = full_img.shape[0], full_img.shape[1]
            
            scale_factor = min(thumbnail_size / w, thumbnail_size / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            
            if len(full_img.shape) == 3:
                thumbnail = np.array(Image.fromarray(full_img).resize((new_w, new_h), Image.LANCZOS))
            else:
                thumbnail = np.array(Image.fromarray(full_img).resize((new_w, new_h), Image.LANCZOS))
                thumbnail = np.stack([thumbnail] * 3, axis=-1)
                
    except Exception as e:
        print(f"Error reading WSI: {e}")
        return False

    print("Creating mask from GeoJSON...")
    try:
        annotations = gpd.read_file(geojson_path)
        
        mask_thumbnail = np.zeros((thumbnail.shape[0], thumbnail.shape[1]), dtype=np.uint8)
        
        with tifffile.TiffFile(wsi_path) as tif:
            original_width = tif.pages[0].shape[1]
            original_height = tif.pages[0].shape[0]
        
        print(f"Original WSI dimensions: {original_width} x {original_height}")
        print(f"Thumbnail dimensions: {thumbnail.shape[1]} x {thumbnail.shape[0]}")
        
        scale_x = thumbnail.shape[1] / original_width
        scale_y = thumbnail.shape[0] / original_height
        
        for idx, geom in enumerate(annotations.geometry):
            if geom is not None and not geom.is_empty:
                if isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        if poly.is_valid:
                            scaled_coords = []
                            if hasattr(poly.exterior, 'coords'):
                                for x, y in poly.exterior.coords:
                                    scaled_coords.append((x * scale_x, y * scale_y))
                                scaled_poly = Polygon(scaled_coords)
                                features.rasterize([(scaled_poly, 1)], out=mask_thumbnail, 
                                                 out_shape=mask_thumbnail.shape,
                                                 merge_alg=rasterio.enums.MergeAlg.add)
                elif isinstance(geom, Polygon) and geom.is_valid:
                    scaled_coords = []
                    if hasattr(geom.exterior, 'coords'):
                        for x, y in geom.exterior.coords:
                            scaled_coords.append((x * scale_x, y * scale_y))
                        scaled_poly = Polygon(scaled_coords)
                        features.rasterize([(scaled_poly, 1)], out=mask_thumbnail, 
                                         out_shape=mask_thumbnail.shape,
                                         merge_alg=rasterio.enums.MergeAlg.add)
        
        print(f"Mask created with {np.sum(mask_thumbnail > 0)} annotated pixels")
        
    except Exception as e:
        print(f"Error processing GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Creating overlay...")
    try:
        if thumbnail.dtype != np.uint8:
            thumbnail = (thumbnail / thumbnail.max() * 255).astype(np.uint8)
        
        pil_thumbnail = Image.fromarray(thumbnail)
        
        overlay = Image.new('RGBA', pil_thumbnail.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        overlay_array = np.zeros((*mask_thumbnail.shape, 4), dtype=np.uint8)
        red_mask = mask_thumbnail > 0
        overlay_array[red_mask] = [255, 0, 0, 128]
        
        red_overlay = Image.fromarray(overlay_array, mode='RGBA')
        
        pil_thumbnail_rgba = pil_thumbnail.convert('RGBA')
        result = Image.alpha_composite(pil_thumbnail_rgba, red_overlay)
        
        result_rgb = result.convert('RGB')
        
        result_rgb.save(output_path, 'JPEG', quality=95)
        print(f"Thumbnail with masks saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating overlay: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_detailed_overlay(wsi_path, geojson_path, output_path, thumbnail_size=1000):
    from matplotlib.patches import Polygon as MplPolygon
    
    img = tifffile.imread(wsi_path, aszarr=True)
    img_data = zarr.open(img, 'r')
    base_level = img_data[0]
    thumbnail = np.array(base_level)
    
    if len(thumbnail.shape) == 3 and thumbnail.shape[2] > 3:
        thumbnail = thumbnail[:, :, :3]
    
    with tifffile.TiffFile(wsi_path) as tif:
        original_width = tif.pages[0].shape[1]
        original_height = tif.pages[0].shape[0]
    
    scale_x = thumbnail.shape[1] / original_width
    scale_y = thumbnail.shape[0] / original_height
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(thumbnail)
    ax.set_title('WSI with Annotations Overlay', fontsize=14)
    
    annotations = gpd.read_file(geojson_path)
    
    for idx, geom in enumerate(annotations.geometry):
        if geom is not None and not geom.is_empty:
            if isinstance(geom, Polygon) and geom.is_valid:
                if hasattr(geom.exterior, 'coords'):
                    scaled_coords = [(x * scale_x, y * scale_y) for x, y in geom.exterior.coords]
                    polygon = MplPolygon(scaled_coords, closed=True, 
                                       edgecolor='red', facecolor='red', 
                                       alpha=0.3, linewidth=2)
                    ax.add_patch(polygon)
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    if poly.is_valid and hasattr(poly.exterior, 'coords'):
                        scaled_coords = [(x * scale_x, y * scale_y) for x, y in poly.exterior.coords]
                        polygon = MplPolygon(scaled_coords, closed=True, 
                                           edgecolor='red', facecolor='red', 
                                           alpha=0.3, linewidth=2)
                        ax.add_patch(polygon)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Detailed overlay saved to: {output_path}")

if __name__ == "__main__":
    wsi_path = ""
    geojson_path = ""
    output_path = ""
    
    success = create_wsi_thumbnail_with_masks(
        wsi_path=wsi_path,
        geojson_path=geojson_path,
        output_path=output_path,
        thumbnail_size=1000
    )
    
    if success:
        print("✅ Thumbnail created successfully!")
        
        detailed_output = output_path.replace('.jpg', '_detailed.png')
        create_detailed_overlay(wsi_path, geojson_path, detailed_output)
        
    else:
        print("❌ Failed to create thumbnail")