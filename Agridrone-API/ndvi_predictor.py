# ndvi_predictor.py
import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from efficientnet.tfkeras import EfficientNetB2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow.keras.models import model_from_json
import traceback
import gc

# Custom loss functions and activation functions
def balanced_mse_loss(y_true, y_pred):
    mse = tf.square(y_true - y_pred)
    negative_weight = tf.where(y_true < -0.2, 1.5, 1.0)
    boundary_weight = tf.where(tf.abs(y_true) > 0.5, 1.5, 1.0)
    weights = negative_weight * boundary_weight
    weighted_mse = weights * mse
    return tf.reduce_mean(mse)

def custom_mae(y_true, y_pred):
    mae = tf.abs(y_true - y_pred)
    return tf.reduce_mean(mae)

def load_model(models_dir):
    """Load NDVI prediction model with custom objects"""
    
    # Define custom objects dictionary
    custom_objects = {
        'balanced_mse_loss': balanced_mse_loss,
        'custom_mae': custom_mae
    }
    
    try:
        # Load model architecture
        with open(os.path.join(models_dir, "model_architecture.json"), "r") as json_file:
            model_json = json_file.read()
        
        model = model_from_json(model_json, custom_objects=custom_objects)
        
        # Load weights
        model.load_weights(os.path.join(models_dir, "best_model_weights.weights.h5"))
        
        # Compile model with custom functions
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss=balanced_mse_loss,
            metrics=[custom_mae, 'mse']
        )
        
        return model
    except Exception as e:
        traceback.print_exc()
        return None

def normalize_rgb(rgb):
    """Normalize RGB image to [0, 1] range using percentile normalization"""
    rgb_norm = rgb.copy().astype(np.float32)
    
    # Handle different input ranges
    if rgb.max() > 1:
        rgb_norm = rgb_norm / 255.0
    
    for b in range(3):
        band = rgb_norm[:, :, b]
        min_val, max_val = np.percentile(band, [1, 99])
        if min_val < max_val:
            rgb_norm[:, :, b] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    
    return rgb_norm

def predict_ndvi(model, rgb_np):
    """
    Faster NDVI prediction with larger tiles and more efficient processing
    
    Args:
        model: Loaded NDVI prediction model
        rgb_np: RGB image as numpy array (H, W, 3) normalized to [0, 1]
    
    Returns:
        ndvi_pred: Predicted NDVI as numpy array (H, W) in range [-1, 1]
    """
    height, width = rgb_np.shape[:2]
    
    # Larger tiles for faster processing
    tile_size = 512
    stride = int(tile_size * 0.75)  # 25% overlap
    
    # For smaller images, process whole image at once
    if height <= tile_size and width <= tile_size:
        # Pad to tile size if needed
        pad_height = max(0, tile_size - height)
        pad_width = max(0, tile_size - width)
        if pad_height > 0 or pad_width > 0:
            rgb_padded = np.pad(rgb_np, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')
        else:
            rgb_padded = rgb_np
        
        # Single prediction
        pred = model.predict(np.expand_dims(rgb_padded, axis=0), verbose=0, batch_size=1)[0, :, :, 0]
        return pred[:height, :width]
    
    # Initialize output arrays
    ndvi_pred = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Pre-compute weights for efficiency
    y, x = np.mgrid[0:tile_size, 0:tile_size]
    base_weights = np.minimum(np.minimum(x, tile_size - x - 1), np.minimum(y, tile_size - y - 1))
    base_weights = np.clip(base_weights, 0, 64) / 64
    
    # Collect all tiles first
    tiles = []
    positions = []
    
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            # Calculate actual tile bounds
            end_i = min(i + tile_size, height)
            end_j = min(j + tile_size, width)
            actual_height = end_i - i
            actual_width = end_j - j
            
            # Extract tile
            tile = rgb_np[i:end_i, j:end_j, :]
            
            # Pad if necessary
            if actual_height < tile_size or actual_width < tile_size:
                pad_height = tile_size - actual_height
                pad_width = tile_size - actual_width
                tile = np.pad(tile, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')
            
            tiles.append(tile)
            positions.append((i, j, actual_height, actual_width))
    
    # Process all tiles in larger batches
    batch_size = 8  # Process 8 tiles at once
    for batch_start in range(0, len(tiles), batch_size):
        batch_end = min(batch_start + batch_size, len(tiles))
        batch_tiles = np.array(tiles[batch_start:batch_end])
        
        # Predict batch
        batch_preds = model.predict(batch_tiles, verbose=0, batch_size=batch_size)
        
        # Apply predictions
        for k in range(batch_end - batch_start):
            pred = batch_preds[k, :, :, 0]
            i, j, actual_height, actual_width = positions[batch_start + k]
            
            # Use appropriate weights
            weights = base_weights[:actual_height, :actual_width]
            
            # Add to output
            ndvi_pred[i:i+actual_height, j:j+actual_width] += pred[:actual_height, :actual_width] * weights
            weight_map[i:i+actual_height, j:j+actual_width] += weights
        
        # Clean up batch
        del batch_tiles, batch_preds
    
    # Normalize by weights
    mask = weight_map > 0
    ndvi_pred[mask] = ndvi_pred[mask] / weight_map[mask]
    
    return ndvi_pred

def create_visualization(rgb, ndvi):
    """
    Create visualization of RGB input and predicted NDVI
    
    Args:
        rgb: RGB image array
        ndvi: NDVI prediction array
    
    Returns:
        buf: BytesIO buffer containing the visualization as PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display RGB image
    rgb_disp = np.clip(rgb / 255 if rgb.max() > 1 else rgb, 0, 1)
    axes[0].imshow(rgb_disp)
    axes[0].set_title("RGB Input")
    axes[0].axis("off")
    
    # Display NDVI with color map
    im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title("Predicted NDVI")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1])
    
    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf