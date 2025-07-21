# yolo_predictor.py
import os
import logging
import tempfile
import numpy as np
import tifffile
from io import BytesIO
import cv2
from PIL import Image
from rasterio.transform import from_bounds
from ultralytics import YOLO
from ndvi_predictor import normalize_rgb, predict_ndvi
from resize_image import resize_image_optimized

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yolo_model(model_path):
    """Load YOLO model from .pt file"""
    logger.info(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)

def predict_yolo(yolo_model, image_path, conf=0.25):
    """
    Predict using YOLO model on 4-channel TIFF image
    
    Args:
        yolo_model: Loaded YOLO model
        image_path: Path to 4-channel TIFF image
        conf: Confidence threshold
    
    Returns:
        results: YOLO results object
    """
    logger.info(f"Starting YOLO prediction on: {image_path} with confidence: {conf}")
    
    # Verify file exists and has correct format
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Quick validation of the TIFF file
        test_array = tifffile.imread(image_path)
        logger.info(f"TIFF file shape: {test_array.shape}, dtype: {test_array.dtype}")
        
        # Validate channels
        if len(test_array.shape) == 3:
            channels = test_array.shape[0] if test_array.shape[0] <= 4 else test_array.shape[2]
        else:
            channels = 1
        
        if channels != 4:
            raise ValueError(f"Expected 4-channel image, got {channels} channels")
            
    except Exception as e:
        logger.error(f"Error validating TIFF file: {e}")
        raise
    
    logger.info("Running YOLO model inference...")
    # Run YOLO prediction directly on the input file
    results = yolo_model([image_path], conf=conf)
    
    logger.info(f"YOLO prediction completed. Results type: {type(results[0])}")
    return results[0]  # Return first result

def create_4channel_tiff(rgb_array, ndvi_array, output_path):
    """
    Create a 4-channel TIFF file with RGB channels + NDVI channel
    
    Args:
        rgb_array: RGB image array (H, W, 3)
        ndvi_array: NDVI array (H, W) with values in [-1, 1]
        output_path: Path to save the 4-channel TIFF
    """
    logger.info(f"Creating 4-channel TIFF file at: {output_path}")
    logger.info(f"RGB shape: {rgb_array.shape}, NDVI shape: {ndvi_array.shape}")
    
    # Ensure RGB is in uint8 format
    if rgb_array.dtype != np.uint8:
        if rgb_array.max() <= 1.0:
            rgb_uint8 = (rgb_array * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_array.astype(np.uint8)
    else:
        rgb_uint8 = rgb_array
    
    # Convert NDVI from [-1, 1] to [0, 255] uint8 format (same as reference code)
    ndvi_scaled = (((ndvi_array + 1) / 2) * 255).astype(np.uint8)
    
    logger.info(f"RGB range: [{rgb_uint8.min()}, {rgb_uint8.max()}]")
    logger.info(f"NDVI scaled range: [{ndvi_scaled.min()}, {ndvi_scaled.max()}]")
    
    # Stack RGB + NDVI to create 4-channel image
    # Format: (channels, height, width) - channel-first format
    four_channel = np.stack([
        rgb_uint8[:, :, 0],  # R channel
        rgb_uint8[:, :, 1],  # G channel  
        rgb_uint8[:, :, 2],  # B channel
        ndvi_scaled          # NDVI channel
    ], axis=0)
    
    logger.info(f"4-channel array shape: {four_channel.shape}, dtype: {four_channel.dtype}")
    logger.info(f"4-channel range: [{four_channel.min()}, {four_channel.max()}]")
    
    # Save as TIFF using tifffile
    tifffile.imwrite(output_path, four_channel)
    logger.info(f"Successfully saved 4-channel TIFF (RGB+NDVI format) to: {output_path}")

def predict_pipeline(ndvi_model, yolo_model, rgb_array, conf=0.25):
    """
    Full pipeline: RGB -> NDVI -> 32-bit 4-channel TIFF (RGB+NDVI) -> YOLO prediction
    
    Args:
        ndvi_model: Loaded NDVI prediction model
        yolo_model: Loaded YOLO model
        rgb_array: RGB image as numpy array (H, W, 3)
        conf: Confidence threshold for YOLO
    
    Returns:
        results: YOLO results object
    """
    logger.info("Starting full prediction pipeline")
    logger.info(f"Input RGB array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
    
    # Step 1: Resize RGB image to target size
    logger.info("Step 1: Resizing RGB image to target size")
    target_size = (640, 640)  # (height, width)
    rgb_resized = resize_image_optimized(rgb_array, target_size)
    logger.info(f"Resized RGB shape: {rgb_resized.shape}")
    
    # Step 2: Normalize RGB image
    logger.info("Step 2: Normalizing RGB image")
    normalized_rgb = normalize_rgb(rgb_resized)
    logger.info(f"Normalized RGB shape: {normalized_rgb.shape}, range: [{normalized_rgb.min():.3f}, {normalized_rgb.max():.3f}]")
    
    # Step 3: Predict NDVI
    logger.info("Step 3: Predicting NDVI from RGB")
    ndvi_prediction = predict_ndvi(ndvi_model, normalized_rgb)
    logger.info(f"NDVI prediction shape: {ndvi_prediction.shape}, range: [{ndvi_prediction.min():.3f}, {ndvi_prediction.max():.3f}]")
    
    # Step 4: Create 4-channel TIFF file
    logger.info("Step 4: Creating 4-channel TIFF file (RGB+NDVI)")
    
    # Create temporary file for the 4-channel TIFF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp_file:
        tiff_path = tmp_file.name
    
    try:
        # Create the 4-channel TIFF using resized RGB and predicted NDVI
        create_4channel_tiff(rgb_resized, ndvi_prediction, tiff_path)
        
        # Verify the created file
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Failed to create 4-channel TIFF at: {tiff_path}")
        
        file_size = os.path.getsize(tiff_path)
        logger.info(f"Created 4-channel TIFF file size: {file_size} bytes")
        
        # Step 5: Run YOLO prediction on the 4-channel TIFF
        logger.info("Step 5: Running YOLO prediction on 4-channel TIFF")
        results = predict_yolo(yolo_model, tiff_path, conf=conf)
        
        logger.info("Full pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(tiff_path):
            try:
                os.unlink(tiff_path)
                logger.info(f"Cleaned up temporary file: {tiff_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

def validate_4channel_tiff(tiff_path):
    """
    Validate that a TIFF file has exactly 4 channels
    
    Args:
        tiff_path: Path to TIFF file
    
    Returns:
        bool: True if valid 4-channel TIFF, False otherwise
    """
    try:
        array = tifffile.imread(tiff_path)
        
        if len(array.shape) == 3:
            channels = array.shape[0] if array.shape[0] <= 4 else array.shape[2]
        else:
            channels = 1
            
        logger.info(f"TIFF validation - Shape: {array.shape}, Channels: {channels}")
        return channels == 4
        
    except Exception as e:
        logger.error(f"Error validating TIFF file: {e}")
        return False

# Additional functions for yolo_predictor.py

def predict_yolo_with_image(yolo_model, image_path, conf=0.25, save_path=None):
    """
    Predict using YOLO model on 4-channel TIFF image and return annotated image
    
    Args:
        yolo_model: Loaded YOLO model
        image_path: Path to 4-channel TIFF image
        conf: Confidence threshold
        save_path: Optional path to save the annotated image
    
    Returns:
        annotated_image: PIL Image object with annotations
    """
    logger.info(f"Starting YOLO prediction with image output on: {image_path} with confidence: {conf}")
    
    # Verify file exists and has correct format
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Quick validation of the TIFF file
        test_array = tifffile.imread(image_path)
        logger.info(f"TIFF file shape: {test_array.shape}, dtype: {test_array.dtype}")
        
        # Validate channels
        if len(test_array.shape) == 3:
            channels = test_array.shape[0] if test_array.shape[0] <= 4 else test_array.shape[2]
        else:
            channels = 1
        
        if channels != 4:
            raise ValueError(f"Expected 4-channel image, got {channels} channels")
            
    except Exception as e:
        logger.error(f"Error validating TIFF file: {e}")
        raise
    
    logger.info("Running YOLO model inference with image output...")
    
    # Run YOLO prediction directly on the input file
    results = yolo_model([image_path], conf=conf)
    result = results[0]
    
    # Create temporary file for saving annotated image
    if save_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            save_path = tmp_file.name
    
    try:
        # Save the annotated image using ultralytics built-in method
        result.save(save_path)
        logger.info(f"Annotated image saved to: {save_path}")
        
        # Load the saved image and convert to PIL Image
        annotated_image = Image.open(save_path).convert('RGB')
        logger.info(f"YOLO prediction with image output completed successfully")
        
        return annotated_image
        
    except Exception as e:
        logger.error(f"Error saving annotated image: {e}")
        raise
    finally:
        # Clean up temporary file if we created it
        if save_path.endswith('.png') and os.path.exists(save_path):
            try:
                os.unlink(save_path)
                logger.info(f"Cleaned up temporary annotated image file: {save_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

def predict_pipeline_with_image(ndvi_model, yolo_model, rgb_array, conf=0.25):
    """
    Full pipeline with image output: RGB -> NDVI -> 32-bit 4-channel TIFF (RGB+NDVI) -> YOLO prediction -> Annotated Image
    
    Args:
        ndvi_model: Loaded NDVI prediction model
        yolo_model: Loaded YOLO model
        rgb_array: RGB image as numpy array (H, W, 3)
        conf: Confidence threshold for YOLO
    
    Returns:
        annotated_image: PIL Image object with YOLO annotations
    """
    logger.info("Starting full prediction pipeline with image output")
    logger.info(f"Input RGB array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
    
    # Step 1: Resize RGB image to target size
    logger.info("Step 1: Resizing RGB image to target size")
    target_size = (640, 640)  # (height, width)
    rgb_resized = resize_image_optimized(rgb_array, target_size)
    logger.info(f"Resized RGB shape: {rgb_resized.shape}")
    
    # Step 2: Normalize RGB image
    logger.info("Step 2: Normalizing RGB image")
    normalized_rgb = normalize_rgb(rgb_resized)
    logger.info(f"Normalized RGB shape: {normalized_rgb.shape}, range: [{normalized_rgb.min():.3f}, {normalized_rgb.max():.3f}]")
    
    # Step 3: Predict NDVI
    logger.info("Step 3: Predicting NDVI from RGB")
    ndvi_prediction = predict_ndvi(ndvi_model, normalized_rgb)
    logger.info(f"NDVI prediction shape: {ndvi_prediction.shape}, range: [{ndvi_prediction.min():.3f}, {ndvi_prediction.max():.3f}]")
    
    # Step 4: Create 4-channel TIFF file
    logger.info("Step 4: Creating 4-channel TIFF file (RGB+NDVI)")
    
    # Create temporary file for the 4-channel TIFF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp_file:
        tiff_path = tmp_file.name
    
    try:
        # Create the 4-channel TIFF using resized RGB and predicted NDVI
        create_4channel_tiff(rgb_resized, ndvi_prediction, tiff_path)
        
        # Verify the created file
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Failed to create 4-channel TIFF at: {tiff_path}")
        
        file_size = os.path.getsize(tiff_path)
        logger.info(f"Created 4-channel TIFF file size: {file_size} bytes")
        
        # Step 5: Run YOLO prediction on the 4-channel TIFF and get annotated image
        logger.info("Step 5: Running YOLO prediction on 4-channel TIFF with image output")
        annotated_image = predict_yolo_with_image(yolo_model, tiff_path, conf=conf)
        
        logger.info("Full pipeline with image output completed successfully")
        return annotated_image
        
    except Exception as e:
        logger.error(f"Error in pipeline with image output: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(tiff_path):
            try:
                os.unlink(tiff_path)
                logger.info(f"Cleaned up temporary file: {tiff_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

def pil_image_to_bytes(image, format='PNG'):
    """
    Convert PIL Image to bytes for API response
    
    Args:
        image: PIL Image object
        format: Image format ('PNG', 'JPEG', etc.)
    
    Returns:
        BytesIO: Image as bytes buffer
    """
    img_bytes = BytesIO()
    image.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes