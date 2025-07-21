# test app.py
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from ndvi_predictor import load_model, normalize_rgb, predict_ndvi, create_visualization
from yolo_predictor import load_yolo_model, predict_yolo, predict_pipeline
from PIL import Image
from io import BytesIO
import numpy as np
import zipfile
import json
import rasterio
from rasterio.transform import from_bounds
import tempfile
import os
import logging
from resize_image import resize_image_optimized, resize_image_simple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load models at startup
try:
    ndvi_model = load_model("ndvi_best_model")
    logger.info("NDVI model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load NDVI model: {e}")
    ndvi_model = None

try:
    yolo_model = load_yolo_model("best_yolo_model.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

@app.get("/")
async def root():
    return {"message": "Welcome to the NDVI and YOLO prediction API!"}

# Example usage in your predict_ndvi endpoint:
@app.post("/predict_ndvi/")
async def predict_ndvi_api(file: UploadFile = File(...)):
    """Predict NDVI from RGB image"""
    if ndvi_model is None:
        return JSONResponse(status_code=500, content={"error": "NDVI model not loaded"})
    
    try:
        # Define target size (height, width)
        target_size = (640, 640)
        
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        
        # Convert to numpy array
        rgb_array = np.array(img)
        
        # Resize image to target size
        rgb_resized = resize_image_optimized(rgb_array, target_size)
        
        # Normalize the resized image
        norm_img = normalize_rgb(rgb_resized)
        
        # Predict NDVI
        pred_ndvi = predict_ndvi(ndvi_model, norm_img)
        
        # Rest of the endpoint remains the same...
        # Visualization image as PNG
        vis_img_bytes = create_visualization(norm_img, pred_ndvi)
        vis_img_bytes.seek(0)
        
        # NDVI band as .npy
        ndvi_bytes = BytesIO()
        np.save(ndvi_bytes, pred_ndvi)
        ndvi_bytes.seek(0)
        
        # Create a ZIP containing both files
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zip_file:
            zip_file.writestr("ndvi_image.png", vis_img_bytes.read())
            ndvi_bytes.seek(0)
            zip_file.writestr("ndvi_band.npy", ndvi_bytes.read())
        
        zip_buf.seek(0)
        return StreamingResponse(
            zip_buf,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=ndvi_output.zip"}
        )
    except Exception as e:
        logger.error(f"Error in predict_ndvi_api: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict_yolo/")
async def predict_yolo_api(file: UploadFile = File(...)):
    """Predict YOLO results from 4-channel TIFF image"""
    if yolo_model is None:
        return JSONResponse(status_code=500, content={"error": "YOLO model not loaded"})
    
    try:
        # Save uploaded file temporarily with proper extension
        file_extension = '.tiff' if file.filename and file.filename.lower().endswith(('.tif', '.tiff')) else '.tiff'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file.flush()  # Ensure data is written
            tmp_file_path = tmp_file.name
        
        try:
            # Verify the file was written correctly
            if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
                raise ValueError("Failed to create temporary file")
            
            logger.info(f"Processing YOLO prediction for file: {file.filename}, temp path: {tmp_file_path}")
            
            # Additional validation: check if file has 4 channels
            try:
                import tifffile
                test_array = tifffile.imread(tmp_file_path)
                if len(test_array.shape) == 3:
                    if test_array.shape[0] == 4 or test_array.shape[2] == 4:
                        channels = 4
                    else:
                        channels = test_array.shape[0] if test_array.shape[0] <= 4 else test_array.shape[2]
                else:
                    channels = 1
                
                if channels != 4:
                    raise ValueError(f"YOLO model expects 4-channel images, but uploaded file has {channels} channels")
                    
            except Exception as validation_error:
                logger.warning(f"Could not validate channels: {validation_error}")
            
            # Predict using YOLO model
            results = predict_yolo(yolo_model, tmp_file_path)
            
            # Convert results to JSON-serializable format
            results_dict = {
                "boxes": {
                    "xyxyn": results.boxes.xyxyn.tolist() if results.boxes is not None else None,
                    "conf": results.boxes.conf.tolist() if results.boxes is not None else None,
                    "cls": results.boxes.cls.tolist() if results.boxes is not None else None
                },
                "classes": results.boxes.cls.tolist() if results.boxes is not None else None,
                "names": results.names,
                "orig_shape": results.orig_shape,
                "speed": results.speed,
                "masks": {
                    "data": results.masks.data.tolist() if results.masks is not None else None,
                    "orig_shape": results.masks.orig_shape if results.masks is not None else None,
                    "xy": [seg.tolist() for seg in results.masks.xy] if results.masks is not None else None,
                    "xyn": [seg.tolist() for seg in results.masks.xyn] if results.masks is not None else None
                }
            }

            # Handle growth stages if present in the results
            if hasattr(results, 'boxes') and results.boxes is not None:
                if hasattr(results.boxes, 'data') and len(results.boxes.data) > 0:
                    # Check if there are additional columns for growth stages
                    if results.boxes.data.shape[1] > 6:
                        growth_stages = results.boxes.data[:, 6:].tolist()
                        results_dict["growth_stages"] = growth_stages
            
            logger.info(f"YOLO prediction completed successfully")
            return JSONResponse(content=results_dict)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error in predict_yolo_api: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_pipeline/")
async def predict_pipeline_api(file: UploadFile = File(...)):
    """Full pipeline: RGB -> NDVI -> 32-bit 4-channel TIFF (RGB+NDVI) -> YOLO prediction"""
    if ndvi_model is None or yolo_model is None:
        return JSONResponse(status_code=500, content={"error": "Models not loaded properly"})
    
    try:
        logger.info(f"Starting full pipeline for file: {file.filename}")
        
        # Read uploaded RGB image
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from uploaded file")
        
        # Convert to PIL Image and then to numpy array
        img = Image.open(BytesIO(contents)).convert("RGB")
        rgb_array = np.array(img)
        logger.info(f"Converted to RGB array with shape: {rgb_array.shape}")
        
        # Run the full pipeline (now includes resizing internally)
        results = predict_pipeline(ndvi_model, yolo_model, rgb_array)
        logger.info("Pipeline processing completed successfully")
        
        # Convert results to JSON-serializable format
        results_dict = {
            "boxes": {
                "xyxyn": results.boxes.xyxyn.tolist() if results.boxes is not None else None,
                "conf": results.boxes.conf.tolist() if results.boxes is not None else None,
                "cls": results.boxes.cls.tolist() if results.boxes is not None else None
            },
            "classes": results.boxes.cls.tolist() if results.boxes is not None else None,
            "names": results.names,
            "orig_shape": results.orig_shape,
            "speed": results.speed,
            "masks": {
                "data": results.masks.data.tolist() if results.masks is not None else None,
                "orig_shape": results.masks.orig_shape if results.masks is not None else None,
                "xy": [seg.tolist() for seg in results.masks.xy] if results.masks is not None else None,
                "xyn": [seg.tolist() for seg in results.masks.xyn] if results.masks is not None else None
            }
        }

        # Handle growth stages if present in the results
        if hasattr(results, 'boxes') and results.boxes is not None:
            if hasattr(results.boxes, 'data') and len(results.boxes.data) > 0:
                # Check if there are additional columns for growth stages
                if results.boxes.data.shape[1] > 6:
                    growth_stages = results.boxes.data[:, 6:].tolist()
                    results_dict["growth_stages"] = growth_stages
        
        logger.info(f"Pipeline prediction completed successfully with {len(results_dict['boxes']['xyxyn']) if results_dict['boxes']['xyxyn'] else 0} detections")
        return JSONResponse(content=results_dict)
        
    except Exception as e:
        logger.error(f"Error in predict_pipeline_api: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# New endpoints to add to your FastAPI app
from yolo_predictor import predict_yolo_with_image, predict_pipeline_with_image, pil_image_to_bytes

@app.post("/predict_yolo_image/")
async def predict_yolo_image_api(file: UploadFile = File(...)):
    """Predict YOLO results from 4-channel TIFF image and return annotated image"""
    if yolo_model is None:
        return JSONResponse(status_code=500, content={"error": "YOLO model not loaded"})
    
    try:
        # Save uploaded file temporarily with proper extension
        file_extension = '.tiff' if file.filename and file.filename.lower().endswith(('.tif', '.tiff')) else '.tiff'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file.flush()  # Ensure data is written
            tmp_file_path = tmp_file.name
        
        try:
            # Verify the file was written correctly
            if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
                raise ValueError("Failed to create temporary file")
            
            logger.info(f"Processing YOLO prediction with image output for file: {file.filename}, temp path: {tmp_file_path}")
            
            # Additional validation: check if file has 4 channels
            try:
                import tifffile
                test_array = tifffile.imread(tmp_file_path)
                if len(test_array.shape) == 3:
                    if test_array.shape[0] == 4 or test_array.shape[2] == 4:
                        channels = 4
                    else:
                        channels = test_array.shape[0] if test_array.shape[0] <= 4 else test_array.shape[2]
                else:
                    channels = 1
                
                if channels != 4:
                    raise ValueError(f"YOLO model expects 4-channel images, but uploaded file has {channels} channels")
                    
            except Exception as validation_error:
                logger.warning(f"Could not validate channels: {validation_error}")
            
            # Predict using YOLO model and get annotated image
            annotated_image = predict_yolo_with_image(yolo_model, tmp_file_path)
            
            # Convert PIL Image to bytes for response
            img_bytes = pil_image_to_bytes(annotated_image, format='PNG')
            
            logger.info(f"YOLO prediction with image output completed successfully")
            
            return StreamingResponse(
                img_bytes,
                media_type="image/png",
                headers={"Content-Disposition": f"attachment; filename=yolo_annotated_{file.filename}.png"}
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error in predict_yolo_image_api: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_pipeline_image/")
async def predict_pipeline_image_api(file: UploadFile = File(...)):
    """Full pipeline with image output: RGB -> NDVI -> 32-bit 4-channel TIFF (RGB+NDVI) -> YOLO prediction -> Annotated Image"""
    if ndvi_model is None or yolo_model is None:
        return JSONResponse(status_code=500, content={"error": "Models not loaded properly"})
    
    try:
        logger.info(f"Starting full pipeline with image output for file: {file.filename}")
        
        # Read uploaded RGB image
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from uploaded file")
        
        # Convert to PIL Image and then to numpy array
        img = Image.open(BytesIO(contents)).convert("RGB")
        rgb_array = np.array(img)
        logger.info(f"Converted to RGB array with shape: {rgb_array.shape}")
        
        # Run the full pipeline with image output (includes resizing internally)
        annotated_image = predict_pipeline_with_image(ndvi_model, yolo_model, rgb_array)
        logger.info("Pipeline processing with image output completed successfully")
        
        # Convert PIL Image to bytes for response
        img_bytes = pil_image_to_bytes(annotated_image, format='PNG')
        
        logger.info(f"Pipeline prediction with image output completed successfully")
        
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=pipeline_annotated_{file.filename}.png"}
        )
        
    except Exception as e:
        logger.error(f"Error in predict_pipeline_image_api: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})