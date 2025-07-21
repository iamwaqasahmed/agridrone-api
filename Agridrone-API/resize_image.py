from PIL import Image
import numpy as np

# Alternative: Simple resize function using PIL directly
def resize_image_simple(image_array, target_size):
    """
    Simple resize function using PIL
    
    Args:
        image_array: Input image array (H, W, C)
        target_size: Tuple (height, width)
    
    Returns:
        Resized image array
    """
    # Ensure image is in correct format
    if image_array.max() <= 1:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(image_array)
    
    # Resize (PIL uses width, height format)
    resized_pil = image_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert back to numpy array and normalize back to [0, 1]
    resized_array = np.array(resized_pil).astype(np.float32) / 255.0
    
    return resized_array

def resize_image_optimized(image_array, target_size):
    """
    Resize image to target size with memory optimization
    
    Args:
        image_array: Input image array (H, W, C)
        target_size: Tuple (height, width) representing target dimensions
    
    Returns:
        Resized image array
    """
    # Convert numpy array to PIL Image
    if image_array.dtype != np.uint8:
        # Convert to uint8 if not already
        if image_array.max() <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    image_pil = Image.fromarray(image_array)
    
    # Resize image (PIL uses (width, height) format)
    resized_pil = image_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert back to numpy array
    result = np.array(resized_pil)
    
    # Clean up
    image_pil.close()
    resized_pil.close()
    
    return result
