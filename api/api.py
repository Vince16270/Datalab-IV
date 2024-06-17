# Description: This script defines the FastAPI app and the endpoint for image segmentation.

# Import the required libraries
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
from model import load_model, predict

# Build the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model()

def split_image(image, patch_size=256):
    """
    Splits an image into smaller patches of a specified size.

    Args:
        image (numpy.array): The input image to be split.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of image patches.
    """
    patches = []
    h, w, c = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            patches.append(patch)
    return patches

def merge_patches(patches, image_shape, patch_size=256):
    """
    Merges image patches back into a single image.

    Args:
        patches (list): A list of image patches.
        image_shape (tuple): The original shape of the image.
        patch_size (int): The size of each patch.

    Returns:
        numpy.array: The reconstructed image.
    """
    h, w, c = image_shape
    reconstructed_image = np.zeros((h, w, c))
    patch_index = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = patches[patch_index]
            if len(patch.shape) == 2:
                patch = np.expand_dims(patch, axis=-1)
            if patch.shape[-1] != c:
                patch = np.repeat(patch, c, axis=-1)
            patch_h, patch_w, patch_c = patch.shape
            if i + patch_h > h:
                patch_h = h - i
            if j + patch_w > w:
                patch_w = w - j
            reconstructed_image[i:i+patch_h, j:j+patch_w, :] = patch[:patch_h, :patch_w, :]
            patch_index += 1
    return reconstructed_image

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Handles image upload and prediction request.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing the original image, 
                      the prediction mask, and the overlay image in hex format.
    """
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) / 255.0
    
    if image.shape[-1] == 1:
        image = np.expand_dims(image, axis=-1)
    
    # Split the image into patches
    patches = split_image(image, patch_size=256)
    patches_array = np.array(patches)
    
    # Predict the segmentation mask for each patch
    pred_patches = model.predict(patches_array)
    pred_patches_thresholded = (pred_patches > 0.3).astype(np.float32)
    
    pred_patches_thresholded = [np.squeeze(patch, axis=-1) if patch.shape[-1] == 1 else patch for patch in pred_patches_thresholded]
    
    # Merge the predicted patches back into a full mask
    reconstructed_mask = merge_patches(pred_patches_thresholded, image.shape, patch_size=256)
    
    # Create an overlay of the original image and the mask
    overlay = cv2.addWeighted(image, 0.7, reconstructed_mask, 0.3, 0)
    
    # Encode images to PNG format
    _, buffer_image = cv2.imencode('.png', (image * 255).astype(np.uint8))
    _, buffer_prediction = cv2.imencode('.png', (reconstructed_mask * 255).astype(np.uint8))
    _, buffer_overlay = cv2.imencode('.png', (overlay * 255).astype(np.uint8))
    
    # Return the encoded images in hex format
    return JSONResponse(content={
        "original_image": buffer_image.tobytes().hex(),
        "prediction_image": buffer_prediction.tobytes().hex(),
        "overlay_image": buffer_overlay.tobytes().hex()
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#http://127.0.0.1:8000/docs