from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Laad je model (zorg ervoor dat het pad naar je opgeslagen model klopt)
model_path = '/Users/vince/School - Datalab IV/Datalab-IV/berenklauw_model.h5'
model = tf.keras.models.load_model(model_path)

async def preprocess_image(image: UploadFile):
    try:
        logging.info("Starting image preprocessing")
        contents = await image.read()
        logging.info(f"Image size: {len(contents)} bytes")
        image = load_img(BytesIO(contents))
        logging.info(f"Loaded image with size: {image.size}")
        image = img_to_array(image) / 255.0
        logging.info("Converted image to array")
        return image
    except Exception as e:
        logging.error(f"Error in preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

def split_image(image, patch_size=256):
    patches = []
    h, w, _ = image.shape
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
    h, w, c = image_shape
    reconstructed_image = np.zeros((h, w, 1))
    patch_index = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = patches[patch_index]
            patch_h, patch_w, _ = patch.shape
            if i + patch_h > h or j + patch_w > w:
                patch = patch[:h-i, :w-j, :]  # Adjust the patch size to fit the image dimensions
            reconstructed_image[i:i+patch_h, j:j+patch_w, :] = patch
            patch_index += 1
    return reconstructed_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info("Received file for prediction")
        # Preprocess de afbeelding
        image = await preprocess_image(file)
        original_shape = image.shape
        logging.info(f"Original image shape: {original_shape}")
        
        # Snijd de afbeelding in stukken van 256x256
        patches = split_image(image, patch_size=256)
        logging.info(f"Image split into {len(patches)} patches")
        
        # Voer voorspellingen uit voor elk stuk
        patches_array = np.array(patches)
        pred_patches = model.predict(patches_array)
        pred_patches_thresholded = (pred_patches > 0.5).astype(np.uint8)
        logging.info("Prediction completed")
        
        # Voeg de voorspelde stukken samen tot één volledige afbeelding
        reconstructed_mask = merge_patches(pred_patches_thresholded, original_shape, patch_size=256)
        logging.info("Patches merged into full image")
        
        # Retourneer de voorspelling als JSON
        return JSONResponse(content={"prediction": reconstructed_mask.tolist()})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)