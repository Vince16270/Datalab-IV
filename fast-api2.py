from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import io
import uvicorn
import time

app = FastAPI()

model_path = '/Users/vince/School - Datalab IV/berenklauw_model.h5'
model = tf.keras.models.load_model(model_path)

def read_imagefile(file: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_option: str = Query("both", enum=["mask", "masked_image", "both"])
):
    try:
        start_time = time.time()
        contents = await file.read()
        image = read_imagefile(contents)
        original_size = image.size
        preprocessed_image = preprocess_image(image)

        prediction_start_time = time.time()
        predictions = model.predict(preprocessed_image)
        print(f"Model prediction took {time.time() - prediction_start_time} seconds")

        predicted_mask = (predictions[0] > 0.5).astype(np.uint8) * 255
        predicted_mask_image = Image.fromarray(predicted_mask.squeeze(), mode='L')
        predicted_mask_image = predicted_mask_image.resize(original_size, Image.NEAREST)

        if return_option == "mask":
            buf = io.BytesIO()
            predicted_mask_image.save(buf, format='PNG')
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        image = image.convert("RGBA")
        mask = Image.new("RGBA", original_size, (255, 0, 0, 0))
        mask_array = np.array(predicted_mask_image)
        mask_array = np.stack((mask_array, mask_array, mask_array, mask_array), axis=-1)
        mask_image = Image.fromarray(np.where(mask_array > 128, [255, 0, 0, 128], [0, 0, 0, 0]).astype(np.uint8))

        combined = Image.alpha_composite(image, mask_image)
        
        if return_option == "masked_image":
            buf = io.BytesIO()
            combined.save(buf, format='PNG')
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        
        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        buf.seek(0)
        combined_output = buf.getvalue()

        buf = io.BytesIO()
        predicted_mask_image.save(buf, format='PNG')
        buf.seek(0)
        mask_output = buf.getvalue()

        return {"mask": mask_output, "masked_image": combined_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


#http://127.0.0.1:8000/docs
