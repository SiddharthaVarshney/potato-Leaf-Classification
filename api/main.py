from fastapi import FastAPI, File, UploadFile
import uvicorn

import numpy as np
import tensorflow as tf
from PIL import Image # to read image files
from io import BytesIO # to mess with bytes

MODEL = tf.keras.models.load_model("/saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

def read_file_as_image(byte_data) -> np.ndarray: # converts byte data(scanned image) to numpy array
    return np.array(Image.open(BytesIO(byte_data)))

@app.get("/ping")
async def ping():
    return "Server is alive!!"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0) # bcz model takes input as batch
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")