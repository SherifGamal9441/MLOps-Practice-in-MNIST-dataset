from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    prediction = model.predict(input_data)
    predicted_class = int(np.argmax(prediction))
    return JSONResponse(content={"prediction": predicted_class})
