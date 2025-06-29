from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps
import io

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "MNIST API is running!"}

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale

    # Invert (if necessary)
    image = ImageOps.invert(image)

    # Resize and center (optional improvement)
    image = image.resize((28, 28))

    # Normalize and reshape
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction))
    return JSONResponse(content={"prediction": predicted_class})
