import argparse
from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    return img_array.reshape(1, 28, 28, 1)

def main():
    parser = argparse.ArgumentParser(description="MNIST Inference")
    parser.add_argument("image_path", type=str, help="Path to image file (e.g. 1.png)")
    parser.add_argument("--model_path", type=str, default="mnist_model.h5", help="Path to saved .h5 model")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    image = preprocess_image(args.image_path)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    print(f"Predicted digit: {predicted_label}")

if __name__ == "__main__":
    main()
