import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = tf.keras.models.load_model("best_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

def predict_milk(img_path):

    try:
        # Load image
        img = image.load_img(img_path, target_size=IMG_SIZE)

        # Convert image to array
        img = image.img_to_array(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Preprocess image
        img = preprocess_input(img)

        # Predict
        prediction = model.predict(img)

        # Get class index
        class_index = int(np.argmax(prediction))

        # Get class name
        predicted_class = class_names[str(class_index)]

        # Confidence
        confidence = float(np.max(prediction)) * 100

        print("\n==========================")
        print("Prediction :", predicted_class)
        print("Confidence : {:.2f}%".format(confidence))
        print("==========================")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":

    path = input("Enter image path: ").strip().replace('"', '')

    predict_milk(path)