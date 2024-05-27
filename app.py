import os
import tempfile
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input
import numpy as np
from PIL import Image  # Ensure Pillow is imported

app = Flask(__name__)

# Load the model
model = load_model('my_model.h5')

# Preprocess the image


def preprocess_image(image_path):
    try:
        # Ensure the size matches the input size of your model
        image = load_img(image_path, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0  # Normalize to [0,1] range
        return image
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {e}")
        return None


@app.route('/')
def home():
    return "Welcome to the Image Classification API. Use the /predict endpoint to get predictions."


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                file_path = tmp.name
                file.save(file_path)

            image = preprocess_image(file_path)
            if image is None:
                return jsonify({"error": "Error preprocessing image"}), 500

            prediction = model.predict(image)

            os.remove(file_path)  # Clean up the file

            return jsonify({"prediction": prediction.tolist()}), 200
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return jsonify({"error": "Error during prediction"}), 500

    return jsonify({"error": "File processing error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
