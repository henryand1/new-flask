# Import necessary modules
# import firebase_admin
from flask import Flask, request, jsonify, json
# from firebase_admin import credentials, firestore
from PIL import Image
import io
import requests
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load Firebase service account credentials

# Load the model
model_url = "https://storage.googleapis.com/myplant_storage/Model_1.h5"
model_response = requests.get(model_url)
model_file_path = "model.h5"
with open(model_file_path, "wb") as f:
    f.write(model_response.content)
model = load_model(model_file_path)

# Load the penyakit data
with open('myPlant-json/penyakit.json') as json_file:
    data = json.load(json_file)

# Define routes and endpoints
@app.route('/', methods=['GET'])
def welcome():
    return "Response Success!"

@app.route('/penyakit', methods=['GET'])
def pagePenyakit():
    filtered_data = [{"nama": penyakit['nama'], "deskripsi": penyakit['deskripsi']} for penyakit in data]
    return jsonify(filtered_data), 200

@app.route('/penyakit/<string:penyakit>', methods=['GET'])
def namaPenyakit(penyakit):
    penyakit_id = penyakit
    for penyakit_data in data:
        if penyakit_data['id'] == penyakit_id:
            return jsonify(penyakit_data), 200
    return jsonify({'message': 'Penyakit tidak ditemukan!'}), 400

def load_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((224, 224))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "image_url" not in request.args:
        return jsonify({"error": "no image_url"})

    image_url = request.args.get("image_url")

    try:
        new_image = load_image_from_url(image_url)
        prediction_labels = [
            "Apple Scab",
            "Apple Black Rot",
            "Apple Cedar rust",
            "Apple Healthy",
            "Corn Cercospora Leaf Spot | Gray Leaf Spot",
            "Corn Common Rust",
            "Corn Northern Leaf Blight",
            "Corn Healthy",
            "Grape Black Rot",
            "Grape Esca | Black Measles",
            "Grape Leaf Blight | Isariopsis Leaf Spot",
            "Grape Healthy",
            "Potato Early Blight",
            "Potato Late Blight",
            "Potato Healthy",
            "Strawberry Leaf Scorch",
            "Strawberry Healthy"
        ]

        prediction = np.argmax(model.predict(new_image)[0])
        result = prediction_labels[prediction]

        data = {"prediction": result}

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

# Remove the model file after using
os.remove(model_file_path)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)