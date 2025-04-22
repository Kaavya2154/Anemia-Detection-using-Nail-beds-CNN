from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'models/fingernail_anemia_mobilenetv2.keras'
model = load_model(MODEL_PATH)

# Home route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Convert RGBA to RGB and resize
        image = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        prediction = model.predict(img_array)[0][0]  # Assuming binary classification
        result = 'Anemic' if prediction > 0.5 else 'Not Anemic'

        return jsonify({'prediction': result, 'confidence': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
