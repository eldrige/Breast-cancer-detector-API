from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('breast_cancer_model.h5')

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((299, 299))  # Resize to match model input
    
    # Check if the image is grayscale (mode 'L')
    if image.mode == 'L':
        # Convert grayscale to RGB by duplicating the single channel
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # Handle other modes (like RGBA, CMYK, etc.) by converting to RGB
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    
    # Final check to ensure we have the right shape
    if image.shape != (299, 299, 3):
        raise ValueError(f"Processed image has shape {image.shape}, expected (299, 299, 3)")
    
    return image


def get_prediction_description(probability):
    if 0.0 <= probability < 0.1:
        return "Very low confidence that the tumor is malignant; strong belief it is benign.", "Benign"
    elif 0.1 <= probability < 0.3:
        return "Low confidence that the tumor is malignant; more likely to be benign than malignant.", "Benign"
    elif 0.3 <= probability < 0.5:
        return "Uncertain; weak evidence for malignancy; could be either benign or malignant.", "Uncertain"
    elif 0.5 <= probability < 0.7:
        return "Moderate confidence that the tumor is malignant; more belief it is malignant than benign.", "Malignant"
    elif 0.7 <= probability < 0.9:
        return "High confidence that the tumor is malignant; strong evidence supporting malignancy.", "Malignant"
    elif 0.9 <= probability <= 1.0:
        return "Very high confidence that the tumor is malignant; almost certain it is malignant.", "Malignant"
    else:
        raise ValueError(f"Invalid probability value: {probability}")

@app.route('/', methods=['GET'])
def return_hello():
    
    return jsonify({'Greetings': 'Hello World from cancer detector!'})    

# API endpoint to predict malignancy
@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.files['image'].read()
    image = preprocess_image(image_data)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(image.shape)
    # Make prediction
    prediction = model.predict(image)
    probability = prediction[0][0]
    # result = "Malignant" if prediction[0][1] > prediction[0][0] else "Benign"
    # print(prediction)
    description, classification = get_prediction_description(probability)
    
    return jsonify({
        'prediction': str(probability),
        'description': description,
        'classification': classification
    })

if __name__ == '__main__':
    app.run(debug=True)
