# Breast Cancer Detection AI
## Overview

This project is an AI-powered breast cancer detection system that uses a Convolutional Neural Network (CNN) to classify breast tumor images as either benign or malignant. The model is deployed using a Flask API for easy integration and accessibility.

## Features
* Trained CNN model for breast cancer classification
* Flask API for receiving image inputs and returning predictions
* Image preprocessing for grayscale and RGB formats
* Confidence-based classification with detailed probability explanations
* RESTful API with endpoints for health check and prediction

## Technologies Used
 * Python
 * TensorFlow/Keras
 * Fask
 * Num py
 * Pillow (PIL) for image processing

 ## Dataset
 The model is trained using a dataset of breast tumor images. Images are resized to 299x299 and normalized for improved model accuracy.

 ## Installation
 1. Clone the repository:
 `git clone https://github.com/eldrige/Breast-cancer-detector-API.git`

 2. Install dependencies:
 `pip install tensorflow flask numpy pillow`
 3. Ensure you have the dataset placed in the correct directory: ./dataset/test/
 4. Train the model:
 `python model.py`
 5. Run the Flask API:
 `python main.py`
 ## Model Architecture
 The CNN consists of:
 * 2 Convolutional layers (32 and 64 filters, 3x3 kernel)
 * 2 MaxPooling layers (2x2 pool size)
 * Flattening layer
 * Dense hidden layer (128 neurons, ReLU activation)
 * Output layer (1 neuron, Sigmoid activation for binary classification)

 ## API Endpoints
 Predict Tumor Classification
 * Endpoint: /predict
 * Method: POST
 * Request: Send an image file as multipart/form-data with key image
 * Response:
 `{
  "prediction": "0.85",
  "description": "High confidence that the tumor is malignant; strong evidence supporting malignancy.",
  "classification": "Malignant"
}`
## How it works
1. The model receives an image and preprocesses it (resizing, color conversion, normalization).
2. The image is passed through the trained CNN model for classification.
3. The probability score determines the classification and confidence level.
4. The API returns the prediction results as JSON.

## Results and Performance
* Training epochs: 10
* Loss function: Binary Crossentropy
* Optimizer: Adam
* Expected accuracy: ~90% (depending on dataset quality)

## Future Improvements
* Enhance dataset size and diversity for better accuracy
* Deploy model to cloud services for scalability
*  Implement a mobile-friendly interface






