# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import os
import librosa  # For audio processing
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename  # For secure file names
import joblib  # For loading the pre-fitted scaler

# Initialize the Flask application
app = Flask(__name__)

# Define a custom loss function called focal loss, commonly used in classification problems
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32 for computation
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())  # Clip y_pred to prevent log(0)
    cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Compute the cross entropy loss
    weights = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)  # Calculate weights
    focal_loss = weights * cross_entropy  # Apply weights to the loss
    return tf.reduce_sum(focal_loss, axis=-1)  # Sum over all classes

# Register the custom loss function globally so it can be identified by Keras
tf.keras.utils.get_custom_objects()['focal_loss_fixed'] = focal_loss_fixed

# Load the pre-trained model and scaler using TensorFlow and joblib
model = tf.keras.models.load_model(
    'models/voice_confidence_model_final.keras', 
    custom_objects={'focal_loss_fixed': focal_loss_fixed}
)

scaler = joblib.load('models/scaler.gz')  # Load the pre-fitted scaler

# Function to extract audio features using librosa
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)  # Load the audio file
        if audio.size == 0:
            return None  # Return None if audio is empty
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Compute MFCCs
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)  # Compute chroma features
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)  # Compute spectral contrast
        features = np.concatenate([  # Concatenate all features into a single array
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Failed to process audio file: {e}")
        return None

# Define the index route for serving the HTML template
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define the predict route for handling file uploads and returning predictions
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')  # Get the uploaded file
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    secure_path = secure_filename(file.filename)  # Secure the file name
    file.save(secure_path)  # Save the file locally

    features = extract_features(secure_path)  # Extract features from the audio file
    if features is None:
        os.remove(secure_path)  # Remove the file if features extraction failed
        return jsonify({'error': 'Failed to extract features or empty audio file'}), 400

    features = np.array([features])  # Convert features into an array
    features = scaler.transform(features)  # Scale the features using the pre-loaded scaler
    features = features.reshape(1, 99, 1)  # Reshape features to match the model's input shape

    prediction = model.predict(features)  # Predict using the pre-trained model
    predicted_confidence = np.max(prediction, axis=1)[0]  # Extract the maximum prediction value
    confidence_percentage = "{:.2f}%".format(predicted_confidence * 100)  # Format the confidence as a percentage

    os.remove(secure_path)  # Clean up the temporary file
    return jsonify({
        'confidence_level': confidence_percentage
    })

# Run the Flask application with debugging enabled
if __name__ == '__main__':
    app.run(debug=True)
