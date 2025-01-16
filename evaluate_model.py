import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model and scaler
model = tf.keras.models.load_model('models/voice_confidence_model_final.keras')
scaler = joblib.load('models/scaler.gz')

# Function to extract features (use the same function as in your main training script)
def extract_features(file_path):
    import librosa
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        if audio.size == 0:
            return None
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Failed to process audio file: {e}")
        return None

# Load test data
def load_test_data(directory='Voicedata'):
    categories = ['confident', 'Non-confident']
    features, labels = [], []
    for label, category in enumerate(categories):
        path = os.path.join(directory, category)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Load and preprocess test data
features_test, labels_test = load_test_data('Voicedata')
features_test = scaler.transform(features_test)  # Scale features using the loaded scaler
features_test = features_test.reshape(-1, 99, 1)  # Reshape to match model input

# Perform predictions and print results
predictions = model.predict(features_test)
for i, prediction in enumerate(predictions):
    true_label = labels_test[i]
    predicted_confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)  # Get the predicted class
    print(f"True Label: {true_label}, Predicted Class: {predicted_class}, Predicted Confidence: {predicted_confidence:.2f}")
