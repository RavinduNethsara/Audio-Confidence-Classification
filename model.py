# Import necessary libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import joblib

# Define a custom focal loss function for handling class imbalance
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # Convert labels to float32
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0) errors in computation
        cross_entropy = -y_true * tf.math.log(y_pred)  # Calculate the cross-entropy component
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)  # Calculate weights
        focal_loss_value = tf.reduce_sum(weight * cross_entropy, axis=-1)  # Apply the weights
        return tf.reduce_mean(focal_loss_value)  # Return the mean loss across the batch
    return focal_loss_fixed

# Function to extract features from audio files with optional data augmentation
def extract_features(file_path, augment=False):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)  # Load the audio file
        if audio.size == 0:
            return None  # Return None if audio is empty
        if augment:
            audio = add_augmentation(audio, sample_rate)  # Apply data augmentation if enabled
        # Extract various types of features from the audio signal
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        return np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1)
        ])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to add noise, pitch shift, and time stretch to audio data for augmentation
def add_augmentation(audio, sample_rate):
    noise_factor = 0.005
    audio += np.random.normal(0, noise_factor, audio.shape)  # Add random noise
    shift_factor = np.random.randint(-2, 3)
    audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=shift_factor)  # Shift pitch
    stretch_rate = np.random.uniform(0.8, 1.2)
    audio = librosa.effects.time_stretch(audio, rate=stretch_rate)  # Stretch time
    return audio

# Function to load and label data from a given directory structure
def load_data(directory='Voicedata'):
    categories = ['confident', 'Non-confident']
    features, labels = [], []
    for label, category in enumerate(categories):
        path = os.path.join(directory, category)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)
    return np.array(features), np.array(labels)

# Load and split data
features, labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Standardize features using sklearn's StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.gz')  # Save the scaler for later use

# Reshape features for input into Conv1D layer
X_train = X_train.reshape((-1, 99, 1))
X_test = X_test.reshape((-1, 99, 1))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model architecture using Keras functional API
input_layer = Input(shape=(99, 1))
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))(x)  # First LSTM layer
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(64, return_sequences=False))(x)  # Second LSTM layer
x = Dense(128, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with custom focal loss and accuracy metric
model.compile(
    optimizer='adam',
    loss=focal_loss(),
    metrics=['accuracy']
)

# Set up model training callbacks
checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the final model
model.save('models/voice_confidence_model_final.keras')
