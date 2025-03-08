import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import glob
from google.colab import drive
drive.mount('/content/drive')
def audio_to_spectrogram(file_path, img_size=(128, 128)):
    """
    Converts an audio file into a Mel spectrogram image.
    """
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save Spectrogram as Image
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")
    plt.savefig("temp_spectrogram.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Load and Resize Image for CNN
    img = cv2.imread("temp_spectrogram.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    return img
def load_data(dataset_path):
    """
    Loads the dataset, converts audio to spectrograms, and prepares labels.
    """
    X, y = [], []
    emotion_labels = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fear": 4, "disgust": 5, "surprise": 6, "neutral": 7}

    for emotion in emotion_labels.keys():
        file_list = glob.glob(os.path.join(dataset_path, emotion, "*.wav"))

        # Print the number of files found for debugging
        print(f"Found {len(file_list)} files for emotion: {emotion}")

        # Check if any files were found for the current emotion
        if not file_list:
            print(f"Warning: No .wav files found for emotion '{emotion}' in '{dataset_path}'")
            continue  # Skip to the next emotion if no files are found

        for file in file_list:
            img = audio_to_spectrogram(file)
            X.append(img)
            y.append(emotion_labels[emotion])

    X = np.array(X).reshape(-1, 128, 128, 1)  # Reshape for CNN input
    y = to_categorical(y, num_classes=8)  # One-hot encode labels
    return X, y
from tensorflow.keras import Input

def build_cnn_lstm_model(input_shape=(128, 128, 1)):
    """
    Builds a CNN + LSTM model for Speech Emotion Recognition.
    """
    inputs = Input(shape=input_shape)

    # CNN Layers to Extract Key Sound Features
    x = layers.Conv2D(64, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Flattening and Reshaping for LSTM
    x = layers.Flatten()(x)

    # Dynamically reshape based on extracted features
    x = layers.Reshape((-1, 256))(x)

    # LSTM Layer to Capture Time-Series Patterns
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # Fully Connected (Dense) Layer to Classify Emotion
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(8, activation='softmax')(x)  # 8 Emotion Classes

    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# Load Dataset
X, y = load_data("/content/drive/MyDrive/tess")

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_cnn_lstm_model()
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
