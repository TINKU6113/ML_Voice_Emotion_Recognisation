import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.utils import to_categorical
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set dataset path in Google Drive
DATASET_PATH = "/content/drive/MyDrive/ravdess/ravdess"

# Define emotion labels in RAVDESS dataset
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
def extract_features(file_path, max_pad_len=200):
    """
    Extract MFCC features from an audio file.

    Args:
        file_path (str): Path to the .wav audio file.
        max_pad_len (int): Maximum padding length for MFCC features.

    Returns:
        np.array: MFCC feature array of shape (40, max_pad_len)
    """
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Pad or trim MFCCs to fixed length
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs
X, y = [], []

# Walk through dataset directory
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = file.split("-")[2]  # Extract emotion code from filename

            if emotion in emotion_labels:
                X.append(extract_features(file_path))  # Extract features
                y.append(int(emotion) - 1)  # Convert emotion label to integer (0-based)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=len(emotion_labels))

# Reshape input for CNN (add channel dimension)
X = X[..., np.newaxis]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed
import numpy as np

# Fix input shape for CNN
X = np.expand_dims(X, axis=-1)  # Ensure (samples, time_steps, frequency_bins, 1)

# Define CNN-LSTM model
model = Sequential()

# CNN layers wrapped in TimeDistributed
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding="same"),
                          input_shape=(X.shape[1], X.shape[2], X.shape[3], 1)))  # Add missing dimension
model.add(TimeDistributed(MaxPooling2D((2, 1))))  # ✅ Fix: Pool only along time axis
model.add(TimeDistributed(Flatten()))

# LSTM layers
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))

# Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(len(emotion_labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
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
