import os
import glob
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Adjusted path because your script is inside E:\speechtotext\
dataset_path = "dataset"

# Get all subfolders (commands)
commands = [folder for folder in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, folder)) and not folder.startswith('_')]

print("Detected command labels:", commands)

# Get all WAV file paths and labels
file_paths = []
labels = []

for label in commands:
    folder_path = os.path.join(dataset_path, label)
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    file_paths.extend(wav_files)
    labels.extend([label] * len(wav_files))

print(f"Total samples: {len(file_paths)}")


# Parameters
sample_rate = 16000
duration = 1  # 1 second
samples_per_track = sample_rate * duration

# Feature extraction
mfcc_features = []
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

for path in file_paths:
    signal, sr = librosa.load(path, sr=sample_rate)
    
    # Pad or truncate to fixed length
    if len(signal) < samples_per_track:
        signal = np.pad(signal, (0, samples_per_track - len(signal)))
    else:
        signal = signal[:samples_per_track]
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfcc = mfcc.T  # Transpose so shape is (time, features)
    
    mfcc_features.append(mfcc)

# Convert list to numpy array
X = np.array(mfcc_features)
y = np.array(encoded_labels)

print("Feature shape (X):", X.shape)
print("Labels shape (y):", y.shape)
import matplotlib.pyplot as plt
import random

# Pick a random sample
idx = random.randint(0, len(file_paths) - 1)
signal, sr = librosa.load(file_paths[idx], sr=sample_rate)
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

# Plot waveform
plt.figure(figsize=(10, 4))
plt.title(f"Waveform - Label: {labels[idx]}")
plt.plot(signal)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Plot MFCCs
plt.figure(figsize=(10, 4))
plt.title(f"MFCC - Label: {labels[idx]}")
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.tight_layout()
plt.show()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Reshape X for CNN input (samples, height, width, channels)
X = X[..., np.newaxis]  # From (64721, 32, 13) to (64721, 32, 13, 1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 13, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, 
          epochs=50, 
          validation_data=(X_test, y_test), 
          callbacks=[early_stop])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to label indices

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')  # macro = treats all classes equally
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(f"✅ Precision: {prec*100:.2f}%")
print(f"✅ Recall: {rec*100:.2f}%")
print(f"✅ F1 Score: {f1*100:.2f}%")

# Optional: See detailed class-wise report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# Save the model to a file
try:
    save_path = os.path.join(os.getcwd(), "speech_to_text_model.h5")
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

except Exception as e:
    print("❌ Error saving model:", e)

from tensorflow.keras.models import load_model
import numpy as np
import librosa

# Define the labels (command words)
labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 
          'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
          'tree', 'two', 'up', 'wow', 'yes', 'zero']

# Load the trained model
model = load_model('speech_to_text_model.h5')

# Function to preprocess audio
def preprocess_audio(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16 kHz
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extract MFCCs with keyword arguments
    
    # Pad or truncate the MFCC array to 32 time steps
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]  # Keep the first 32 time steps

    # Add an additional dimension for channels and batch size
    mfcc = mfcc.T  # Shape becomes (32, 13) (time_steps, n_mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Shape becomes (32, 13, 1)
    mfcc = np.expand_dims(mfcc, axis=0)  # Shape becomes (1, 32, 13, 1)
    
    return mfcc

# Path to the audio file you want to predict
audio_path = 'myaudio.wav'  # Replace with your audio file path

# Preprocess the audio file
mfcc = preprocess_audio(audio_path)

# Predict the label (command) from the audio features
prediction = model.predict(mfcc)

# Get the index of the predicted label
predicted_index = np.argmax(prediction)

# Map the index to the actual label
predicted_label = labels[predicted_index]

# Print the predicted command
print(f"Predicted Command: {predicted_label}")


