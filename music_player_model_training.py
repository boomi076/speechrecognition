import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import librosa.display
warnings.filterwarnings("ignore")

# Define dataset path
DATA_DIR = 'music_dataset'  # adjust if needed
LABELS = ['down', 'up']

# Data containers
X = []
y = []

# Function to extract and clean audio features
def extract_mfcc(filepath, max_pad_len=130):
    try:
        audio, sr = librosa.load(filepath, sr=16000)  # Resample to 16kHz
        audio, _ = librosa.effects.trim(audio)
        audio = librosa.util.normalize(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = pad_sequences([mfcc.T], maxlen=max_pad_len, padding='post', dtype='float32')
        return mfcc[0]
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Load and process dataset
for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            features = extract_mfcc(file_path)
            if features is not None:
                X.append(features)
                y.append(0 if label == 'down' else 1)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Preprocessing done.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Visualization Functions
def plot_waveform(label):
    path = os.path.join(DATA_DIR, label)
    file = os.listdir(path)[0]
    file_path = os.path.join(path, file)
    signal, sr = librosa.load(file_path, sr=16000)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Waveform - {label}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_mfcc(label):
    path = os.path.join(DATA_DIR, label)
    file = os.listdir(path)[1]
    file_path = os.path.join(path, file)
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC - {label}")
    plt.tight_layout()
    plt.show()

def plot_label_distribution():
    counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in LABELS}
    plt.bar(counts.keys(), counts.values(), color=['red', 'blue'])
    plt.title("Label Distribution")
    plt.ylabel("Count")
    plt.show()

# Run visualizations
for label in LABELS:
    plot_waveform(label)
    plot_mfcc(label)

plot_label_distribution()

# Build the model
model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(130, 13)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # 2 classes: down and up

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Save model
model.save('speech_music_player_model.h5')

# Evaluate model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"\n✅ Validation Loss: {val_loss}")
print(f"✅ Validation Accuracy: {val_accuracy}")

# Predict on validation set
y_pred_probs = model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Separate metrics
report = classification_report(y_val, y_pred_classes, target_names=LABELS, output_dict=True)
for label in LABELS:
    print(f"\n🔍 Class: {label}")
    print(f"Precision: {report[label]['precision']:.2f}")
    print(f"Recall:    {report[label]['recall']:.2f}")
    print(f"F1-Score:  {report[label]['f1-score']:.2f}")

# Full report
print("\n📊 Full Classification Report:")
print(classification_report(y_val, y_pred_classes, target_names=LABELS))

# Test with an external audio file
def test_model_with_audio(file_path):
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 130 - mfcc.shape[1]))), mode='constant')
    mfcc = mfcc[:, :130]
    mfcc = np.transpose(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Test audio prediction
test_file = 'myaudi.wav'  # Replace with actual test file
predicted_class = test_model_with_audio(test_file)
print(f"\n🎧 Predicted class for {test_file}: {LABELS[predicted_class]}")
