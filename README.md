# speechrecognition
 Speech Recognition Timer Control
This project is a simple yet powerful voice-controlled timer system using speech recognition. The user can start and stop the timer using spoken commands — "go" to start the timer and "stop" to halt it. It utilizes machine learning techniques to recognize these commands from audio input and performs real-time control.

🎯 Project Objective
To demonstrate how voice commands can control time-based operations using machine learning and speech recognition. This project eliminates the need for physical input to start or stop the timer, making it useful in hands-free or accessibility scenarios.

📂 Dataset Used
Source: Google Speech Commands Dataset

Used Classes:

go.wav – to start the timer

stop.wav – to stop the timer

Format: 1-second .wav audio clips for training a speech recognition model

⚙️ Technologies Used
Language: Python

Libraries:

Librosa – for audio feature extraction (MFCC)

NumPy, Pandas – for data handling

TensorFlow/Keras – for training the voice command classification model

Tkinter or command line – for simple timer UI

Matplotlib – for model accuracy/loss plotting

🔧 How It Works
Preprocessing:

Extracts MFCC features from go and stop commands.

Builds a labeled dataset.

Model Training:

A CNN model classifies audio into "go" or "stop".

Evaluated for accuracy and loss.

Real-time Recognition:

Microphone input is classified.

If "go" is detected, the timer starts.

If "stop" is detected, the timer halts and displays elapsed time.

🖥️ How to Run
Install requirements:

bash

Train or load the model:

bash
CopyEdit

Run the application:

bash
CopyEdit

✅ Features
Start/Stop timer with voice commands

No manual input required

Works in real time with a microphone

Easy to train and test

🚧 Limitations
Only recognizes two commands: “go” and “stop”

Best performance in a noise-free environment

Short command duration (1 sec)

🚀 Future Enhancements
Add support for pause and reset commands

Improve accuracy in noisy environments

Add GUI with real-time feedback

👩‍💻 Developer
Bharathi A – Project documentation support
BOOMIKA B


📚 References
Google Speech Commands Dataset

Librosa Documentation

Keras Audio Classification Tutoria
