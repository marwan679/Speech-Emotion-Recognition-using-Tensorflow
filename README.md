English Speech Emotion Recognition (SER)
This repository contains a complete pipeline for English Speech Emotion Recognition (SER) using Deep Learning. The project leverages audio signal processing techniques and Convolutional Neural Networks (CNN) to classify human emotions from vocal recordings.

# 🚀 Features
Data Preprocessing: Audio loading and manipulation using librosa and pydub.

Data Augmentation: Robustness is improved by adding random noise and pitch shifting.

Feature Extraction: Utilization of Mel-frequency cepstral coefficients (MFCCs) to represent audio in a format suitable for deep learning.

Deep Learning Model: A Keras-based model (CNN/LSTM potential) designed to recognize 8 distinct emotions.

Visualization: Integrated audio wave plotting and spectrogram generation.

# 📊 Dataset
The model is trained on the RAVDESS (Radford Emotional Speech and Song) dataset, which includes 24 professional actors (12 female, 12 male) vocalizing in the following emotional states:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

# 🛠 Installation
Prerequisites
Python 3.10+
ffmpeg (required for pydub and audio processing)

Setup
Clone the repository:

```Bash
git clone https://github.com/your-username/english-speech-emotion-recognition.git
cd english-speech-emotion-recognition
```
Install dependencies:

```Bash
pip install numpy matplotlib librosa tensorflow keras pandas sklearn tqdm colorama art pydub playsound==1.2.2
```
# 🏗 Model Architecture
The project utilizes a Sequential model built with Keras. Key layers include:

Conv1D/Conv2D Layers: For spatial/temporal feature extraction from MFCCs.

Batch Normalization: To ensure stable training and faster convergence.

Dropout: To prevent overfitting.

Dense Layers: For classification into the 8 emotion categories.

# 💻 Usage
1. Data Collection
Ensure your dataset is zipped as speech-emotion-recognition-ravdess-data.zip. The notebook includes a utility function to unzip and load the paths:

```Python
# Unzip and load data paths
unzip_file('speech-emotion-recognition-ravdess-data.zip')
emotions, pathes = load_data('speech-emotion-recognition-ravdess-data/')
```
2. Visualization
You can visualize any audio file in the dataset using the built-in display tool:

```Python
# Displays audio wave, spectrogram, and identifies the emotion label
display(10)
```
3. Feature Extraction
Features are extracted using 20 MFCC bands, resized to a consistent shape for the model:

```Python
features = get_features(audio_path)
```
4. Real-time Monitoring
The project includes a monitor function to run detection on a specific file:

```Python
def monitor(path: str):
    print(f"Current emotion is: {emotion_detector(path)}")
    print(f"Gender is: {detect_gender(path)}")
```
# 📈 Results
The model analyzes the nuances in pitch, intensity, and timbre to predict emotions. Detailed performance metrics (Accuracy/Loss curves and Confusion Matrices) are generated within the notebook during the training phase.
