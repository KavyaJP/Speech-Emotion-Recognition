# 🎤 Speech Emotion Recognition using LSTM

This project aims to recognize human emotions from speech using a Long Short-Term Memory (LSTM) based deep learning model. By analyzing audio signals, the system can classify emotions such as happiness, sadness, anger, fear, and more. It features a simple Tkinter-based GUI for easy voice input and live emotion prediction.

---

## 📌 Features

- 🎙️ Real-time voice recording via GUI
- 🤖 Emotion prediction using LSTM neural networks
- 📊 Trained on RAVDESS dataset with ~82% accuracy
- 🧠 Uses MFCC, Chroma, and Mel Spectrogram features
- 🪟 Simple, desktop-based interface built with Tkinter

---

## 🛠️ Technologies Used

- Python
- TensorFlow + Keras
- Librosa
- Pandas, NumPy
- Scikit-learn
- Tkinter
- Joblib
- Matplotlib
- Jupyter Notebook (for development)

---

## 🗃️ Dataset

We use the **[RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)** (Ryerson Audio-Visual Database of Emotional Speech and Song), which contains audio recordings of actors expressing various emotions.

---

## 🚀 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KavyaJP/SGP-I.git
   cd SGP-I
   ```
2. **Install Dependancies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Record the Audio:**
   ```bash
   python record.py
   ```
4. **Run each cells Jupyter Notebook:**
   Ensure that you update the audio input path when loading the data, and specify the path to your recorded audio file when predicting the emotion in your audio.

---

## Contributors

- 23AIML056 - Kavya Prajapati - kavya31052005@gmail.com
- 23AIML064 - Aarya Shah - shahaarya465@gmail.com
- 23AIML074 - Vansh Mehta - vansh161976@gmail.com
