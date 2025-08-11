# 🎤 Speech Emotion Recognition using LSTM

This project aims to recognize human emotions from speech using a Long Short-Term Memory (LSTM) based deep learning model. By analyzing audio signals, the system can classify emotions such as happiness, sadness, anger, fear, and more. It features a simple Tkinter-based GUI for easy voice input and live emotion prediction.

---

## 📌 Features

- 🎙️ Real-time voice recording via React frontend
- 📃 Single Page Application for all three different input methods
- 🤖 Emotion prediction using LSTM neural networks, lightbgm classifiers
- 📊 Trained on Three different dataset with ~70% accuracy

---

## 🛠️ Technologies Used

- Python
- TensorFlow + Keras
- Librosa
- Pandas, NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Jupyter Notebook (for development)
- FlaskAPI
- ReactJS

---

## 🗃️ Dataset

- We use the **[RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)** (Ryerson Audio-Visual Database of Emotional Speech and Song), which contains audio recordings of actors expressing various emotions.

- For Hindi SER we have used **[Dataset by Vishal Bhardwaj](https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi?select=my+Dataset)**, which contains audio recorings in Hindi language and different Emotions

- For Indian Accent English we have used **[Public TTS dataset](https://github.com/skit-ai/emotion-tts-dataset)**, which contains audio recordings for different emotions in Indian Accent english

Note: The structure for all of these datasets was modified for easier importing in code.

---

## 🚀 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KavyaJP/SGP-I.git
   cd SGP-I
   ```

### 1. Train the Model yourself (optional, trained model and encodings are already saved in backend)

Note: Do this only if you want to train the model on custom data

1. Change Directory
   ```bash
   cd Training
   ```

2. **Install Dependancies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run each cells Jupyter Notebook:**
   Ensure that you update the audio input path when loading the data, and specify the path to your recorded audio file when predicting the emotion in your audio.

### 2. Backend

Note: if you trained the model on custom data then save the model and move the saved .keras and .pkl file and replace the already existing .keras and .pkl file

1. Change Directory
   ```bash
   cd backend
   ```

2. **Install Dependancies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend:**
   ```bash
   python app.py
   ```

Note: keep the backend running so that frontend can send data to the backend and backend can send a response to the frontend

### 3. Frontend

1. Now open up another terminal while keeping the backend terminal running and then

2. Change Directory
   ```bash
   cd frontend
   ```

3. **Install Dependancies:**
   ```bash
   npm install
   ```

4. **Run the backend:**
   ```bash
   npm run dev
   ```

---

## Contributors

- Kavya Prajapati - [E-Mail](mailto:kavya31052005@gmail.com)
- Aarya Shah - [E-Mail](mailto:shahaarya465@gmail.com)
- Vansh Mehta - [E-Mail](mailto:vansh161976@gmail.com)
