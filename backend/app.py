import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import librosa

app = Flask(__name__)

# Load the trained model & encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ser_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
enc = joblib.load(ENCODER_PATH)


# Feature extraction (matches original training)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Keep original sample rate

        features = []

        # MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        features.append(mfccs)

        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        features.append(chroma)

        return np.concatenate(features)

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


emotion_classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "ps",
    "sad",
    "surprise",
]


@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files["file"]

    # Save and preprocess
    file_path = "temp.wav"
    audio_file.save(file_path)
    features = extract_features(file_path)  # should return shape (40,)
    features = np.expand_dims(features, axis=(0, -1))  # shape (1, 40, 1)

    # Predict
    prediction = model.predict(features)  # shape (1, 8)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_classes[predicted_class]

    return jsonify({"emotion": predicted_emotion})


if __name__ == "__main__":
    app.run(debug=True)
