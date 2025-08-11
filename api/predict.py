import os
import warnings
import json
from http.server import BaseHTTPRequestHandler
import cgi

import numpy as np
import librosa
import joblib
import tensorflow as tf
import scipy.stats
import soundfile as sf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PIPELINE_OBJECTS = {}


def load_pipeline_objects(base_path="export/"):
    """
    Loads all the necessary .joblib and .keras files for the pipeline.
    It looks for an 'export' folder inside the same directory as this script.
    """
    print("--- Loading all pipeline objects ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_base_path = os.path.join(script_dir, base_path)
    print(f"Attempting to load from: {absolute_base_path}")
    objects = {}
    files_to_load = {
        "router_model": "router_model.joblib",
        "router_scaler": "router_scaler.joblib",
        "router_encoder": "router_encoder.joblib",
        "specialist_high_model": "specialist_high.keras",
        "specialist_high_scaler": "scaler_h.joblib",
        "specialist_high_encoder": "le_high.joblib",
        "specialist_low_model": "specialist_low.keras",
        "specialist_low_scaler": "scaler_l.joblib",
        "specialist_low_encoder": "le_low.joblib",
    }
    try:
        for name, filename in files_to_load.items():
            full_path = os.path.join(absolute_base_path, filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Required file not found: {full_path}")
            print(f"Loading: {full_path}")
            if filename.endswith(".keras"):
                objects[name] = tf.keras.models.load_model(full_path)
            elif filename.endswith(".joblib"):
                objects[name] = joblib.load(full_path)
        print("--- All pipeline objects loaded successfully! ---")
        return objects
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        return None


def extract_features_detailed(file_path):
    # This is your existing feature extraction function, no changes needed.
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        for row in mfcc_features:
            features.extend(
                [
                    np.mean(row),
                    np.std(row),
                    scipy.stats.skew(row),
                    scipy.stats.kurtosis(row),
                ]
            )
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for row in chroma:
            features.extend([np.mean(row), np.std(row)])
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        for row in mel_db:
            features.extend([np.mean(row), np.std(row)])
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for row in contrast:
            features.extend([np.mean(row), np.std(row)])
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for row in tonnetz:
            features.extend([np.mean(row), np.std(row)])
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        rmse = librosa.feature.rms(y=y)
        features.extend([np.mean(rmse), np.std(rmse)])
        return np.array(features)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None


# --- Load models and run warm-up ONCE when the serverless function starts ---
PIPELINE_OBJECTS = load_pipeline_objects()
if PIPELINE_OBJECTS:
    print("--- Running warm-up call to compile audio functions ---")
    try:
        sr_warmup = 22050
        y_warmup = np.zeros(sr_warmup, dtype=np.float32)
        # Use /tmp directory for writable space in serverless environments
        warmup_file = "/tmp/warmup_silent.wav"
        sf.write(warmup_file, y_warmup, sr_warmup)
        extract_features_detailed(warmup_file)
        os.remove(warmup_file)
        print("--- Warm-up complete, application is ready. ---")
    except Exception as e:
        print(f"An error occurred during warm-up: {e}")


# --- Vercel Serverless Handler ---
class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        # This is where your Flask route logic now lives.
        try:
            # Check if models were loaded correctly
            if not PIPELINE_OBJECTS:
                raise RuntimeError("Pipeline models not loaded. Check server logs.")

            # Parse the multipart form data to get the file
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"],
                },
            )
            if "file" not in form:
                raise ValueError("No audio file provided in the 'file' field.")

            audio_file = form["file"]

            # Save the file temporarily
            temp_file_path = "/tmp/temp_audio.wav"
            with open(temp_file_path, "wb") as f:
                f.write(audio_file.file.read())

            # --- Your existing prediction logic ---
            features = extract_features_detailed(temp_file_path)
            os.remove(temp_file_path)
            if features is None:
                raise ValueError("Could not extract features from the audio file.")

            features_2d = features.reshape(1, -1)

            router_scaler = PIPELINE_OBJECTS["router_scaler"]
            features_scaled_router = router_scaler.transform(features_2d)
            router_model = PIPELINE_OBJECTS["router_model"]
            energy_prediction_index = router_model.predict(features_scaled_router)[0]
            router_encoder = PIPELINE_OBJECTS["router_encoder"]
            predicted_energy = router_encoder.classes_[energy_prediction_index]

            if predicted_energy == "high":
                scaler = PIPELINE_OBJECTS["specialist_high_scaler"]
                model = PIPELINE_OBJECTS["specialist_high_model"]
                encoder = PIPELINE_OBJECTS["specialist_high_encoder"]
            else:
                scaler = PIPELINE_OBJECTS["specialist_low_scaler"]
                model = PIPELINE_OBJECTS["specialist_low_model"]
                encoder = PIPELINE_OBJECTS["specialist_low_encoder"]

            features_scaled_specialist = scaler.transform(features_2d)
            features_reshaped = features_scaled_specialist.reshape(
                (1, features_scaled_specialist.shape[1], 1)
            )
            prediction_probabilities = model.predict(features_reshaped)[0]
            predicted_class_index = np.argmax(prediction_probabilities)
            confidence = np.max(prediction_probabilities)
            final_emotion = encoder.classes_[predicted_class_index]

            response_data = {
                "predicted_energy": predicted_energy,
                "predicted_emotion": final_emotion,
                "confidence": f"{confidence * 100:.2f}%",
            }

            # Send successful HTTP response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        except Exception as e:
            # Handle any errors and send an error response
            error_data = {"error": str(e)}
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode("utf-8"))

        return
