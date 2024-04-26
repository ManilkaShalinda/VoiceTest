import librosa
import tensorflow as tf
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from typing import Optional

app = FastAPI()

SAVED_MODEL_PATH = "../model.h5"
SAMPLES_TO_CONSIDER = 22050

class KeywordSpottingService:
    """Singleton class for keyword spotting inference with trained models."""

    model = None
    _mapping = [
        "dataset\\backward",
        "dataset\\bed",
        "dataset\\bird",
        "dataset\\cat",
        "dataset\\dog",
        "dataset\\down",
        "dataset\\eight",
        "dataset\\five",
        "dataset\\follow",
        "dataset\\forward",
        "dataset\\four",
        "dataset\\go",
        "dataset\\happy",
        "dataset\\house",
        "dataset\\learn",
        "dataset\\left",
        "dataset\\nine",
        "dataset\\no",
        "dataset\\off",
        "dataset\\on",
        "dataset\\one",
        "dataset\\right",
        "dataset\\seven",
        "dataset\\six",
        "dataset\\stop",
        "dataset\\three",
        "dataset\\tree",
        "dataset\\two",
        "dataset\\up",
        "dataset\\visual",
        "dataset\\wow",
        "dataset\\yes",
        "dataset\\zero"
    ]
    _instance = None

    def __init__(self):
        self.model = tf.keras.models.load_model(SAVED_MODEL_PATH)

    def predict(self, file_path: str) -> str:
        """Predict keyword from audio file."""
        MFCCs = self.preprocess(file_path)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        return self._mapping[predicted_index]

    def preprocess(self, file_path: str, num_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Extract MFCCs from audio file."""
        signal, sample_rate = librosa.load(file_path)
        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]
        else:
            # Pad the signal with zeros or truncate it to the required length
            shortage = SAMPLES_TO_CONSIDER - len(signal)
            signal = np.pad(signal, (0, shortage), mode='constant')
        MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

kss = KeywordSpottingService()

@app.post("/predict/")
async def predict_keyword(file: UploadFile = File(...)):
    """Endpoint to predict keyword from audio file."""
    file_path = "temp_audio.wav"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        predicted_keyword = kss.predict(file_path)
        return {"predicted_keyword": predicted_keyword}
    finally:
        os.remove(file_path)
