import os
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
MENTAL_MODEL_PATH = Path(os.getenv("MENTAL_MODEL_PATH", MODELS_DIR / "mental_health_model_v219.keras"))
_model = None


def load_mental_model():
    global _model
    if _model is None:
        if not MENTAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Mental health model not found at {MENTAL_MODEL_PATH}")
        _model = tf.keras.models.load_model(str(MENTAL_MODEL_PATH), compile=False)
    return _model


def predict_mental(text: str) -> dict:
    model = load_mental_model()
    try:
        if hasattr(model.input, "dtype") and model.input.dtype == tf.string:
            input_data = tf.constant([text], dtype=tf.string)
        else:
            input_data = np.array([text], dtype=object)
    except Exception:
        input_data = np.array([text], dtype=object)

    predictions = model.predict(input_data, verbose=0)
    result = predictions.tolist()
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return {
        "model_path": str(MENTAL_MODEL_PATH),
        "input": text,
        "predictions": result,
        "output_shape": list(np.asarray(predictions).shape),
    }
