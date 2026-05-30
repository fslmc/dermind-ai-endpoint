import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
SKIN_MODEL_PATH = Path(os.getenv("SKIN_MODEL_PATH", MODELS_DIR / "best_skin_classifier_full.keras"))
_model = None


def load_skin_model():
    global _model
    if _model is None:
        if not SKIN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Skin model not found at {SKIN_MODEL_PATH}")
        _model = tf.keras.models.load_model(str(SKIN_MODEL_PATH), compile=False)
    return _model


def _resolve_target_size(model):
    shape = model.input_shape
    if not shape or len(shape) < 3:
        return 224, 224

    if len(shape) == 4:
        height, width = shape[1], shape[2]
    elif len(shape) == 3:
        height, width = shape[0], shape[1]
    else:
        return 224, 224

    if height is None or width is None:
        return 224, 224
    return int(height), int(width)


def _preprocess_image(image_bytes: bytes, target_size: tuple[int, int]):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    array = np.asarray(image).astype("float32") / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    batch = np.expand_dims(array, axis=0)
    return batch


def predict_skin(image_bytes: bytes) -> dict:
    model = load_skin_model()
    target_size = _resolve_target_size(model)
    batch = _preprocess_image(image_bytes, target_size)
    predictions = model.predict(batch, verbose=0)
    result = predictions.tolist()
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return {
        "model_path": str(SKIN_MODEL_PATH),
        "predictions": result,
        "output_shape": list(np.asarray(predictions).shape),
    }
