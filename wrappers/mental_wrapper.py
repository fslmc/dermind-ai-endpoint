import os
import pickle
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"

DEFAULT_MAX_LEN = 200

_TOKENIZER_PATHS = [
    Path(os.getenv("MENTAL_TOKENIZER_PATH", ROOT / "tokenizer.pkl")),
    HERE / "tokenizer.pkl",
]
_LABEL_ENCODER_PATHS = [
    Path(os.getenv("MENTAL_LABEL_ENCODER_PATH", ROOT / "label_encoder.pkl")),
    HERE / "label_encoder.pkl",
]
_MODEL_PATHS = [
    Path(os.getenv("MENTAL_MODEL_PATH", MODELS_DIR / "mental_health_model_v219.keras")),
    MODELS_DIR / "mental_health_model.h5",
    ROOT / "mental_health_model.h5",
]

_model = None
_tokenizer = None
_le = None


class _KerasUnpickler(pickle.Unpickler):
    KERAS_MODULE_MAP = {
        "keras.preprocessing.text": "tensorflow.keras.preprocessing.text",
        "keras.preprocessing.sequence": "tensorflow.keras.preprocessing.sequence",
        "keras.preprocessing.image": "tensorflow.keras.preprocessing.image",
        "keras.utils": "tensorflow.keras.utils",
        "keras.layers": "tensorflow.keras.layers",
        "keras.initializers": "tensorflow.keras.initializers",
        "keras.callbacks": "tensorflow.keras.callbacks",
        "keras.models": "tensorflow.keras.models",
    }

    def find_class(self, module, name):
        if module in self.KERAS_MODULE_MAP:
            module = self.KERAS_MODULE_MAP[module]
        elif module.startswith("keras."):
            module = module.replace("keras", "tensorflow.keras", 1)
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as handle:
        return _KerasUnpickler(handle).load()


def _find_existing(paths):
    for path in paths:
        if path and path.exists():
            return path
    return None


TOKENIZER_PATH = _find_existing(_TOKENIZER_PATHS)
LABEL_ENCODER_PATH = _find_existing(_LABEL_ENCODER_PATHS)
MENTAL_MODEL_PATH = _find_existing(_MODEL_PATHS)


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if TOKENIZER_PATH is None:
            raise FileNotFoundError(
                "Tokenizer not found. Create tokenizer.pkl or set MENTAL_TOKENIZER_PATH."
            )
        _tokenizer = _load_pickle(TOKENIZER_PATH)
    return _tokenizer


def load_label_encoder():
    global _le
    if _le is None:
        if LABEL_ENCODER_PATH is None:
            raise FileNotFoundError(
                "Label encoder not found. Create label_encoder.pkl or set MENTAL_LABEL_ENCODER_PATH."
            )
        with open(LABEL_ENCODER_PATH, "rb") as handle:
            _le = pickle.load(handle)
    return _le


# =====================================================================
# CRITICAL FIX: Explicitly register and allow empty defaults for Keras 3
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen=None, vocab_size=None, embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Only initialize internal sublayers if parameters are passed
        if vocab_size is not None and embed_dim is not None:
            self.token_emb = layers.Embedding(vocab_size, embed_dim)
        if maxlen is not None and embed_dim is not None:
            self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[-1])
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=None, num_heads=None, ff_dim=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        if num_heads is not None and embed_dim is not None:
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        if ff_dim is not None and embed_dim is not None:
            self.ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.drop1(self.attn(x, x), training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config


def load_mental_model():
    global _model
    if _model is None:
        if MENTAL_MODEL_PATH is None:
            raise FileNotFoundError(
                "Mental health model not found. Place the model under models/ or set MENTAL_MODEL_PATH."
            )
        # Using the absolute string path, compile=False, and our registered custom structures
        _model = tf.keras.models.load_model(
            str(MENTAL_MODEL_PATH),
            compile=False,
            custom_objects={
                "TransformerBlock": TransformerBlock,
                "PositionalEmbedding": PositionalEmbedding,
            },
        )
    return _model


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def translate_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def predict_mental(text: str) -> dict:
    tokenizer = load_tokenizer()
    le = load_label_encoder()
    model = load_mental_model()

    translated = translate_to_english(text)
    cleaned = clean_text(translated)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=DEFAULT_MAX_LEN,
        padding="post",
        truncating="post",
    )
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    classes = list(le.classes_)

    breakdown = {
        str(cls): round(float(p) * 100, 2) for cls, p in zip(classes, probs)
    }

    return {
        "label": str(classes[idx]),
        "confidence": round(float(probs[idx]) * 100, 2),
        "original_text": text,
        "translated_text": translated,
        "breakdown": breakdown,
    }