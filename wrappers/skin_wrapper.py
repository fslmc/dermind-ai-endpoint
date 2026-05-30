import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import regularizers

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
SKIN_MODEL_PATH = Path(os.getenv("SKIN_MODEL_PATH", MODELS_DIR / "best_skin_classifier_full.keras"))

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Acne & Rosacea", "Infections", "Eczema & Dermatitis"]

_model = None


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        bottleneck_channels = max(1, channels // reduction)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(
            bottleneck_channels,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self.fc2 = tf.keras.layers.Dense(
            channels, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-4)
        )

    def call(self, inputs):
        x = self.global_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, use_se=True, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.use_se = use_se
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        if self.use_se:
            self.se = SqueezeExcitation(filters)
        self.use_shortcut = False
        self.shortcut_conv = None
        self.shortcut_bn = None

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.use_shortcut = True
            self.shortcut_conv = tf.keras.layers.Conv2D(
                self.filters,
                kernel_size=1,
                strides=self.stride,
                use_bias=False,
                kernel_regularizer=regularizers.l2(1e-4),
            )
            self.shortcut_bn = tf.keras.layers.BatchNormalization()
        super(ResBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.use_se:
            x = self.se(x)
        if self.use_shortcut:
            shortcut_x = self.shortcut_conv(inputs)
            shortcut_x = self.shortcut_bn(shortcut_x, training=training)
        else:
            shortcut_x = inputs
        x = tf.keras.layers.add([x, shortcut_x])
        return tf.nn.gelu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "stride": self.stride, "use_se": self.use_se})
        return config


class SkinClassifierNet(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(SkinClassifierNet, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.num_classes = num_classes
        self.blocks = tf.keras.Sequential(
            [
                ResBlock(64, stride=1),
                ResBlock(64, stride=1),
                ResBlock(128, stride=2),
                ResBlock(128, stride=1),
                ResBlock(256, stride=2),
                ResBlock(256, stride=1),
                ResBlock(512, stride=2),
                ResBlock(512, stride=1),
            ]
        )
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(
            num_classes, activation="softmax", kernel_regularizer=regularizers.l2(1e-4)
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.gelu(x)
        x = self.blocks(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


def load_skin_model():
    global _model
    if _model is None:
        if not SKIN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Skin model not found at {SKIN_MODEL_PATH}")
        _model = tf.keras.models.load_model(
            str(SKIN_MODEL_PATH),
            custom_objects={
                "SkinClassifierNet": SkinClassifierNet,
                "ResBlock": ResBlock,
                "SqueezeExcitation": SqueezeExcitation,
            },
            compile=False,
            safe_mode=False,
        )
    return _model


def predict_skin(image_bytes: bytes) -> dict:
    """
    Preprocesses image bytes and runs skin classification inference.
    Returns predicted class, confidence, and per-class probabilities.
    """
    model = load_skin_model()

    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert to numpy and normalize
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # Add batch dimension
    batch = np.expand_dims(image_array, axis=0)

    # Run prediction
    predictions = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))

    # Format results to match inference notebook
    results = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(predictions[predicted_idx]),
        "probabilities": {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))},
        "model_path": str(SKIN_MODEL_PATH),
    }

    return results
