import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Define NotEqual as a proper custom Keras layer
@tf.keras.utils.register_keras_serializable()
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.not_equal(x, y)

    def get_config(self):
        config = super(NotEqual, self).get_config()
        return config


def load_artifacts(model_dir="artifacts"):
    """Load model, tokenizer, and metadata from artifacts folder."""

    # Paths
    model_path = os.path.join(model_dir, "seq2seq.h5")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    metadata_path = os.path.join(model_dir, "mta.json")

    # Load model with custom object scope
    model = load_model(model_path, custom_objects={"NotEqual": NotEqual})

    # Load tokenizer (JSON)
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)

    # Load metadata (JSON)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, tokenizer, metadata


def greedy_decode(model, tokenizer, metadata, input_text):
    """Greedy decoding for chatbot response generation."""

    max_len = metadata.get("max_len", 32)

    # Convert input text → sequence
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    # Predict
    preds = model.predict(seq)
    preds = np.argmax(preds, axis=-1)

    # Convert token IDs → words
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    tokens = []
    for idx in preds[0]:
        if idx == 0:  # skip padding
            continue
        tokens.append(index_word.get(idx, ""))

    return " ".join(tokens).strip()
