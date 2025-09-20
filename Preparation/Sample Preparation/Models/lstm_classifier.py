"""Simple LSTM-based classifier using Keras (TensorFlow backend).

This is a lightweight wrapper; training deep models requires an appropriate amount of data
and GPU acceleration for reasonable runtimes. This class expects input X shaped (N, T, F)
where T is time steps and F is features per timestep. For our pipeline we typically have
precomputed features per fixed-length segment, so LSTM use may require reshaping.
"""
from typing import Optional
import numpy as np

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    Sequential = None


class LSTMClassifier:
    def __init__(self, input_shape: Optional[tuple] = None, n_classes: int = 2, lr: float = 1e-3):
        if Sequential is None:
            raise RuntimeError("TensorFlow/Keras not available in the environment. Install tensorflow to use LSTMClassifier.")
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.model = None

    def build(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=self.input_shape, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(optimizer=Adam(self.lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32, **kwargs):
        if self.model is None:
            if self.input_shape is None:
                # infer shape
                self.input_shape = X.shape[1:]
            self.build()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
