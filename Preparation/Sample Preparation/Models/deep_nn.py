"""Simple feed-forward deep neural network classifier using Keras.

Wrapper for students to experiment with deeper models. Expects input X shaped (N, F).
"""
from typing import Optional
import numpy as np

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    Sequential = None


class DeepNNClassifier:
    def __init__(self, input_dim: Optional[int] = None, n_classes: int = 2, lr: float = 1e-3):
        if Sequential is None:
            raise RuntimeError("TensorFlow/Keras not available in the environment. Install tensorflow to use DeepNNClassifier.")
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.model = None

    def build(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.input_dim))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(optimizer=Adam(self.lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32, **kwargs):
        if self.model is None:
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            self.build()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
