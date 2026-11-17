import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .layers import Linear, ReLU


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class MLP:
    """Wielowarstwowa sieć neuronowa z obsługą SGD, ADAM, batchingu i metod inicjalizacji."""

    def __init__(
        self,
        layer_dims: list[int],
        alpha: float = 0.01,
        epochs: int = 100,
        batch_size: int | None = 32,
        optimizer: str = "sgd",
        init_method: str = "he",
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.layers = []

        for i in range(len(layer_dims) - 2):
            self.layers.append(
                Linear(
                    layer_dims[i],
                    layer_dims[i + 1],
                    init_method=init_method,
                    random_state=random_state + i,
                )
            )
            self.layers.append(ReLU())

        self.layers.append(
            Linear(
                layer_dims[-2],
                layer_dims[-1],
                init_method=init_method,
                random_state=random_state + len(layer_dims),
            )
        )

        self.history = {
            "loss": [],
            "test_loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> "MLP":
        """
        Trenuje model na X/y.

        Jeśli podano X_eval i y_eval, po każdej epoce oblicza również loss
        na tym zbiorze. Dane ewaluacyjne nie uczestniczą w propagacji wstecznej
        ani aktualizacji wag.
        """
        if (X_eval is None) != (y_eval is None):
            raise ValueError("X_eval i y_eval muszą być przekazane razem.")

        n_samples = X.shape[0]
        y_col = y.reshape(-1, 1)
        y_eval_col = y_eval.reshape(-1, 1) if y_eval is not None else None
        rng = np.random.default_rng(42)

        self.history = {
            "loss": [],
            "test_loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        for _ in range(self.epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_col[indices]

            if self.batch_size is None:
                batches = [(X_shuffled, y_shuffled)]
            else:
                batches = [
                    (
                        X_shuffled[i : i + self.batch_size],
                        y_shuffled[i : i + self.batch_size],
                    )
                    for i in range(0, n_samples, self.batch_size)
                ]

            for X_batch, y_batch in batches:
                b_size = X_batch.shape[0]

                out = X_batch
                for layer in self.layers:
                    out = layer.forward(out)

                p = sigmoid(out)

                grad_output = (p - y_batch) / b_size
                curr_grad = grad_output
                for layer in reversed(self.layers):
                    curr_grad = layer.backward(curr_grad)

                for layer in self.layers:
                    if isinstance(layer, Linear):
                        layer.update(self.alpha, optimizer=self.optimizer)

            # Loss i metryki na zbiorze treningowym.
            full_out = X
            for layer in self.layers:
                full_out = layer.forward(full_out)
            full_p = sigmoid(full_out)

            train_loss = compute_loss(y_col, full_p)
            y_pred = (full_p >= 0.5).astype(int)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(accuracy_score(y, y_pred))
            self.history["f1"].append(f1_score(y, y_pred, zero_division=0))
            self.history["precision"].append(precision_score(y, y_pred, zero_division=0))
            self.history["recall"].append(recall_score(y, y_pred, zero_division=0))

            # Ewaluacja na zbiorze testowym: bez backpropagation i aktualizacji wag.
            if X_eval is not None and y_eval_col is not None:
                eval_proba = self.predict_proba(X_eval)
                eval_loss = compute_loss(y_eval_col, eval_proba)
                self.history["test_loss"].append(eval_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return sigmoid(out)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
