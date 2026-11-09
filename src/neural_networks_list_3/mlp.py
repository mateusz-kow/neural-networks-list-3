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
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLP:
        n_samples = X.shape[0]
        y_col = y.reshape(-1, 1)
        rng = np.random.default_rng(42)

        self.history = {
            "loss": [],
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

            full_out = X
            for layer in self.layers:
                full_out = layer.forward(full_out)
            full_p = sigmoid(full_out)

            loss = compute_loss(y_col, full_p)
            y_pred = (full_p >= 0.5).astype(int)

            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy_score(y, y_pred))
            self.history["f1"].append(f1_score(y, y_pred, zero_division=0))
            self.history["precision"].append(precision_score(y, y_pred, zero_division=0))
            self.history["recall"].append(recall_score(y, y_pred, zero_division=0))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return sigmoid(out)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
