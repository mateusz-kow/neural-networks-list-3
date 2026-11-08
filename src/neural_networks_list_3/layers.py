import numpy as np


class Linear:
    """Warstwa liniowa Wx + b wspierająca inicjalizację wag oraz optymalizatory SGD i ADAM."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = "he",
        random_state: int = 42,
    ):
        rng = np.random.default_rng(random_state)

        # Wybór metody inicjalizacji wag (Część 1 & Część 2)
        if init_method == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init_method == "normal_001":
            scale = 0.01
        elif init_method == "std_1":
            scale = 1.0
        elif init_method == "broken_large":
            scale = 5.0
        else:
            raise ValueError(f"Nieznana metoda inicjalizacji: {init_method}")

        self.W = rng.normal(loc=0.0, scale=scale, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        self.cache_x: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # Statystyki dla optymalizatora ADAM
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0  # Licznik kroków dla korekcji biasu

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # dL/dW = X^T * dL/dY
        self.dW = np.dot(self.cache_x.T, grad_output)
        # dL/db = sum(dL/dY)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # dL/dX = dL/dY * W^T
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def update(
        self,
        alpha: float,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        if optimizer == "sgd":
            self.W -= alpha * self.dW
            self.b -= alpha * self.db

        elif optimizer == "adam":
            self.t += 1
            # Aktualizacja momentów dla wag W
            self.m_W = beta1 * self.m_W + (1.0 - beta1) * self.dW
            self.v_W = beta2 * self.v_W + (1.0 - beta2) * (self.dW**2)

            # Korekcja biasu w m i v
            m_W_corrected = self.m_W / (1.0 - beta1**self.t)
            v_W_corrected = self.v_W / (1.0 - beta2**self.t)

            self.W -= alpha * m_W_corrected / (np.sqrt(v_W_corrected) + eps)

            # Aktualizacja momentów dla biasów b
            self.m_b = beta1 * self.m_b + (1.0 - beta1) * self.db
            self.v_b = beta2 * self.v_b + (1.0 - beta2) * (self.db**2)

            m_b_corrected = self.m_b / (1.0 - beta1**self.t)
            v_b_corrected = self.v_b / (1.0 - beta2**self.t)

            self.b -= alpha * m_b_corrected / (np.sqrt(v_b_corrected) + eps)


class ReLU:
    """Funkcja aktywacji ReLU."""

    def __init__(self):
        self.cache_x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_x = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.cache_x > 0)
