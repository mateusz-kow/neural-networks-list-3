# SIECI NEURONOWE – ćwiczenie

Implementacja propagacji wstecznej na zbiorze Heart Disease

Repozytorium: [https://github.com/mateusz-kow/neural-networks-list-3](https://github.com/mateusz-kow/neural-networks-list-3)

Autor: Mateusz Kowalski 280543

---

### Implementacja

```python
class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = "he",
        random_state: int = 42,
    ):
        rng = np.random.default_rng(random_state)

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

        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.dW = np.dot(self.cache_x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
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

            self.m_W = beta1 * self.m_W + (1.0 - beta1) * self.dW
            self.v_W = beta2 * self.v_W + (1.0 - beta2) * (self.dW**2)

            m_W_corrected = self.m_W / (1.0 - beta1**self.t)
            v_W_corrected = self.v_W / (1.0 - beta2**self.t)

            self.W -= alpha * m_W_corrected / (np.sqrt(v_W_corrected) + eps)

            self.m_b = beta1 * self.m_b + (1.0 - beta1) * self.db
            self.v_b = beta2 * self.v_b + (1.0 - beta2) * (self.db**2)

            m_b_corrected = self.m_b / (1.0 - beta1**self.t)
            v_b_corrected = self.v_b / (1.0 - beta2**self.t)

            self.b -= alpha * m_b_corrected / (np.sqrt(v_b_corrected) + eps)
```

```python
class ReLU:
    def __init__(self):
        self.cache_x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_x = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.cache_x > 0)
```

### Wyniki eksperymentów

#### Wpływ wymiarowości warstwy ukrytej

![part1_layers_loss](plots/part1_layers_loss.png)

Im większa wyiarowość warstwy ukryte, tym model radzi sobie lepiej.

#### Wpływ normalizacji danych

![part1_normalization_loss](plots/part1_normalization_loss.png)

Model ma lepsze wyniki na danych znormalizowanych po większej ilości epok.

#### Wpływ głębokości sieci (liczba warstw ukrytych)

![part1_hidden_size_loss](plots/part1_hidden_size_loss.png)

Zwiększenie liczby warst ukrytych wpływa pozytwnie na jakość modelu.

#### Wpływ paczkowania

![part2_batches_loss](plots/part2_batches_loss.png)

Model najszybciej uczy się na pojedyńczych przykładach niż na paczkach. Natomiast może to doprowadzić do overfittingu modelu.

#### Wpływ rozkładu początkowego

![part2_init_loss](plots/part2_init_loss.png)

Najlpeszy okazał się być rozkład He.

#### Wpływ optymalizacji przez prosty gradient vs. ADAM

![part2_optimizer_loss](plots/part2_optimizer_loss.png)

Optymalizator ADAM pozwolił algorytmowi na uniknięcię tzw. punktów siodłowych, przez co poradził sobie zdecydowanie lepiej od SGD.

---

Wykorzystanie generatywnej sztucznej inteligencji w ramach zadania ogranicało się głównie do wygenerowania kodu tworzącego wykresy oraz liczącego metryki a także optymalizacji wykorzystania biblioteki numpy.
