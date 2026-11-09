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

![](plots/part1_layers_loss.png)

Hidden size: Mała (4)                  | Loss: 0.4914 | Acc: 0.869 | Prec: 0.833 | Rec: 0.893 | F1: 0.862

Hidden size: Równa cechom (25)         | Loss: 0.4389 | Acc: 0.885 | Prec: 0.889 | Rec: 0.857 | F1: 0.873

Hidden size: Duża (128)                | Loss: 0.3681 | Acc: 0.869 | Prec: 0.833 | Rec: 0.893 | F1: 0.862

W pierwszych epokach najlepiej radzą sobię modele z małą ilością wymiarów warstwy ukrytej. Natoamist przy większej ilości epok przewagę zyskują modele z większą wymiarowością warstwy ukrytej.

#### Wpływ normalizacji danych

![](plots/part1_normalization_loss.png)

Dane: Znormalizowane       | Loss: 0.4982 | Acc: 0.885 | Prec: 0.862 | Rec: 0.893 | F1: 0.877

Dane: Nieznormalizowane    | Loss: 0.6295 | Acc: 0.738 | Prec: 0.700 | Rec: 0.750 | F1: 0.724

Dla danych nieznormalizowanych model ma na początku bardzo duży loss, który jednak szybko maleje i nawet jest lepszy niż dla danych znormalizowanych. Natomaist w długim terminie model trenowany na danych znormalizowanych okazuje się być lepszy.

#### Wpływ głębokości sieci (liczba warstw ukrytych)

![](plots/part1_hidden_size_loss.png)

Arch: 1 warstwa ukryta [16]        | Loss: 0.4982 | Acc: 0.885 | Prec: 0.862 | Rec: 0.893 | F1: 0.877

Arch: 2 warstwy ukryte [16, 16]    | Loss: 0.4874 | Acc: 0.787 | Prec: 0.778 | Rec: 0.750 | F1: 0.764

Arch: 3 warstwy ukryte [16, 16, 16] | Loss: 0.3938 | Acc: 0.820 | Prec: 0.743 | Rec: 0.929 | F1: 0.825

Zwiększenie liczby warst ukrytych wpływa pozytwnie na jakość modelu.

#### Wpływ paczkowania

![](plots/part2_batches_loss.png)

Tryb: Pełny zbiór (Batch GD)           | Loss: 0.5970 | Acc: 0.656 | Prec: 0.630 | Rec: 0.607 | F1: 0.618

Tryb: Pojedyncze przykłady (SGD, b=1)  | Loss: 0.0025 | Acc: 0.852 | Prec: 0.806 | Rec: 0.893 | F1: 0.847

Tryb: Paczki (Mini-batch, b=32)        | Loss: 0.3225 | Acc: 0.885 | Prec: 0.839 | Rec: 0.929 | F1: 0.881

Model szybciej uczy się na pojedyńczych przykładach niż na paczkach.

#### Wpływ rozkładu początkowego

![](plots/part2_init_loss.png)

Init: Inicjalizacja He (optymalna)        | Loss: 0.3225 | Acc: 0.885 | Prec: 0.839 | Rec: 0.929 | F1: 0.881

Init: Odchylenie standardowe = 1.0        | Loss: 0.2579 | Acc: 0.787 | Prec: 0.742 | Rec: 0.821 | F1: 0.780

Init: Zepsuta (duże odchylenie std=5.0)   | Loss: 0.0164 | Acc: 0.754 | Prec: 0.710 | Rec: 0.786 | F1: 0.746

Najlpeszy okazał się być rozkład He. Duże odchylenie prowadzi do widocznie słabych wyników.

#### Wpływ optymalizacji przez prosty gradient vs. ADAM

![](plots/part2_optimizer_loss.png)

Optymalizator: Prosty SGD (alpha=0.01)             | Loss: 0.3225 | Acc: 0.885 | Prec: 0.839 | Rec: 0.929 | F1: 0.881

Optymalizator: Optymalizator ADAM (alpha=0.01)     | Loss: 0.0004 | Acc: 0.852 | Prec: 0.788 | Rec: 0.929 | F1: 0.852

Optymalizator ADAM pozwolił algorytmowi na uniknięcię tzw. punktów siodłowych, przez co poradził sobie zdecydowanie lepiej od SGD.

---

Wykorzystanie generatywnej sztucznej inteligencji w ramach zadania ogranicało się głównie do wygenerowania kodu tworzącego wykresy oraz liczącego metryki a także optymalizacji wykorzystania biblioteki numpy.
