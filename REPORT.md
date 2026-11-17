# SIECI NEURONOWE – ćwiczenie

Implementacja propagacji wstecznej na zbiorze Heart Disease

Repozytorium: [https://github.com/mateusz-kow/neural-networks-list-3](https://github.com/mateusz-kow/neural-networks-list-3)

Autor: Mateusz Kowalski 280543

---

### Wyniki eksperymentów

#### Wpływ wymiarowości warstwy ukrytej

![](plots/part1_hidden_size_loss.png)

Hidden size: Mała (4)                  | Test Loss: 0.5859 | Acc: 0.885 | Prec: 0.957 | Rec: 0.786 | F1: 0.863

Hidden size: Równa cechom (25)         | Test Loss: 0.4825 | Acc: 0.885 | Prec: 0.889 | Rec: 0.857 | F1: 0.873

Hidden size: Duża (128)                | Test Loss: 0.3539 | Acc: 0.902 | Prec: 0.893 | Rec: 0.893 | F1: 0.893

W pierwszych epokach najlepiej radzą sobię modele z małą ilością wymiarów warstwy ukrytej. Natoamist przy większej ilości epok przewagę zyskują modele z większą wymiarowością warstwy ukrytej.

#### Wpływ normalizacji danych

![](plots/part1_normalization_loss.png)

Dane: Znormalizowane       | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Dane: Nieznormalizowane    | Test Loss: 0.5500 | Acc: 0.721 | Prec: 0.704 | Rec: 0.679 | F1: 0.691

Dla danych nieznormalizowanych model ma na początku bardzo duży loss, który jednak szybko maleje i nawet jest lepszy niż dla danych znormalizowanych. Natomaist wykres straty dla danych znormalizowanych jest znacznie mniej zaszumiony.

#### Wpływ głębokości sieci (liczba warstw ukrytych)

![](plots/part1_layers_loss.png)

Arch: 1 warstwa ukryta [16]        | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Arch: 2 warstwy ukryte [16, 16]    | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

Arch: 3 warstwy ukryte [16, 16, 16] | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

Zwiększenie liczby warst ukrytych wpływa negatywnie na jakość modelu.

#### Wpływ paczkowania

![](plots/part2_batches_loss.png)

Tryb: Pełny zbiór (Batch GD)           | Test Loss: 0.6791 | Acc: 0.574 | Prec: 0.542 | Rec: 0.464 | F1: 0.500

Tryb: Pojedyncze przykłady (SGD, b=1)  | Test Loss: 1.1517 | Acc: 0.852 | Prec: 0.806 | Rec: 0.893 | F1: 0.847

Tryb: Paczki (Mini-batch, b=32)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Model szybciej uczy się na pojedyńczych przykładach niż na paczkach, ale wtedy jest też bardziej podatny na overfitting.

#### Wpływ rozkładu początkowego

![](plots/part2_init_loss.png)

Init: Inicjalizacja He (optymalna)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Init: Odchylenie standardowe = 1.0        | Test Loss: 0.8317 | Acc: 0.787 | Prec: 0.742 | Rec: 0.821 | F1: 0.780

Init: Zepsuta (duże odchylenie std=5.0)   | Test Loss: 5.8577 | Acc: 0.803 | Prec: 0.767 | Rec: 0.821 | F1: 0.793

Najlpeszy okazał się być rozkład He. Duże odchylenie prowadzi do widocznie słabych wyników.

#### Wpływ optymalizacji przez prosty gradient vs. ADAM

![](plots/part2_optimizer_loss.png)

Optymalizator: Prosty SGD (alpha=0.01)             | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Optymalizator: Optymalizator ADAM (alpha=0.01)     | Test Loss: 0.9795 | Acc: 0.852 | Prec: 0.788 | Rec: 0.929 | F1: 0.852

Optymalizator ADAM pozwolił algorytmowi na uniknięcię tzw. punktów siodłowych, przez co uczył się zdecydowanie szybciej od SGD. Szybko jednak ta metoda omptymalizacji doprowadziła do overfittingu modelu.

---

Wykorzystanie generatywnej sztucznej inteligencji w ramach zadania ogranicało się głównie do wygenerowania kodu tworzącego wykresy oraz liczącego metryki a także optymalizacji wykorzystania biblioteki numpy.
