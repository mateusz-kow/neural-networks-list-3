import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .data import prepare_data
from .mlp import MLP
from .utils import plot_learning_curves


def main() -> None:
    os.makedirs("plots", exist_ok=True)
    X_train_norm, X_test_norm, y_train, y_test = prepare_data(normalize=True)
    n_features = X_train_norm.shape[1]

    print(f"Liczba cech po przetworzeniu: {n_features}")

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 1: Wymiarowość warstwy ukrytej
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 1: Wymiarowość warstwy ukrytej")
    print("=" * 80)

    hidden_sizes = {
        "Mała (4)": 4,
        f"Równa cechom ({n_features})": n_features,
        "Duża (128)": 128,
    }
    results_hidden = {}

    for label, h_size in hidden_sizes.items():
        model = MLP(
            layer_dims=[n_features, h_size, 1],
            alpha=0.01,
            epochs=300,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_train_norm, y_train)
        results_hidden[label] = model.history

        y_pred = model.predict(X_test_norm)
        print(
            f"Hidden size: {label:25s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_hidden,
        "Wpływ wymiarowości warstwy ukrytej na funkcję kosztu",
        "plots/part1_hidden_size_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 2: Normalizacja danych
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 2: Normalizacja danych")
    print("=" * 80)

    X_train_unnorm, X_test_unnorm, _, _ = prepare_data(normalize=False)
    results_norm = {}

    data_variants = {
        "Znormalizowane": (X_train_norm, X_test_norm),
        "Nieznormalizowane": (X_train_unnorm, X_test_unnorm),
    }

    for label, (X_tr, X_te) in data_variants.items():
        model = MLP(
            layer_dims=[n_features, 16, 1],
            alpha=0.01,
            epochs=300,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_tr, y_train)
        results_norm[label] = model.history

        y_pred = model.predict(X_te)
        print(
            f"Dane: {label:20s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_norm,
        "Wpływ normalizacji danych na zbieżność modelu",
        "plots/part1_normalization_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 3: Różna liczba warstw (Głębokość sieci)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 3: Różna liczba warstw (głębokość sieci)")
    print("=" * 80)

    architecture_variants = {
        "1 warstwa ukryta [16]": [n_features, 16, 1],
        "2 warstwy ukryte [16, 16]": [n_features, 16, 16, 1],
        "3 warstwy ukryte [16, 16, 16]": [n_features, 16, 16, 16, 1],
    }
    results_layers = {}

    for label, arch in architecture_variants.items():
        model = MLP(
            layer_dims=arch,
            alpha=0.01,
            epochs=300,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_train_norm, y_train)
        results_layers[label] = model.history

        y_pred = model.predict(X_test_norm)
        print(
            f"Arch: {label:28s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_layers,
        "Wpływ liczby warstw ukrytych na zbieżność modelu",
        "plots/part1_layers_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 1: Tryby paczkowania danych (Full vs SGD vs Mini-batch)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 1: Tryby paczkowania danych")
    print("=" * 80)

    deep_architecture = [n_features, 16, 16, 1]
    batch_variants = {
        "Pełny zbiór (Batch GD)": None,
        "Pojedyncze przykłady (SGD, b=1)": 1,
        "Paczki (Mini-batch, b=32)": 32,
    }
    results_batches = {}

    for label, b_size in batch_variants.items():
        model = MLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=150,
            batch_size=b_size,
            optimizer="sgd",
            init_method="he",
        )
        model.fit(X_train_norm, y_train)
        results_batches[label] = model.history

        y_pred = model.predict(X_test_norm)
        print(
            f"Tryb: {label:32s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_batches,
        "Porównanie trybów paczkowania danych",
        "plots/part2_batches_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 2: Inicjalizacja wag (Std=1 vs He vs Zepsuta)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 2: Metody inicjalizacji wag")
    print("=" * 80)

    init_variants = {
        "Inicjalizacja He (optymalna)": "he",
        "Odchylenie standardowe = 1.0": "std_1",
        "Zepsuta (duże odchylenie std=5.0)": "broken_large",
    }
    results_init = {}

    for label, init_m in init_variants.items():
        model = MLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=150,
            batch_size=32,
            optimizer="sgd",
            init_method=init_m,
        )
        model.fit(X_train_norm, y_train)
        results_init[label] = model.history

        y_pred = model.predict(X_test_norm)
        print(
            f"Init: {label:35s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_init,
        "Porównanie metod inicjalizacji wag (dla głębszej sieci)",
        "plots/part2_init_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 3: Prosty SGD vs Optymalizator ADAM
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 3: Optymalizacja przez prosty gradient vs ADAM")
    print("=" * 80)

    optimizer_variants = {
        "Prosty SGD (alpha=0.01)": "sgd",
        "Optymalizator ADAM (alpha=0.01)": "adam",
    }
    results_optimizers = {}

    for label, opt in optimizer_variants.items():
        model = MLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=150,
            batch_size=32,
            optimizer=opt,
            init_method="he",
        )
        model.fit(X_train_norm, y_train)
        results_optimizers[label] = model.history

        y_pred = model.predict(X_test_norm)
        print(
            f"Optymalizator: {label:35s} | Loss: {model.history['loss'][-1]:.4f} | "
            f"Acc: {accuracy_score(y_test, y_pred):.3f} | Prec: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
            f"Rec: {recall_score(y_test, y_pred, zero_division=0):.3f} | F1: {f1_score(y_test, y_pred, zero_division=0):.3f}"
        )

    plot_learning_curves(
        results_optimizers,
        "Porównanie optymalizatorów SGD oraz ADAM",
        "plots/part2_optimizer_loss.png",
    )


if __name__ == "__main__":
    main()
