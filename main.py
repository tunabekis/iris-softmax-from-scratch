"""
Manual softmax regression on the Iris dataset.

Pipeline:
  1. Split the data 70/15/15 (train/val/test) and standardize it.
  2. Pick the best polynomial degree (1, 2, or 3) via 3-fold cross-validation.
  3. Train 9 models (3 learning rates x 3 regularization types) at that degree.
  4. Plot train/validation/test loss curves for all 9 models.
  5. Report test-set metrics for the model with the lowest final validation loss.
"""

import torch

from src.config import (
    CANDIDATE_DEGREES,
    CV_EPOCHS,
    CV_FOLDS,
    CV_LEARNING_RATE,
    FINAL_EPOCHS,
    FINAL_LEARNING_RATES,
    NUM_CLASSES,
    RANDOM_SEED,
    REGULARIZATION_STRENGTH,
    REGULARIZATION_TYPES,
)
from src.data import load_and_prepare_data
from src.evaluation import compute_classification_metrics
from src.features import generate_polynomial_features
from src.model import predict
from src.training import cross_validate_degree, train_softmax_regression
from src.visualization import plot_loss_curves


def select_best_degree(data) -> int:
    """Run k-fold CV over candidate polynomial degrees and report the winner."""
    print("Running 3-Fold Cross-Validation on the training dataset")
    best_degree, cv_scores = cross_validate_degree(
        X_train_scaled=data.train.X,
        y_train=data.train.y,
        candidate_degrees=CANDIDATE_DEGREES,
        num_classes=NUM_CLASSES,
        folds=CV_FOLDS,
        cv_learning_rate=CV_LEARNING_RATE,
        cv_epochs=CV_EPOCHS,
        seed=RANDOM_SEED,
        poly_feature_fn=generate_polynomial_features,
    )
    for degree, acc in cv_scores.items():
        print(f"Degree {degree} Mean CV Accuracy: {acc:.4f}")
    print(f"--- Winner: Degree {best_degree} Polynomial ---")
    return best_degree


def train_final_models(data, best_degree: int) -> dict:
    """Train every (learning rate, regularization) combination at the chosen degree."""
    X_train_final = generate_polynomial_features(data.train.X, best_degree)
    X_val_final = generate_polynomial_features(data.val.X, best_degree)
    X_test_final = generate_polynomial_features(data.test.X, best_degree)

    results = {}
    for lr in FINAL_LEARNING_RATES:
        for reg_type in REGULARIZATION_TYPES:
            name = f"{reg_type}_LR_{lr}"
            W, b, history = train_softmax_regression(
                X_train_final,
                data.train.y,
                num_features=X_train_final.shape[1],
                num_classes=NUM_CLASSES,
                learning_rate=lr,
                epochs=FINAL_EPOCHS,
                reg_type=reg_type,
                reg_strength=REGULARIZATION_STRENGTH,
                eval_datasets={"val": (X_val_final, data.val.y), "test": (X_test_final, data.test.y)},
            )
            results[name] = {"history": history, "val_final": history["val"][-1], "W": W, "b": b}

    return results, X_test_final


def report_best_model(results: dict, X_test_final, y_test) -> None:
    """Pick the model with the lowest final validation loss and print test metrics."""
    best_model_name = min(results, key=lambda name: results[name]["val_final"])
    best_model = results[best_model_name]

    y_pred = predict(X_test_final, best_model["W"], best_model["b"])
    metrics = compute_classification_metrics(y_test, y_pred)

    print(f"\n--- Best Model Selection (Lowest Val Error): {best_model_name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")


def main() -> None:
    # A single global seed makes both the sklearn splits/CV folds and the
    # torch weight initializations reproducible across runs.
    torch.manual_seed(RANDOM_SEED)

    data = load_and_prepare_data()

    best_degree = select_best_degree(data)
    results, X_test_final = train_final_models(data, best_degree)

    plot_loss_curves(results)
    report_best_model(results, X_test_final, data.test.y)


if __name__ == "__main__":
    main()
