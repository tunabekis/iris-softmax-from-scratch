"""Training loop shared by the cross-validation stage and the final model sweep."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.model import cross_entropy_loss, init_parameters, predict, regularization_penalty, softmax


def train_softmax_regression(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_features: int,
    num_classes: int,
    learning_rate: float,
    epochs: int,
    reg_type: Optional[str] = None,
    reg_strength: float = 0.0,
    eval_datasets: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    """Train softmax regression via manual gradient descent.

    Logs the pure (unregularized) cross-entropy loss every epoch for the training
    set and, if provided, for each named tensor pair in `eval_datasets` (e.g.
    validation/test), so training and evaluation curves stay directly comparable.
    """
    W, b = init_parameters(num_features, num_classes)
    history = {"train": []}
    if eval_datasets:
        history.update({name: [] for name in eval_datasets})

    for _ in range(epochs):
        logits = torch.matmul(X_train, W) + b
        probs = softmax(logits)

        pure_loss = cross_entropy_loss(probs, y_train)
        history["train"].append(pure_loss.item())

        opt_loss = pure_loss.clone()
        if reg_type is not None:
            opt_loss = opt_loss + regularization_penalty(W, reg_type, reg_strength)

        opt_loss.backward()
        with torch.no_grad():
            W -= learning_rate * W.grad
            b -= learning_rate * b.grad
            W.grad.zero_()
            b.grad.zero_()

        if eval_datasets:
            with torch.no_grad():
                for name, (X_eval, y_eval) in eval_datasets.items():
                    eval_logits = torch.matmul(X_eval, W) + b
                    history[name].append(cross_entropy_loss(softmax(eval_logits), y_eval).item())

    return W.detach(), b.detach(), history


def cross_validate_degree(
    X_train_scaled: np.ndarray,
    y_train: torch.Tensor,
    candidate_degrees,
    num_classes: int,
    folds: int,
    cv_learning_rate: float,
    cv_epochs: int,
    seed: int,
    poly_feature_fn,
) -> Tuple[int, Dict[int, float]]:
    """Pick the polynomial degree with the highest mean k-fold CV accuracy."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    cv_scores: Dict[int, float] = {}
    best_degree, best_acc = candidate_degrees[0], -1.0

    for degree in candidate_degrees:
        X_poly = poly_feature_fn(X_train_scaled, degree)
        fold_accuracies = []

        for train_idx, val_idx in kf.split(X_poly):
            X_fold_train, X_fold_val = X_poly[train_idx], X_poly[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            W, b, _ = train_softmax_regression(
                X_fold_train, y_fold_train, X_poly.shape[1], num_classes, cv_learning_rate, cv_epochs
            )
            y_pred = predict(X_fold_val, W, b)
            fold_accuracies.append(accuracy_score(y_fold_val, y_pred))

        cv_scores[degree] = float(np.mean(fold_accuracies))
        if cv_scores[degree] > best_acc:
            best_acc = cv_scores[degree]
            best_degree = degree

    return best_degree, cv_scores
