"""Manually implemented softmax regression building blocks (no nn.Linear/nn.CrossEntropyLoss)."""

import torch


def init_parameters(num_features: int, num_classes: int):
    """Create randomly initialized, trainable weight and bias tensors."""
    W = torch.randn(num_features, num_classes, requires_grad=True)
    b = torch.randn(num_classes, requires_grad=True)
    return W, b


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Softmax with the log-sum-exp stability trick (subtract the row-wise max)."""
    shifted_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted_logits)
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)


def cross_entropy_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the target class."""
    return -torch.mean(torch.log(probs[range(len(targets)), targets] + 1e-9))


def regularization_penalty(W: torch.Tensor, reg_type: str, strength: float) -> torch.Tensor:
    """L2 (Ridge), L1 (Lasso), or a 50/50 blend of both (ElasticNet) penalty on the weights."""
    if reg_type == "Ridge":
        return strength * torch.sum(W ** 2)
    if reg_type == "Lasso":
        return strength * torch.sum(torch.abs(W))
    if reg_type == "ElasticNet":
        return strength * (0.5 * torch.sum(W ** 2) + 0.5 * torch.sum(torch.abs(W)))
    raise ValueError(f"Unknown regularization type: {reg_type}")


def predict(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Predicted class indices (argmax over logits)."""
    with torch.no_grad():
        logits = torch.matmul(X, W) + b
        return torch.argmax(logits, dim=1)
