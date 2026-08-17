"""Polynomial feature expansion."""

import torch
from sklearn.preprocessing import PolynomialFeatures


def generate_polynomial_features(data, degree: int) -> torch.FloatTensor:
    """Expand standardized features into polynomial terms of the given degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return torch.FloatTensor(poly.fit_transform(data))
