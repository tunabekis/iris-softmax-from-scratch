"""Loading, splitting, and scaling of the Iris dataset."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, TEST_SIZE_OF_REMAINDER, VALIDATION_AND_TEST_SIZE


@dataclass
class DatasetSplit:
    """Scaled features (as tensors) and integer labels for one split."""

    X: torch.FloatTensor
    y: torch.LongTensor
    y_raw: np.ndarray


@dataclass
class PreparedData:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def load_and_prepare_data() -> PreparedData:
    """Load Iris, split 70/15/15, and standardize features using train statistics."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VALIDATION_AND_TEST_SIZE, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE_OF_REMAINDER, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return PreparedData(
        train=DatasetSplit(torch.FloatTensor(X_train_s), torch.LongTensor(y_train), y_train),
        val=DatasetSplit(torch.FloatTensor(X_val_s), torch.LongTensor(y_val), y_val),
        test=DatasetSplit(torch.FloatTensor(X_test_s), torch.LongTensor(y_test), y_test),
    )
