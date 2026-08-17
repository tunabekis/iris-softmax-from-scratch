"""Central configuration for the experiment (splits, hyperparameters, seed)."""

# Reproducibility. Applied to both sklearn (data splitting/CV) and torch
# (weight initialization) so repeated runs reproduce the same numbers.
RANDOM_SEED = 42

# Dataset split ratios: 70% train, 15% validation, 15% test.
VALIDATION_AND_TEST_SIZE = 0.30
TEST_SIZE_OF_REMAINDER = 0.50

NUM_CLASSES = 3

# Model-selection stage: pick the best polynomial degree via k-fold CV.
CANDIDATE_DEGREES = [1, 2, 3]
CV_FOLDS = 3
CV_LEARNING_RATE = 0.01
CV_EPOCHS = 100

# Final training stage: sweep learning rates x regularization types.
FINAL_LEARNING_RATES = [0.000015, 0.0015, 0.15]
REGULARIZATION_TYPES = ["Ridge", "Lasso", "ElasticNet"]
REGULARIZATION_STRENGTH = 0.01
FINAL_EPOCHS = 50
