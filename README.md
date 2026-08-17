# Iris Softmax Regression (Manual Implementation)

A from-scratch (no `nn.Linear`, no `nn.CrossEntropyLoss`) softmax regression pipeline
for the scikit-learn Iris dataset, built with PyTorch autograd for gradient computation.

## What it does

1. **Data preparation** — splits Iris into 70% train / 15% validation / 15% test and
   standardizes features with `StandardScaler` (fit on train only).
2. **Model selection** — expands features into polynomial terms (degree 1, 2, 3) and
   picks the best degree via 3-fold cross-validation on the training set.
3. **Hyperparameter sweep** — trains 9 models at the winning degree: every combination
   of 3 learning rates (`1.5e-5`, `1.5e-3`, `0.15`) x 3 regularizers (Ridge, Lasso,
   ElasticNet), each for 50 epochs of manual gradient descent.
4. **Visualization** — plots train/validation/test cross-entropy loss curves for all
   9 models side by side.
5. **Final evaluation** — selects the model with the lowest final validation loss and
   reports its accuracy, precision, recall, and F1-score (macro-averaged) on the test set.

Softmax is implemented manually with the log-sum-exp stability trick, and all runs are
seeded (`RANDOM_SEED` in `src/config.py`) for reproducible results.

## Project structure

```
main.py               # Entry point: orchestrates the full pipeline
src/
  config.py            # Hyperparameters, split ratios, random seed
  data.py               # Dataset loading, splitting, scaling
  features.py            # Polynomial feature expansion
  model.py                 # Manual softmax, cross-entropy, regularization, prediction
  training.py                # Gradient-descent training loop + cross-validation
  evaluation.py                 # Accuracy / precision / recall / F1 computation
  visualization.py                 # Loss-curve plotting
```

## Tech stack

- Python 3
- PyTorch (autograd only — no built-in linear/loss layers)
- scikit-learn (dataset, train/val/test split, K-fold CV, scaling, metrics)
- NumPy
- Matplotlib

## Setup & usage

```bash
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt

python main.py
```

Running the script prints the cross-validation accuracy per polynomial degree, opens
a window with the train/validation/test loss curves for all 9 models, and prints the
final test-set metrics for the best-performing model.
