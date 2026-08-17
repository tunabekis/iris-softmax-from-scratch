"""Loss-curve plotting for the learning-rate x regularization sweep."""

import matplotlib.pyplot as plt

_SUBPLOT_TITLES = ("Train Loss (Pure)", "Validation Loss", "Test Loss")
_HISTORY_KEYS = ("train", "val", "test")


def plot_loss_curves(results: dict) -> None:
    """Plot train/validation/test loss curves for every model in `results`."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (title, key) in enumerate(zip(_SUBPLOT_TITLES, _HISTORY_KEYS)):
        for name, data in results.items():
            axes[i].plot(data["history"][key], label=name)
        axes[i].set_title(title)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend(fontsize="x-small")

    plt.tight_layout()
    plt.show()
