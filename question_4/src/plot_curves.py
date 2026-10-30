import json

import matplotlib.pyplot as plt

from src import config


def main():
    figure, axis = plt.subplots(figsize=(8, 5))
    styles = {"rnn": "tomato", "lstm": "steelblue"}
    for name, color in styles.items():
        with open(config.ARTIFACT_DIR / f"{name}_history.json") as file:
            history = json.load(file)
        epochs = range(1, len(history["train_loss"]) + 1)
        axis.plot(epochs, history["train_loss"], color=color, linestyle="--", label=f"{name} train")
        axis.plot(epochs, history["val_loss"], color=color, label=f"{name} val")
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.ARTIFACT_DIR / "loss_curves.png", dpi=150)


if __name__ == "__main__":
    main()
