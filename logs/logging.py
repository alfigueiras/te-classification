import mlflow
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def init_mlflow(experiment_name):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

def log_metrics(metrics, step=None):
    #step is the epoch number
    mlflow.log_metrics(metrics, step=step)

def log_confusion_matrix(
    y_true,
    y_pred,
    split="train",
    step=None,
    labels=(0, 1),
    class_names=None,
    cmap="Blues",
    normalize=True,
):
    y_true = y_true.int().cpu().numpy()
    y_pred = y_pred.int().cpu().numpy()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    mlflow.log_metric(f"{split}/tn", tn, step=step)
    mlflow.log_metric(f"{split}/fp", fp, step=step)
    mlflow.log_metric(f"{split}/fn", fn, step=step)
    mlflow.log_metric(f"{split}/tp", tp, step=step)

    if class_names is None:
        class_names = [str(x) for x in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, where=row_sums != 0)
    else:
        cm_display = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion" if normalize else "Count")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{split.capitalize()} Confusion Matrix - Epoch {step}",
    )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    threshold = cm_display.max() / 2.0 if cm_display.size else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count_text = f"{cm[i, j]:,}"
            if normalize:
                pct_text = f"{cm_display[i, j]*100:.2f}%"
                text = f"{count_text}\n({pct_text})"
            else:
                text = count_text

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > threshold else "black",
                fontsize=12,
                fontweight="bold",
            )

    fig.tight_layout()

    suffix = "normalized" if normalize else "raw"
    mlflow.log_figure(
        fig,
        f"confusion_matrices/{split}/{split}_{suffix}_epoch_{step:03d}.png"
    )
    plt.close(fig)