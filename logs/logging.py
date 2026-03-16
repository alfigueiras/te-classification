import mlflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def init_mlflow(experiment_name):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

def log_metrics(metrics, step=None):
    #step is the epoch number
    mlflow.log_metrics(metrics, step=step)

def log_confusion_matrix(y_true, y_pred, split="train", step=None, labels=[0,1]):
    cm = confusion_matrix(y_true.int().numpy(), y_pred.int().numpy(), labels=labels)
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric(f"{split}/tn", tn, step=step)
    mlflow.log_metric(f"{split}/fp", fp, step=step)
    mlflow.log_metric(f"{split}/fn", fn, step=step)
    mlflow.log_metric(f"{split}/tp", tp, step=step)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{split.capitalize()} Confusion Matrix - Epoch {step}")

    mlflow.log_figure(fig, f"confusion_matrices/{split}/{split}_epoch_{step:03d}.png")
    plt.close(fig)