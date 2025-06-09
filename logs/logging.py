import wandb

def init_wandb(project_name, config_dict, run_name=None):
    wandb.init(
        project=project_name,
        config=config_dict,
        name=run_name,
        reinit=True,
    )

def log_metrics(metrics, step=None):
    wandb.log(metrics, step=step)

def log_confusion_matrix(y_true, y_pred, split="train"):
    wandb.log({
        f"{split}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=[str(i) for i in range(1,10)]
        )
    })

def finish_wandb():
    wandb.finish()