import wandb

def init_wandb(project_name, config_dict, run_name=None):
    run=wandb.init(
        project=project_name,
        config=config_dict,
        name=run_name,
        reinit=True,
    )

    run.define_metric("epoch")

    for split in ["train", "test"]:
        for metric in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc"]:
            run.define_metric(f"{split}/{metric}", step_metric="epoch")
            
    return run

def log_metrics(run, metrics, step=None):
    run.log(metrics)

def log_confusion_matrix(run, y_true, y_pred, split="train"):
    run.log({
        f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.int().numpy(),
            preds=y_pred.int().numpy(),
            class_names=["Not TE", "TE"]
        )
    })

def finish_wandb(run):
    run.finish()