import torch
import mlflow
import os
import json 
from collections import Counter

def save_checkpoint(run, epoch, model, path, name="model"):
    os.makedirs(f"{path}/{run.data.tags.get('mlflow.runName')}", exist_ok=True)
    save_path = f"{path}/{run.data.tags.get('mlflow.runName')}/{name}_{epoch}.pt"
    torch.save(model.state_dict(), save_path)
    mlflow.pytorch.log_model(model, name=f"{name}_{epoch}")

def save_best(run, best_epoch, epoch, model, best_val, current_val, path, name="model", sum_name="best_test_f1"):
    
    if current_val > best_val:
        run_name = run.data.tags.get("mlflow.runName")
        os.makedirs(f"{path}/{run_name}", exist_ok=True)
        save_path = f"{path}/{run_name}/best_{name}.pt"

        torch.save(model.state_dict(), save_path)
        mlflow.log_metric(sum_name, current_val, step=epoch)
        mlflow.pytorch.log_model(model, name=f"best_{name}")
        return epoch, current_val

    return best_epoch, best_val

def filter_counter_by_keys(te_counts, mask):

    train_counter = Counter({key: te_counts[key] for key in mask[2]})
    test_counter = Counter({key: te_counts[key] for key in mask[3]})

    combined = {
        "train_families": dict(train_counter.most_common()),
        "test_families": dict(test_counter.most_common())
    }

    # Save to file
    file_name = f"family_counter.json"
    with open(file_name, "w") as f:
        json.dump(combined, f, indent=2)