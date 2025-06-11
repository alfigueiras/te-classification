import torch
import wandb
import os
import json 
from collections import Counter

def save_checkpoint(run, epoch, model, path, name="model"):
    save_path = f"{path}/{name}_{epoch}.pt"
    torch.save(model.state_dict(), save_path)
    run.save(save_path)

def save_best(run, best_epoch, epoch, model, best_val, current_val, path, name="model"):
    if current_val > best_val:
        torch.save(model.state_dict(), f"{path}/best_{name}.pt")
        run.summary["best_val_loss"] = current_val
        run.save(f"{path}/best_model.pt")
        return epoch, current_val

    return best_epoch, best_val

def filter_counter_by_keys(run, te_counts, mask):
    artifact = wandb.Artifact(name=f"family_counters", type="family-counters")

    train_counter = Counter({key: te_counts[key] for key in mask[2]})
    test_counter = Counter({key: te_counts[key] for key in mask[3]})

    combined = {
        "train_families": dict(train_counter.most_common()),
        "test_families": dict(test_counter.most_common())
    }

    # Save to temp file
    file_name = f"family_counter.json"
    with open(file_name, "w") as f:
        json.dump(combined, f, indent=2)

    # Add to artifact
    artifact.add_file(file_name)

    # Log the artifact to W&B
    run.log_artifact(artifact)

    # Optionally: clean up temp files
    os.remove(f"family_counter.json")