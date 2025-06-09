import torch
import wandb
import os
import json 
from collections import Counter

def save_checkpoint(epoch, model, path, name="model"):
    save_path = f"{path}/{name}_{epoch}.pt"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)

def save_best(best_epoch, epoch, model, best_val, current_val, path, name="model"):
    if current_val > best_val:
        torch.save(model.state_dict(), f"{path}/best_{name}.pt")
        wandb.run.summary["best_val_loss"] = current_val
        wandb.save(f"{path}/best_model.pt")
        return epoch, current_val

    return best_epoch, best_val

def filter_counter_by_keys(te_counts, masks):
    artifact = wandb.Artifact(name=f"family_counters", type="family-counters")

    for i, mask in enumerate(masks):
        train_counter = Counter({key: te_counts[key] for key in mask[2]})
        test_counter = Counter({key: te_counts[key] for key in mask[3]})

        combined = {
            "train_families": dict(train_counter.most_common()),
            "test_families": dict(test_counter.most_common())
        }

        # Save to temp file
        file_name = f"family_counter{i+1}.json"
        with open(file_name, "w") as f:
            json.dump(combined, f, indent=2)

        # Add to artifact
        artifact.add_file(file_name)

    # Log the artifact to W&B
    wandb.log_artifact(artifact)

    # Optionally: clean up temp files
    for i in range(len(masks)):
        os.remove(f"family_counter{i+1}.json")