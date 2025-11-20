from logs.logging import init_wandb, finish_wandb
from configs.default import get_config
from data.dataset import create_dataset, dataset_split_by_components
from models.train import train

import os
import pickle
import torch
import torch.multiprocessing as mp
import numpy as np

def main(config=None, run=None):

    gpu_ids = config.get("gpu_ids", "all")  # e.g. [0,2,3]

    if gpu_ids != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    processed_file=f"{config['species']}{config['k_mers']}{config['fam_type']}.pt"

    os.makedirs("data/processed", exist_ok=True)

    if processed_file not in os.listdir("data/processed") or config["recreate_dataset"]:
        print(f"Creating dataset...")
        dataset,G=create_dataset(config)
    else:
        print("Found processed dataset, loading...")
        dataset=torch.load(f"data/processed/{processed_file}", weights_only=False)
        G=pickle.load(open(f"data/processed/graph_{config['species']}{config['k_mers']}{config['fam_type']}.pickle", 'rb'))

    mask=dataset_split_by_components(G, dataset, config, run)

    dataset.train_mask = mask[0]
    dataset.test_mask = mask[1]

    if torch.cuda.is_available():
        if gpu_ids == "all":
            world_size = torch.cuda.device_count()
        else:
            world_size = len(gpu_ids)
        print(f"Using {world_size} GPUs for training")
    else:
        world_size = 1
        print("CUDA not available, using CPU for training")

    print("Starting training...")
    mp.spawn(train, args=(world_size, dataset, config, run), nprocs=world_size, join=True)

if __name__ == "__main__":
    config=get_config()

    run=init_wandb(
        project_name=config["project_name"],
        config_dict=config,
        run_name=config.get("run_name", None)
    )
    config.update(run.config)
    print(config['learning_rate'])
    try:
        main(config, run)
    except Exception as e:
        print(f"An error occurred: {e}")
        run.log({"error": str(e)})
        raise e
    finally:
        finish_wandb(run)