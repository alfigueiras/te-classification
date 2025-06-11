from logs.logging import init_wandb, finish_wandb
from configs.default import get_config
from data.dataset import create_dataset, dataset_split_by_components
from models.train import train

import os
import pickle
import torch
import torch.multiprocessing as mp

def main():
    config=get_config()
    init_wandb(project_name=config['project_name'], config_dict=config, run_name=config.get('run_name', None))

    processed_file=f"data/processed/{config['species']}{config['kmers']}{config['fam_type']}.pt"

    if processed_file not in os.listdir("data/processed") or config["recreate_dataset"]:
        print(f"Creating dataset...")
        dataset,G=create_dataset(config['species'], config['kmers'], config['fam_type'])
    else:
        print("Found processed dataset, loading...")
        dataset=torch.load(processed_file)
        G=pickle.load(open(f"data/processed/graph_{config['species']}{config['kmers']}{config['fam_type']}.pickle", 'rb'))

    mask=dataset_split_by_components(G, dataset, config)

    dataset.train_mask = mask[0]
    dataset.test_mask = mask[1]

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)

    finish_wandb()

if __name__ == "__main__":
    main()