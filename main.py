import mlflow
from configs.default import get_config
from data.dataset import create_dataset, dataset_split_by_components, standardize_selected_columns, random_dataset_split
from models.train import train

import os
import pickle
import torch
import torch.multiprocessing as mp
import numpy as np

def main(config=None):

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
        dataset, G=create_dataset(config)
    else:
        print("Found processed dataset, loading...")
        dataset=torch.load(f"data/processed/{processed_file}", weights_only=False)
        G=pickle.load(open(f"data/processed/graph_{config['species']}{config['k_mers']}{config['fam_type']}.pickle", 'rb'))

    if config["features_subset"]=="none":
        dataset.x=torch.ones((dataset.num_nodes, 1), dtype=torch.float32)
    elif config["features_subset"]=="original":
        selected_features=dataset.base_features
    elif config["features_subset"]=="structural":
        selected_features=dataset.base_features+dataset.structural_features
    elif config["features_subset"]=="alg":
        selected_features=dataset.base_features+dataset.structural_features+dataset.alg_features
    elif config["features_subset"]=="dnabert":
        selected_features=dataset.base_features+dataset.structural_features+dataset.alg_features+dataset.dnabert_features
    elif config["features_subset"]=="k_mer_counts":
        selected_features=dataset.base_features+dataset.structural_features+dataset.alg_features+dataset.kmer_features

    if config["features_subset"]!="all":
        indices = [dataset.feature_names.index(f) for f in selected_features]
        dataset.x = dataset.x[:, indices]   
        dataset.feature_names = selected_features


    if config["partition"]=="families":
        mask=dataset_split_by_components(G, dataset, config)
        dataset.train_mask = mask[0]
        dataset.test_mask = mask[1]
    elif config["partition"]=="random":
        train_mask, test_mask = random_dataset_split(dataset, config)
        dataset.train_mask = train_mask
        dataset.test_mask = test_mask
    elif config["partition"]=="two_graphs":
        dataset.train_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
        dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)

    #standardize
    if config["features_subset"]!="none":
        exclude = [f for f in dataset.alg_features if f in dataset.feature_names]
        dataset,mean,std=standardize_selected_columns(dataset, dataset.train_mask, exclude_feature_names=exclude)
        dataset.x_train_mean=mean
        dataset.x_train_std=std

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
    mp.spawn(train, args=(world_size, dataset, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    config=get_config()

    print(config['learning_rate'])
    try:
        main(config)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        mlflow.end_run()