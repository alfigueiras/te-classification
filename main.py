from copy import deepcopy
import random

from configs.default import get_config
from data.dataset import create_dataset, dataset_choose_single_family, dataset_split_by_components, merge_pyg_datasets, standardize_selected_columns, random_dataset_split, test_standardize
from models.train import train

import os
import pickle
import torch
import torch.multiprocessing as mp
import numpy as np
import optuna
import json

def run_trial(config=None):

    gpu_ids = config.get("gpu_ids", "all")  # e.g. [0,2,3]

    if gpu_ids != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    SEED = config.get("seed", 42)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(config["seed"])

    processed_file=f"{config['species']}{config['k_mers']}{config['fam_type']}.pt"

    os.makedirs("data/processed", exist_ok=True)

    fam_counts=None

    if config["species"]!="all":
        if processed_file not in os.listdir("data/processed") or config["recreate_dataset"]:
            print(f"Creating dataset...")
            dataset, G=create_dataset(config)
        else:
            print("Found processed dataset, loading...")
            dataset=torch.load(f"data/processed/{processed_file}", weights_only=False)
            G=pickle.load(open(f"data/processed/graph_{config['species']}{config['k_mers']}{config['fam_type']}.pickle", 'rb'))
    elif config["species"]=="all":

        mus_dataset=torch.load(f"data/processed/mouse3Novo.pt", weights_only=False)
        dog_dataset=torch.load(f"data/processed/dog3Novo.pt", weights_only=False)
        dro_dataset=torch.load(f"data/processed/dro3Novo.pt", weights_only=False)

        G_mus=pickle.load(open(f"data/processed/graph_mouse3Novo.pickle", 'rb'))
        G_dog=pickle.load(open(f"data/processed/graph_dog3Novo.pickle", 'rb'))
        G_dro=pickle.load(open(f"data/processed/graph_dro3Novo.pickle", 'rb'))

        if config["partition"]=="single_family":
            mus_dataset=dataset_choose_single_family(G_mus, mus_dataset, config["single_family_fam"])
            dog_dataset=dataset_choose_single_family(G_dog, dog_dataset, config["single_family_fam"])
            dro_dataset=dataset_choose_single_family(G_dro, dro_dataset, config["single_family_fam"])
        elif config["partition"]=="random":
            mus_train_mask, mus_test_mask = random_dataset_split(mus_dataset, config)
            mus_dataset.train_mask = mus_train_mask
            mus_dataset.test_mask = mus_test_mask

            dog_train_mask, dog_test_mask = random_dataset_split(dog_dataset, config)
            dog_dataset.train_mask = dog_train_mask
            dog_dataset.test_mask = dog_test_mask

            dro_train_mask, dro_test_mask = random_dataset_split(dro_dataset, config)
            dro_dataset.train_mask = dro_train_mask
            dro_dataset.test_mask = dro_test_mask
        elif config["partition"]=="families":
            mus_mask, mus_fam_counts=dataset_split_by_components(G_mus, mus_dataset, config)
            if config["families_test"]=="mouse" or config["families_test"]=="all":
                mus_dataset.train_mask = mus_mask[0]
                mus_dataset.test_mask = mus_mask[1]
            else:
                mus_dataset.train_mask = torch.ones(mus_dataset.num_nodes, dtype=torch.bool)
                mus_dataset.test_mask = torch.zeros(mus_dataset.num_nodes, dtype=torch.bool)

            mus_dataset._node_families = [
                G_mus.nodes[node].get("lst_dfam_repeats", [])
                for node in G_mus.nodes()
            ]

            dog_mask, dog_fam_counts=dataset_split_by_components(G_dog, dog_dataset, config)
            if config["families_test"]=="dog" or config["families_test"]=="all":
                dog_dataset.train_mask = dog_mask[0]
                dog_dataset.test_mask = dog_mask[1]
            else:
                dog_dataset.train_mask = torch.ones(dog_dataset.num_nodes, dtype=torch.bool)
                dog_dataset.test_mask = torch.zeros(dog_dataset.num_nodes, dtype=torch.bool)

            dog_dataset._node_families = [
                G_dog.nodes[node].get("lst_dfam_repeats", [])
                for node in G_dog.nodes()
            ]

            dro_mask, dro_fam_counts=dataset_split_by_components(G_dro, dro_dataset, config)
            if config["families_test"]=="dro" or config["families_test"]=="all":
                dro_dataset.train_mask = dro_mask[0]
                dro_dataset.test_mask = dro_mask[1]
            else:
                dro_dataset.train_mask = torch.ones(dro_dataset.num_nodes, dtype=torch.bool)
                dro_dataset.test_mask = torch.zeros(dro_dataset.num_nodes, dtype=torch.bool)

            dro_dataset._node_families = [
                G_dro.nodes[node].get("lst_dfam_repeats", [])
                for node in G_dro.nodes()
            ]

            if config["families_test"]=="all":
                fam_counts={"mouse": mus_fam_counts, "dog": dog_fam_counts, "dro": dro_fam_counts}
            else:
                fam_counts=mus_fam_counts if config["families_test"]=="mouse" else (dog_fam_counts if config["families_test"]=="dog" else dro_fam_counts)

        datasets=[mus_dataset, dog_dataset, dro_dataset]
        Gs=[G_mus, G_dog, G_dro]

        dataset=merge_pyg_datasets(datasets)

        if config["partition"]=="families":
            dataset._node_families = (
                mus_dataset._node_families
                + dog_dataset._node_families
                + dro_dataset._node_families
            )

    if config["features_subset"]=="none":
        dataset.x=torch.ones((dataset.num_nodes, 1), dtype=torch.float32)
    elif config["features_subset"]=="original":
        selected_features=dataset.base_features
    elif config["features_subset"]=="structural":
        selected_features=dataset.base_features+dataset.struct_features
    elif config["features_subset"]=="alg":
        selected_features=dataset.base_features+dataset.struct_features+dataset.alg_features
    elif config["features_subset"]=="dnabert":
        selected_features=dataset.base_features+dataset.struct_features+dataset.alg_features+dataset.dnabert_features
    elif config["features_subset"]=="k_mer_counts":
        selected_features=dataset.base_features+dataset.struct_features+dataset.alg_features+dataset.kmer_features
    elif config["features_subset"]=="all_less_alg":
        selected_features=dataset.base_features+dataset.struct_features+dataset.dnabert_features+dataset.kmer_features
    elif config["features_subset"]=="all_less_struct":
        selected_features=dataset.base_features+dataset.alg_features+dataset.dnabert_features+dataset.kmer_features
    elif config["features_subset"]=="all_less_original":
        selected_features=dataset.struct_features+dataset.alg_features+dataset.dnabert_features+dataset.kmer_features
    elif config["features_subset"]=="entropy":
        selected_features=dataset.base_features+dataset.struct_features+dataset.alg_features+dataset.entropy_features
    elif config["features_subset"]=="all":
        selected_features=dataset.base_features+dataset.struct_features+dataset.alg_features+dataset.dnabert_features+dataset.kmer_features

    if config["features_subset"]!="none":
        indices = [dataset.feature_names.index(f) for f in selected_features]
        dataset.x = dataset.x[:, indices]   
        dataset.feature_names = selected_features
    elif config["features_subset"]=="none":
        dataset.feature_names = ["constant_feature"]

    if config["partition"]=="families" and config["species"]!="all":
        mask, fam_counts=dataset_split_by_components(G, dataset, config)
        dataset.train_mask = mask[0]
        dataset.test_mask = mask[1]
        dataset._node_families = [G.nodes[node].get("lst_dfam_repeats", []) for node in G.nodes()]
    elif config["partition"]=="single_family" and config["species"]!="all":
        dataset=dataset_choose_single_family(G, dataset, config["single_family_fam"])
    elif config["partition"]=="random":
        train_mask, test_mask = random_dataset_split(dataset, config)
        dataset.train_mask = train_mask
        dataset.test_mask = test_mask
    elif config["partition"]=="two_graphs":
        test_processed_file=f"{config['test_species']}{config['test_k_mers']}{config['test_fam_type']}.pt"

        if test_processed_file not in os.listdir("data/processed") or config["recreate_dataset"]:
            print(f"Creating dataset...")
            new_config=config.copy()
            new_config["species"]=config['test_species']
            new_config["k_mers"]=config['test_k_mers']
            new_config["fam_type"]=config['test_fam_type']
            test_dataset, test_G=create_dataset(new_config)
        else:
            print("Found processed dataset, loading...")
            test_dataset=torch.load(f"data/processed/{test_processed_file}", weights_only=False)
            test_G=pickle.load(open(f"data/processed/graph_{test_processed_file.replace('.pt', '.pickle')}", 'rb'))

        dataset.train_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
        dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
        test_dataset.train_mask = torch.zeros(test_dataset.num_nodes, dtype=torch.bool)
        test_dataset.test_mask = torch.ones(test_dataset.num_nodes, dtype=torch.bool)

        if config["features_subset"]!="all":
            test_dataset.x = test_dataset.x[:, indices]   
            test_dataset.feature_names = selected_features

    #standardize
    if config["features_subset"]!="none":
        exclude = [f for f in dataset.alg_features if f in dataset.feature_names]
        exclude += [f for f in dataset.dnabert_features if f in dataset.feature_names]
        dataset,mean,std=standardize_selected_columns(dataset, dataset.train_mask, exclude_feature_names=exclude)
        dataset.x_train_mean=mean
        dataset.x_train_std=std

        if config["partition"]=="two_graphs":
            test_dataset=test_standardize(test_dataset, mean, std, exclude_feature_names=exclude)

    if config["partition"]!="two_graphs":
        test_dataset=None

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
    mp.spawn(train, args=(world_size, dataset, config, fam_counts, test_dataset), nprocs=world_size, join=True)

def objective(trial):
    base_config = get_config()
    config = deepcopy(base_config)

    config["trial_number"] = trial.number
    config["mlflow_run_name"] = f"trial_{trial.number}"

    # sample hyperparameters
    if config["experiment_mode"]=="basic":
        config["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [512, 768, 1024])
        config["embedding_dim"] = trial.suggest_categorical("embedding_dim", [512, 768, 1024])
        config["dropout_p"] = trial.suggest_float("dropout_p", 0.1, 0.5)
        config["edge_dropout_p"] = trial.suggest_float("edge_dropout_p", 0.2, 0.5)
        config["num_layers"] = trial.suggest_int("num_layers", 2, 5)

        if config["model"] in ["GAT", "GATv2"]:
            config["heads"] = trial.suggest_categorical("heads", [1, 2, 4])

        if config["use_focal_loss"]:
            config["focal_loss_alpha"] = trial.suggest_float("focal_loss_alpha", 0.1, 0.9)
            config["focal_loss_gamma"] = trial.suggest_float("focal_loss_gamma", 1.0, 5.0)
    elif config["experiment_mode"]=="features_subsets":
        config["features_subset"] = trial.suggest_categorical("features_subset", ["none", "original", "structural", "alg", "dnabert", "k_mer_counts", "all_less_alg", "all_less_struct", "all_less_original"])
        config["mlflow_run_name"] = f"trial_{config['features_subset']}"
    elif config["experiment_mode"]=="sequence_experiment":
        config["features_subset"] = trial.suggest_categorical("features_subset", ["all", "alg", "entropy"])
        config["mlflow_run_name"] = f"trial_{config['features_subset']}"
    
    try:
        run_trial(config)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
    finally:
        torch.cuda.empty_cache()

    final_path=f"results/{config.get('project_name', 'default')}/trial_{trial.number}/final_result_{trial.number}.json"

    if os.path.exists(final_path):
        with open(final_path, "r") as f:
            result = json.load(f)

        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("best_f1", result["best_f1"])
        trial.set_user_attr("final_test_f1", result["final_test_f1"])

        return result["final_test_f1"]
    else:
        print(f"Final result file not found for trial {trial.number}, setting default values")
        trial.set_user_attr("best_epoch", -1)
        trial.set_user_attr("best_f1", float("-inf"))
        trial.set_user_attr("final_test_f1", float("-inf"))
        return float("-inf")

    

def run_hpo(n_trials=20, experiment_mode="basic"):
    if experiment_mode=="basic":
        study = optuna.create_study(direction="maximize", study_name="gnn_te_classification")
        study.optimize(objective, n_trials=n_trials)
    elif experiment_mode=="features_subsets":
        search_space = {
            "features_subset": ["none", "original", "structural", "alg", "dnabert", "k_mer_counts", "all_less_alg", "all_less_struct", "all_less_original"]
        }
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(sampler=sampler, direction="maximize", study_name="gnn_te_classification_features_subsets")
        study.optimize(objective)
    elif experiment_mode=="sequence_experiment":
        search_space = {
            "features_subset": ["all", "alg", "entropy"]
        }
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(sampler=sampler, direction="maximize", study_name="gnn_te_classification_sequence_experiment")
        study.optimize(objective)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

if __name__ == "__main__":
    config=get_config()
    if config["use_hpo"]:
        run_hpo(n_trials=config["n_trials"], experiment_mode=config["experiment_mode"])
    else:
        run_trial(config)