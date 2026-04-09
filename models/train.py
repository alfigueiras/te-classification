from logs.checkpoint import save_checkpoint, save_best
from logs.logging import log_metrics, log_confusion_matrix, init_mlflow
from models.att_models import GAT_Kmer_Classifier, GATv2Conv_Kmer_Classifier, GraphTransformer_Kmer_Classifier, SAGEConv_Kmer_Classifier, GIN_Kmer_Classifier
from models.loss import FocalLoss

import torch
import torch.distributed as dist
import os
import mlflow
import json

from torch import nn,optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader

def train(rank, world_size, dataset, config, test_dataset=None):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f'cuda:{rank}')

    print(f'Running on rank {rank}, using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(rank)}')

    # Model initialization
    use_dnabert_proj = config["features_subset"] in ["all", "dnabert"]

    if config['model'] == 'GAT':
        model = GAT_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'], use_dnabert_proj=use_dnabert_proj, dnabert_proj_dim=config['dnabert_proj_dim']).to(device)
    elif config['model'] == 'GATv2':
        model = GATv2Conv_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'], use_dnabert_proj=use_dnabert_proj, dnabert_proj_dim=config['dnabert_proj_dim']).to(device)
    elif config['model'] == 'SAGE':
        model = SAGEConv_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'], use_dnabert_proj=use_dnabert_proj, dnabert_proj_dim=config['dnabert_proj_dim']).to(device)
    elif config['model'] == 'GIN':
        model = GIN_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'], use_dnabert_proj=use_dnabert_proj, dnabert_proj_dim=config['dnabert_proj_dim']).to(device)
    elif config['model'] == 'GraphTransformer':
        model = GraphTransformer_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers']).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    
    model = DDP(model, device_ids=[rank])

    train_loader = NeighborLoader(
        dataset,
        input_nodes=dataset.train_mask,
        num_neighbors=[-1, -1],
        batch_size=config['batch_size'],
        shuffle=True,
        filter_per_worker=True,
        num_workers=4,
    )

    if rank==0:
        if config["partition"]!="two_graphs":
            test_dataset=dataset

        test_loader = NeighborLoader(
            test_dataset,
            input_nodes=test_dataset.test_mask,
            num_neighbors=[-1, -1],
            batch_size=config['batch_size'],
            shuffle=False,
            filter_per_worker=True,
            num_workers=4,
        )
    else:
        test_loader=None

    early_stopper = EarlyStop(patience=config.get("early_stop_patience", 10), min_delta=config.get("early_stop_min_delta", 0))
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = FocalLoss(alpha=config['focal_loss_alpha'], gamma=config['focal_loss_gamma']) if config['use_focal_loss'] else nn.CrossEntropyLoss()
    epoch = 1

    best_f1=0
    best_epoch = 0

    os.makedirs("results", exist_ok=True)

    if rank == 0:
        init_mlflow(config["project_name"])

        run = mlflow.start_run(run_name=config.get("mlflow_run_name", None))
        if not config["use_hpo"]:
            result_path = f"results/{config.get('project_name', 'default')}/{run.data.tags.get('mlflow.runName')}"
        else: 
            result_path = f"results/{config.get('project_name', 'default')}/trial_{config.get('trial_number', -1)}"

        config["result_path"] = result_path

        mlflow.log_params({
            "trial_number": config.get("trial_number", -1),
            "model": config["model"],
            "learning_rate": config["learning_rate"],
            "hidden_dim": config["hidden_dim"],
            "embedding_dim": config["embedding_dim"],
            "dropout_p": config["dropout_p"],
            "edge_dropout_p": config["edge_dropout_p"],
            "num_layers": config["num_layers"],
            "batch_size": config["batch_size"],
            "world_size": dist.get_world_size(),
            "features_subset": config["features_subset"],
            "partition": config["partition"],
        })
        if "heads" in config:
            mlflow.log_param("heads", config["heads"])
        if config["use_focal_loss"]:
            mlflow.log_param("focal_loss_alpha", config["focal_loss_alpha"])
            mlflow.log_param("focal_loss_gamma", config["focal_loss_gamma"])
    else:
        run = None

    while epoch <= config['epochs']:
        train_gnn_epoch(epoch, model, train_loader, optimizer, criterion, device)
        stop_flag = torch.tensor([0], device=device)
        if rank==0:
            metrics = test_gnn_epoch(epoch, model, test_loader, criterion, device)
            
            final_test_f1 = metrics["test/f1"]   # overwrite every epoch, so last one remains
            stop = early_stopper.step(metrics['test/loss'])

            if stop:
                print(f"Early stopping at epoch {epoch}")
                save_checkpoint(epoch, model, config["result_path"], config['model'])
                stop_flag[0] = 1
            else:
                if epoch % config['save_interval'] == 0:
                    save_checkpoint(epoch, model, config["result_path"], config['model'])

                if metrics['test/f1'] > best_f1:
                    best_epoch, best_f1 = save_best(best_epoch, epoch, model, best_f1, metrics['test/f1'], config["result_path"], config['model'])

        if world_size > 1:
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break
            
        epoch += 1

    if rank == 0:
        path=f"{config['result_path']}/final_result.json"
        with open(path, "w") as f:
            json.dump({
                "final_test_f1": float(final_test_f1),
                "best_f1": float(best_f1),
                "best_epoch": int(best_epoch),
            }, f)

        mlflow.log_metric("final_test_f1", final_test_f1)
        mlflow.log_metric("best_test_f1", best_f1)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.end_run()

    dist.destroy_process_group()

    print(f"Best test F1 score: {best_f1:.4f} at epoch {best_epoch}")
    
def train_gnn_epoch(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs = torch.empty(0)
    all_preds = torch.empty(0)
    all_targets = torch.empty(0)

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index)
        logits = logits.squeeze(-1)

        loss = criterion(logits[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits[:batch.batch_size])
        predictions = (probs>=0.5).long()
        targets = batch.y[:batch.batch_size]

        all_probs = torch.cat([all_probs, probs.detach().cpu()])
        all_preds = torch.cat([all_preds, predictions.detach().cpu()])
        all_targets = torch.cat([all_targets, targets.detach().cpu()])

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    all_probs, all_preds, all_targets, avg_loss = gather_all_predictions(all_probs, all_preds, all_targets, avg_loss)

    if dist.get_rank() == 0:
        metrics=compute_metrics(epoch, all_probs, all_preds, all_targets, avg_loss, split="train")
        log_metrics(metrics, step=epoch)
        log_confusion_matrix(all_targets, all_preds, split="train", step=epoch)

def test_gnn_epoch(epoch, model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = torch.empty(0)
    all_preds = torch.empty(0)
    all_targets = torch.empty(0)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            logits = model(batch.x, batch.edge_index)
            logits = logits.squeeze(-1)

            loss = criterion(logits[:batch.batch_size], batch.y[:batch.batch_size])

            probs = torch.sigmoid(logits[:batch.batch_size])
            predictions = (probs>=0.5).long()
            targets = batch.y[:batch.batch_size]

            all_probs = torch.cat([all_probs, probs.detach().cpu()])
            all_preds = torch.cat([all_preds, predictions.detach().cpu()])
            all_targets = torch.cat([all_targets, targets.detach().cpu()])

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    metrics={}
    
    if dist.get_rank() == 0:
        metrics=compute_metrics(epoch, all_probs, all_preds, all_targets, avg_loss, split="test")
        log_metrics(metrics, step=epoch)
        log_confusion_matrix(all_targets, all_preds, split="test", step=epoch)
    return metrics

def gather_all_predictions(local_probs, local_preds, local_targets, local_avg_loss):
    if not dist.is_initialized():
        return local_preds, local_targets

    world_size = dist.get_world_size()

    prob_list = [None for _ in range(world_size)]
    pred_list = [None for _ in range(world_size)]
    target_list = [None for _ in range(world_size)]
    avg_loss_list = [None for _ in range(world_size)]

    dist.all_gather_object(prob_list, local_probs)
    dist.all_gather_object(pred_list, local_preds)
    dist.all_gather_object(target_list, local_targets)
    dist.all_gather_object(avg_loss_list, local_avg_loss)

    all_probs = torch.cat(prob_list, dim=0)
    all_preds = torch.cat(pred_list, dim=0)
    all_targets = torch.cat(target_list, dim=0)
    all_avg_loss = sum(avg_loss_list) / world_size

    return all_probs, all_preds, all_targets, all_avg_loss

def compute_metrics(epoch, all_probs, all_preds, all_targets, avg_loss, split):

    acc=(all_preds == all_targets).float().mean().item()
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)

    metrics = {
        f"epoch": epoch,
        f"{split}/loss": avg_loss,
        f"{split}/accuracy": acc,
        f"{split}/precision": precision,
        f"{split}/recall": recall,
        f"{split}/f1": f1,
        f"{split}/roc_auc": roc_auc
    }

    return metrics

class EarlyStop:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0 
        self.best_loss = float('inf')
    
    def step(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.count = 0
        elif val_loss + self.min_delta > self.best_loss:
            self.count += 1
        return self.count >= self.patience