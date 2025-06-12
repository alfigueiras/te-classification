from logs.checkpoint import save_checkpoint, save_best
from logs.logging import log_metrics, log_confusion_matrix
from models.att_models import GAT_Kmer_Classifier, GATv2Conv_Kmer_Classifier, GraphTransformer_Kmer_Classifier 
from models.loss import FocalLoss

import torch
import torch.distributed as dist
import os

from torch import nn,optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader

def train(rank, world_size, dataset, config, run):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f'cuda:{rank}')

    # Model initialization
    if config['model'] == 'GAT':
        model = GAT_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p']).to(device)
    elif config['model'] == 'GATv2':
        model = GATv2Conv_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers']).to(device)
    elif config['model'] == 'GraphTransformer':
        model = GraphTransformer_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers']).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    
    model = DDP(model, device_ids=[rank])

    train_loader = NeighborLoader(
        dataset,
        input_nodes=dataset.train_mask,
        num_neighbors=[15, 10],
        batch_size=config['batch_size'],
        shuffle=True,
        filter_per_worker=True,
        num_workers=4,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset.test_mask.nonzero(as_tuple=True)[0],
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        dataset,
        input_nodes=None,  # Sampler handles it
        num_neighbors=[15, 10],
        batch_size=config['batch_size'],
        sampler=val_sampler,
        filter_per_worker=True,
        shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = FocalLoss(alpha=config['focal_loss_alpha'], gamma=config['focal_loss_gamma']) if config['use_focal_loss'] else nn.CrossEntropyLoss()
    epoch = 1

    best_f1=0
    best_epoch = 0

    os.makedirs("results", exist_ok=True)

    while epoch <= config['epochs']:
        train_gnn_epoch(epoch, model, train_loader, optimizer, criterion, device, run)
        metrics = test_gnn_epoch(epoch, model, test_loader, criterion, device, run)
        if epoch % config['save_interval'] == 0 and rank == 0:
            save_checkpoint(run, epoch, model, "results", config['model'])
        if metrics['test/f1'] > best_f1 and rank == 0:
            best_epoch, best_f1 = save_best(run, best_epoch, epoch, model, best_f1, metrics['test/f1'], "results", config['model'])
        
        epoch += 1

    dist.destroy_process_group()
    print(f"Best test F1 score: {best_f1:.4f} at epoch {best_epoch}")
    
def train_gnn_epoch(epoch, model, train_loader, optimizer, criterion, device, run):
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
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_targets, all_preds, split="train")

def test_gnn_epoch(epoch, model, test_loader, criterion, device, run):
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
    all_probs, all_preds, all_targets, avg_loss = gather_all_predictions(all_probs, all_preds, all_targets, avg_loss)

    if dist.get_rank() == 0:
        metrics=compute_metrics(epoch, all_probs, all_preds, all_targets, avg_loss, split="test")
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_targets, all_preds, split="test")
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

