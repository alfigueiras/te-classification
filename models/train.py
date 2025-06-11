from logs.checkpoint import save_checkpoint, save_best
from logs.logging import log_metrics, log_confusion_matrix
from models.att_models import GAT_Kmer_Classifier, GATv2Conv_Kmer_Classifier, GraphTransformer_Kmer_Classifier 

from torch import nn,optim
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader

def train(rank, world_size, dataset, config):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f'cuda:{rank}')

    # Model initialization
    if config['model'] == 'GAT':
        model = GAT_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'])
    elif config['model'] == 'GATv2':
        model = GATv2Conv_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'])
    elif config['model'] == 'GraphTransformer':
        model = GraphTransformer_Kmer_Classifier(dataset, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], heads=config['heads'], dropout_p=config['dropout_p'], edge_dropout_p=config['edge_dropout_p'], num_layers=config['num_layers'])
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    
    model = DDP(model, device_ids=[rank])
