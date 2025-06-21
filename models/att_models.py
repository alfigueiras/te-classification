import torch.nn.functional as F

from torch.nn import Module, Dropout, LeakyReLU, Linear, ModuleList, Identity, Sequential, Dropout
from torch_geometric.nn import BatchNorm, GATConv, TransformerConv, GATv2Conv
from torch_geometric.utils import dropout_edge

class GAT_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2):
        super(GAT_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p

        self.linear_pre = Sequential(
            Linear(dataset.num_node_features, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.gats = ModuleList([GATConv(embedding_dim, hidden_dim//heads, heads=heads, concat=True), 
                                GATConv(hidden_dim, hidden_dim//heads, heads=heads, concat=True)])

        self.bns = ModuleList([BatchNorm((hidden_dim // heads) * heads), BatchNorm((hidden_dim // heads) * heads)])

        self.lins = ModuleList([Linear((hidden_dim // heads) * heads, hidden_dim), Linear(hidden_dim, hidden_dim//2), Linear(hidden_dim//2, 1)])


    def forward(self, x, edge_index):
        h = self.linear_pre(x)

        # Apply edge dropout only during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (gat, bn) in enumerate(zip(self.gats, self.bns)):
            h = gat(h, edge_index)
            h = bn(h)
            h = F.leaky_relu(h)

        for lin in self.lins[:-1]:
            h = lin(h)
            h = F.leaky_relu(h)

        out=self.lins[-1](h)

        return out
    
class GATv2Conv_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2):
        super(GATv2Conv_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear_pre = Sequential(
            Linear(dataset.num_node_features, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.res_proj = Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else Identity()

        self.gatv2 = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.gatv2.append(
                GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout_p)
            )
            self.norms.append(BatchNorm((hidden_dim // heads) * heads))

        self.lins = ModuleList([
            Linear((hidden_dim // heads) * heads, hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            Linear(hidden_dim // 2, 1)
        ])

    def forward(self, x, edge_index):
        h = self.linear_pre(x)

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (conv, norm) in enumerate(zip(self.gatv2, self.norms)):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

            if i == 0 and self.embedding_dim != self.hidden_dim:
                h_in = self.res_proj(h_in)

            h = h + h_in

        for lin in self.lins[:-1]:
            h = lin(h)
            h = F.leaky_relu(h)

        out = self.lins[-1](h)
        return out
    
class GraphTransformer_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2):
        super(GraphTransformer_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear_pre = Sequential(
            Linear(dataset.num_node_features, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.res_proj = Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else Identity()
        self.transformers = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.transformers.append(
                TransformerConv(in_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout_p, beta=True)
            )
            self.norms.append(BatchNorm((hidden_dim // heads) * heads))

        self.lins = ModuleList([
            Linear((hidden_dim // heads) * heads, hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            Linear(hidden_dim // 2, 1)
        ])

    def forward(self, x, edge_index):
        h = self.linear_pre(x)

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (conv, norm) in enumerate(zip(self.transformers, self.norms)):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            
            if i == 0 and self.embedding_dim != self.hidden_dim:
                h_in = self.res_proj(h_in)

            h = h + h_in  # Residual connection

        for lin in self.lins[:-1]:
            h = lin(h)
            h = F.leaky_relu(h)

        out = self.lins[-1](h)
        return out