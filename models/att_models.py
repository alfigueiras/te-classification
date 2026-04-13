import torch
import torch.nn.functional as F

from torch.nn import Module, Dropout, LeakyReLU, Linear, ModuleList, Identity, Sequential, Dropout
from torch_geometric.nn import BatchNorm, GATConv, GINConv, SAGEConv, TransformerConv, GATv2Conv
from torch_geometric.utils import dropout_edge
    
class GAT_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2, use_dnabert_proj=True, dnabert_proj_dim=128):
        super(GAT_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dnabert_proj_dim = dnabert_proj_dim
        self.use_dnabert_proj = use_dnabert_proj
        self.num_total_features = len(dataset.feature_names)
        self.num_dna_features = len(dataset.dnabert_features)
        self.num_other_features = self.num_total_features - self.num_dna_features

        if self.use_dnabert_proj:
            self.dna_proj = Sequential(
                Linear(self.num_dna_features, dnabert_proj_dim),
                BatchNorm(dnabert_proj_dim),
                LeakyReLU(),
                Dropout(dropout_p),
            )
            linear_pre_input_dim = self.num_total_features - self.num_dna_features + dnabert_proj_dim
        else:
            self.dna_proj = None
            linear_pre_input_dim = self.num_total_features

        self.linear_pre = Sequential(
            Linear(linear_pre_input_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.res_proj = Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else Identity()

        self.gat = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.gat.append(
                GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout_p)
            )
            self.norms.append(BatchNorm((hidden_dim // heads) * heads))

        self.lins = ModuleList([
            Linear((hidden_dim // heads) * heads, hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            Linear(hidden_dim // 2, 1)
        ])

    def forward(self, x, edge_index):
        if self.dna_proj is not None:
            x_other = x[:, :self.num_other_features]
            x_dna = x[:, self.num_other_features:]

            x_dna = self.dna_proj(x_dna)
            x = torch.cat([x_other, x_dna], dim=1)

        h = self.linear_pre(x)

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (conv, norm) in enumerate(zip(self.gat, self.norms)):
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
    
class GATv2Conv_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2, use_dnabert_proj=True, dnabert_proj_dim=128):
        super(GATv2Conv_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dnabert_proj_dim = dnabert_proj_dim
        self.use_dnabert_proj = use_dnabert_proj
        self.num_total_features = len(dataset.feature_names)
        self.num_dna_features = len(dataset.dnabert_features)
        self.num_other_features = self.num_total_features - self.num_dna_features

        if self.use_dnabert_proj:
            self.dna_proj = Sequential(
                Linear(self.num_dna_features, dnabert_proj_dim),
                BatchNorm(dnabert_proj_dim),
                LeakyReLU(),
                Dropout(dropout_p),
            )
            linear_pre_input_dim = self.num_total_features - self.num_dna_features + dnabert_proj_dim
        else:
            self.dna_proj = None
            linear_pre_input_dim = self.num_total_features

        self.linear_pre = Sequential(
            Linear(linear_pre_input_dim, embedding_dim),
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
        if self.dna_proj is not None:
            x_other = x[:, :self.num_other_features]
            x_dna = x[:, self.num_other_features:]

            x_dna = self.dna_proj(x_dna)
            x = torch.cat([x_other, x_dna], dim=1)

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
    
class GIN_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2, use_dnabert_proj=True, dnabert_proj_dim=128):
        super(GIN_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dnabert_proj_dim = dnabert_proj_dim
        self.use_dnabert_proj = use_dnabert_proj
        self.num_total_features = len(dataset.feature_names)
        self.num_dna_features = len(dataset.dnabert_features)
        self.num_other_features = self.num_total_features - self.num_dna_features

        if self.use_dnabert_proj:
            self.dna_proj = Sequential(
                Linear(self.num_dna_features, dnabert_proj_dim),
                BatchNorm(dnabert_proj_dim),
                LeakyReLU(),
                Dropout(dropout_p),
            )
            linear_pre_input_dim = self.num_total_features - self.num_dna_features + dnabert_proj_dim
        else:
            self.dna_proj = None
            linear_pre_input_dim = self.num_total_features

        self.linear_pre = Sequential(
            Linear(linear_pre_input_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.res_proj = Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else Identity()

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim

            gin_mlp = Sequential(
                Linear(in_dim, hidden_dim),
                BatchNorm(hidden_dim),
                LeakyReLU(),
                Dropout(dropout_p),
                Linear(hidden_dim, hidden_dim)
            )

            self.convs.append(GINConv(gin_mlp, train_eps=True))
            self.norms.append(BatchNorm(hidden_dim))

        self.lins = ModuleList([
            Linear((hidden_dim // heads) * heads, hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            Linear(hidden_dim // 2, 1)
        ])

    def forward(self, x, edge_index):
        if self.dna_proj is not None:
            x_other = x[:, :self.num_other_features]
            x_dna = x[:, self.num_other_features:]

            x_dna = self.dna_proj(x_dna)
            x = torch.cat([x_other, x_dna], dim=1)

        h = self.linear_pre(x)

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
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
    
class SAGEConv_Kmer_Classifier(Module):
    def __init__(self, dataset, embedding_dim=512, hidden_dim=768, heads=4, dropout_p=0.4, edge_dropout_p=0.2, num_layers=2, use_dnabert_proj=True, dnabert_proj_dim=128):
        super(SAGEConv_Kmer_Classifier, self).__init__()

        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dnabert_proj_dim = dnabert_proj_dim
        self.use_dnabert_proj = use_dnabert_proj
        self.num_total_features = len(dataset.feature_names)
        self.num_dna_features = len(dataset.dnabert_features)
        self.num_other_features = self.num_total_features - self.num_dna_features

        if self.use_dnabert_proj:
            self.dna_proj = Sequential(
                Linear(self.num_dna_features, dnabert_proj_dim),
                BatchNorm(dnabert_proj_dim),
                LeakyReLU(),
                Dropout(dropout_p),
            )
            linear_pre_input_dim = self.num_total_features - self.num_dna_features + dnabert_proj_dim
        else:
            self.dna_proj = None
            linear_pre_input_dim = self.num_total_features

        self.linear_pre = Sequential(
            Linear(linear_pre_input_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p),
            Linear(embedding_dim, embedding_dim),
            BatchNorm(embedding_dim),
            LeakyReLU(),
            Dropout(dropout_p)
        )

        self.res_proj = Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else Identity()

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.lins = ModuleList([
            Linear((hidden_dim // heads) * heads, hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            Linear(hidden_dim // 2, 1)
        ])

    def forward(self, x, edge_index):
        if self.dna_proj is not None:
            x_other = x[:, :self.num_other_features]
            x_dna = x[:, self.num_other_features:]

            x_dna = self.dna_proj(x_dna)
            x = torch.cat([x_other, x_dna], dim=1)

        h = self.linear_pre(x)

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
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