from utils.graph_utils import create_digraph_new
from logs.checkpoint import filter_counter_by_keys
from data.dnabert import compute_or_load_dnabert_embeddings

import os 
import random
import torch
import pickle
import itertools
import copy
import numpy as np
import networkx as nx

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

def create_dataset(config):

    if config['species'] == "mouse":
        if config['k_mers'] == 0 and config['fam_type'] == '':
            node_path=f"data/raw/unitig_annotated_mouse.nodes"
        elif config['k_mers'] == 0 and config['fam_type'] == 'Novo':
            node_path=f"data/raw/OutNovoMouse.nodes"
        else:
            node_path=f"data/raw/Out{str(config['k_mers'])}Mers{config['fam_type']}.nodes"
    elif config['species'] == "dog":
        if config['k_mers'] == 0 and config['fam_type'] == '':
            node_path=f"data/raw/unitig_annotated_dog.nodes"
        elif config['k_mers'] == 0 and config['fam_type'] == 'Novo':
            node_path=f"data/raw/OutNovoDog.nodes"
        else:
            node_path=f"data/raw/Out{str(config['k_mers'])}Mers{config['fam_type']}Dog.nodes"
    elif config['species'] == "dro":
        if config['k_mers'] == 0 and config['fam_type'] == '':
            node_path=f"data/raw/unitig_annotated_dro.nodes"
        elif config['k_mers'] == 0 and config['fam_type'] == 'Novo':
            node_path=f"data/raw/OutNovoDRO.nodes"
        else:
            node_path=f"data/raw/Out{str(config['k_mers'])}Mers{config['fam_type']}DRO.nodes"

    edge_path=f"data/raw/unitig_{config['species']}.edges"

    zero_column=config['species']=='mouse'

    if config['species']=='mouse':
        k_core_val=4
    else:
        k_core_val=3

    undirected_G=create_digraph_new(node_path, edge_path, add_in_superbubble_atr=True, add_in_local_cluster_atr=True, add_unitig_entropy_atr=True, zero_column=zero_column, kmers=config['k_mers'], disable_tqdm=True, k_core_val=k_core_val)

    pickle.dump(undirected_G, open(f"data/processed/graph_{config['species']}{str(config['k_mers'])}{config['fam_type']}.pickle", 'wb'))
    # get dnabert embeddings

    sequences=[]
    node_ids=[]

    for node, data in undirected_G.nodes(data=True):
        sequences.append(data["unitig"])
        node_ids.append(node)
    
    embeddings, saved=compute_or_load_dnabert_embeddings(sequences=sequences, save_path=f"data/processed/{config['species']}_dnabert.pt", node_ids=node_ids, force_recompute=config["dnabert_recompute"])

    # just in case the order changed for some reason
    node_ids=saved["node_ids"]

    print(f"Number of nodes: {undirected_G.number_of_nodes()}")
    print(f"Creating torch dataset.")

    # Create torch dataset and masks for training and testing
    dataset=data_to_torch(undirected_G, config['k_mers'], embeddings, node_ids)
    torch.save(dataset, f"data/processed/{config['species']}{str(config['k_mers'])}{config['fam_type']}.pt")

    return dataset, undirected_G


def data_to_torch(undirected_G, k_mers, embeddings, node_ids):
    g_node_attrs=['in_superbubble', 'in_superbubble_chain', 'is_superbubble_boundary', 'in_local_cluster', 'weight', 'abundance']

    base_features = [
        'weight',
        'abundance'
    ]

    alg_features=[
        'in_superbubble',
        'in_superbubble_chain',
        'is_superbubble_boundary',
        'in_local_cluster'
    ]

    entropy_features=['unitig_entropy']

    g_node_attrs=base_features.copy()

    g_node_attrs+=alg_features

    g_node_attrs+=entropy_features

    undirected_G, struct_features_names = structural_features(undirected_G)
    g_node_attrs+=struct_features_names
    k_mer_features=[]
    if k_mers > 0:
        for p in itertools.product(['A','C','G','T'], repeat=k_mers):
            k_mer_features.append(''.join(p))
    
    g_node_attrs+=k_mer_features
            
    g_node_attrs.append('is_dfam')

    dataset = from_networkx(undirected_G, group_node_attrs=g_node_attrs)

    y = dataset.x[:, -1].to(torch.float32)
    x = dataset.x[:, :-1].to(torch.float32)

    # dnabert embeddings
    nx_node_order = list(undirected_G.nodes())

    # map node_id -> embedding
    emb_dict = {nid: emb for nid, emb in zip(node_ids, embeddings)}

    # reorder embeddings to match PyG order
    embeddings_ordered = torch.stack(
        [emb_dict[n] for n in nx_node_order],
        dim=0
    )

    x = torch.cat([x, embeddings_ordered], dim=1)

    dnabert_feature_names = [f"dnabert_{i}" for i in range(embeddings.shape[1])]

    dataset = Data(x=x, y=y, edge_index=dataset.edge_index)

    dataset.feature_names = g_node_attrs[:-1] + dnabert_feature_names
    dataset.base_features = base_features
    dataset.alg_features = alg_features
    dataset.entropy_features = entropy_features
    dataset.struct_features = struct_features_names
    dataset.kmer_features = k_mer_features
    dataset.dnabert_features = dnabert_feature_names

    return dataset


def structural_features(undirected_G):
    # Calculate node degree and clustering coefficient
    degree_dict = dict(undirected_G.degree())
    clustering_dict = nx.clustering(undirected_G)
    
    # Get connected components
    components = list(nx.connected_components(undirected_G))
    node_to_component = {}
    for comp_idx, component in enumerate(components):
        for node in component:
            node_to_component[node] = comp_idx
    
    # Calculate component-level statistics
    component_stats = {}
    for comp_idx, component in enumerate(components):
        comp_degrees = [degree_dict[node] for node in component]
        component_stats[comp_idx] = {
            'min_degree': min(comp_degrees),
            'max_degree': max(comp_degrees),
            'mean_degree': np.mean(comp_degrees),
            'size': len(component)
        }
    
    # Assign features to each node
    features = {}
    for node in undirected_G.nodes():
        comp_idx = node_to_component[node]
        stats = component_stats[comp_idx]
        features[node] = {
            'degree': degree_dict[node],
            'clustering_coefficient': clustering_dict[node],
            'component_min_degree': stats['min_degree'],
            'component_max_degree': stats['max_degree'],
            'component_mean_degree': stats['mean_degree'],
            'component_size': np.log(1+stats['size']) #log since it is extremely skewed because of giant components
        }

    struct_features_names=['degree', 'clustering_coefficient', 'component_min_degree', 'component_max_degree', 'component_mean_degree', 'component_size']
    nx.set_node_attributes(undirected_G, features)
    
    return undirected_G, struct_features_names


def standardize_selected_columns(dataset, train_mask, exclude_feature_names=None, eps=1e-8):
    if exclude_feature_names is None:
        exclude_feature_names = []

    # columns to standardize
    cols_to_standardize = [
        i for i, f in enumerate(dataset.feature_names)
        if f not in exclude_feature_names
    ]

    if len(cols_to_standardize) == 0:
        return dataset, None, None

    cols_to_standardize = torch.tensor(cols_to_standardize, device=dataset.x.device)

    # fit only on training nodes
    x_train = dataset.x[train_mask][:, cols_to_standardize]

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)

    # avoid division by zero
    std = torch.where(std < eps, torch.ones_like(std), std)

    # apply same train statistics to all nodes
    dataset.x[:, cols_to_standardize] = (
        dataset.x[:, cols_to_standardize] - mean
    ) / std

    return dataset, mean, std

def test_standardize(dataset, mean, std, exclude_feature_names):
    """
    Standardizes dataset using pre-computed mean and std values.
    """
    if exclude_feature_names is None:
        exclude_feature_names = []

    # columns to standardize
    cols_to_standardize = [
        i for i, f in enumerate(dataset.feature_names)
        if f not in exclude_feature_names
    ]

    if len(cols_to_standardize) == 0:
        return dataset, None, None
    
    cols_to_standardize = torch.tensor(cols_to_standardize, device=dataset.x.device)
    dataset.x[:, cols_to_standardize] = (
        dataset.x[:, cols_to_standardize] - mean
    ) / std
    return dataset

def merge_pyg_datasets(datasets):
    x_list = []
    y_list = []
    edge_index_list = []
    train_mask_list = []
    test_mask_list = []
    graph_id_list = []

    node_offset = 0

    for graph_idx, data in enumerate(datasets):
        num_nodes = data.num_nodes

        x_list.append(data.x)
        y_list.append(data.y)
        edge_index_list.append(data.edge_index + node_offset)

        train_mask_list.append(data.train_mask)
        test_mask_list.append(data.test_mask)

        graph_id_list.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        node_offset += num_nodes

    merged = Data(
        x=torch.cat(x_list, dim=0),
        y=torch.cat(y_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
    )

    merged.train_mask = torch.cat(train_mask_list, dim=0)
    merged.test_mask = torch.cat(test_mask_list, dim=0)
    merged.graph_id = torch.cat(graph_id_list, dim=0)

    if hasattr(datasets[0], "feature_names"):
        merged.feature_names = datasets[0].feature_names
    if hasattr(datasets[0], "dnabert_features"):
        merged.dnabert_features = datasets[0].dnabert_features
    if hasattr(datasets[0], "base_features"):
        merged.base_features = datasets[0].base_features
    if hasattr(datasets[0], "alg_features"):
        merged.alg_features = datasets[0].alg_features
    if hasattr(datasets[0], "struct_features"):
        merged.struct_features = datasets[0].struct_features
    if hasattr(datasets[0], "kmer_features"):
        merged.kmer_features = datasets[0].kmer_features
    if hasattr(datasets[0], "entropy_features"):
        merged.entropy_features = datasets[0].entropy_features

    torch.save(merged, f"data/processed/all_species.pt")

    return merged

def dataset_choose_single_family(G: nx.Graph, dataset, fam):
    selected_nodes = [node for node, data in G.nodes(data=True) if 'lst_dfam_repeats' in data and fam in data['lst_dfam_repeats']]
    train_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    for node in selected_nodes:
        train_mask[node_id_map[node]] = False
        test_mask[node_id_map[node]] = True
    dataset.train_mask = train_mask
    dataset.test_mask = test_mask
    return dataset 

def dataset_split_by_components(G: nx.Graph, data_full, config):
    """
    Splits the graph into train and test datasets, separating families between them. All the nodes of a family are in the same dataset.
    """
    lst_dfam_repeats_values = [data['lst_dfam_repeats'] for node, data in G.nodes(data=True) if 'lst_dfam_repeats' in data]
    transposable_e=Counter(list(itertools.chain.from_iterable(lst_dfam_repeats_values)))

    # Computes intersections between families, i.e., if there are nodes with two families, there is an intersection between them
    # and they should be in the same dataset. 
    all_fams=sorted(transposable_e.keys(), key=lambda x: transposable_e[x], reverse=True)
    all_fams = [fam for fam in all_fams if fam != "*"]
    intersections = np.zeros((len(all_fams), len(all_fams)), dtype=int)

    for node in lst_dfam_repeats_values:
        for i, j in itertools.combinations(node, 2):
            if i != j:
                intersections[all_fams.index(i)][all_fams.index(j)] = 1
                intersections[all_fams.index(j)][all_fams.index(i)] = 1

    # Based on the intersections, we create a graph where each node is a family and there is an edge between two families if they have an intersection
    adj_matrix=np.array(intersections)
    intersection_G=nx.from_numpy_array(adj_matrix)
    intersection_G=nx.relabel_nodes(intersection_G, {i: all_fams[i] for i in range(len(all_fams))})

    # Creates a list of connected components, where each component is a list of families
    family_conn_components=list(nx.connected_components(intersection_G))

    # Attributes each node of the original graph to a component or as a node without family
    component_nodes = defaultdict(set)
    nodes_no_family = []

    nodes_w_fam=0
    for node, data in G.nodes(data=True):
        families = data.get('lst_dfam_repeats', [])
        if families and families != ['*']:
            nodes_w_fam += 1
            for i, comp in enumerate(family_conn_components):
                if set(families).intersection(comp):
                    component_nodes[i].add(node)
                    break
        else:
            nodes_no_family.append(node)

    total_nodes = len(G.nodes)
    target_nodes_w_fam_in_test = int(config["test_split"] * nodes_w_fam)

    train_nodes = set()
    test_nodes = set()
    train_fams=set()
    test_fams=set()

    # Nodes belonging to families
    components = list(component_nodes.keys())
    random.shuffle(components)

    for comp in components:
        fam_nodes = component_nodes[comp]

        # Only add family if it fits into the target test size
        if len(test_nodes) + len(fam_nodes) <= target_nodes_w_fam_in_test:
            test_nodes.update(fam_nodes)
            test_fams.update(family_conn_components[comp])
        else:
            train_nodes.update(fam_nodes)
            train_fams.update(family_conn_components[comp])

    # Nodes with no family split
    random.shuffle(nodes_no_family)

    # Number of nodes with no family in the test set
    # We want to have 20% of the total nodes in the test set, so we need to add some nodes with no family, 20% total minus the one we already have
    num_no_family_test = int(total_nodes*config["test_split"]) - len(test_nodes)

    train_nodes.update(nodes_no_family[num_no_family_test:])
    test_nodes.update(nodes_no_family[:num_no_family_test])
    
    print(f"Number of nodes in train set: {len(train_nodes)}")
    print(f"Number of nodes in test set: {len(test_nodes)}")

    train_total = len(train_nodes)
    test_total = len(test_nodes)

    train_with_family = count_nodes_with_families(G, train_nodes)
    test_with_family = count_nodes_with_families(G, test_nodes)

    train_family_pct = 100 * train_with_family / train_total
    test_family_pct = 100 * test_with_family / test_total

    print(f"Train set: {train_with_family}/{train_total} nodes ({train_family_pct:.2f}% with TE families)")
    print(f"Test set: {test_with_family}/{test_total} nodes ({test_family_pct:.2f}% with TE families)")

    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    num_nodes = data_full.num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for node in train_nodes:
        train_mask[node_id_map[node]] = True
    for node in test_nodes:
        test_mask[node_id_map[node]] = True

    mask=(train_mask, test_mask, train_fams, test_fams)

    fam_counts=filter_counter_by_keys(transposable_e, mask)
    return mask, fam_counts

def count_nodes_with_families(G, nodes_subset):
    count = 0
    for node in nodes_subset:
        families = G.nodes[node].get('lst_dfam_repeats', [])
        if families and families != ['*']:
            count += 1
    return count

def counts(dataset, mask):
    total = dataset.y[mask].shape[0]
    zeros = (dataset.y[mask] == 0).sum().item()
    ones = (dataset.y[mask] == 1).sum().item()
    print(f"Total: {total}, Zeros: {zeros}, Ones: {ones}")

def random_dataset_split(dataset, config):
    
    # Split each class into training and validation sets
    train_class_idx, val_class_idx = train_test_split(range(dataset.num_nodes), test_size=config["test_split"], random_state=config["seed"])

    # Create boolean masks
    train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)

    train_mask[train_class_idx] = True
    val_mask[val_class_idx] = True

    print('[TRAIN]', end=' ')
    counts(dataset, train_mask)
    print('[VAL]', end=' ')
    counts(dataset, val_mask)

    return train_mask, val_mask