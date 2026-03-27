from utils.graph_utils import create_digraph_new
from logs.checkpoint import filter_counter_by_keys

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

    undirected_G=create_digraph_new(node_path, edge_path, add_in_superbubble_atr=True, add_in_local_cluster_atr=True, zero_column=zero_column, kmers=config['k_mers'], disable_tqdm=True, k_core_val=k_core_val)

    # Convert to undirected graph
    # undirected_G=nx.Graph()
    # undirected_G.add_nodes_from(G.nodes(data=True))
    # undirected_G.add_edges_from(G.edges())

    pickle.dump(undirected_G, open(f"data/processed/graph_{config['species']}{str(config['k_mers'])}{config['fam_type']}.pickle", 'wb'))

    print(f"Number of nodes: {undirected_G.number_of_nodes()}")
    print(f"Creating torch dataset.")

    # Create torch dataset and masks for training and testing
    dataset=data_to_torch(undirected_G, config['k_mers'])
    torch.save(dataset, f"data/processed/{config['species']}{str(config['k_mers'])}{config['fam_type']}.pt")

    return dataset, undirected_G


def data_to_torch(undirected_G, k_mers):
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

    g_node_attrs=base_features.copy()

    g_node_attrs+=alg_features

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

   #columns_to_standardize = list(range(4, len(g_node_attrs)))

    dataset = Data(x=x, y=y, edge_index=dataset.edge_index)

    dataset.feature_names = g_node_attrs[:-1]
    dataset.base_features = base_features
    dataset.alg_features = alg_features
    dataset.struct_features = struct_features_names
    dataset.kmer_features = k_mer_features

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

    filter_counter_by_keys(transposable_e, mask)
    return mask

def count_nodes_with_families(G, nodes_subset):
    count = 0
    for node in nodes_subset:
        families = G.nodes[node].get('lst_dfam_repeats', [])
        if families and families != ['*']:
            count += 1
    return count