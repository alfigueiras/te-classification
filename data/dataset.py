from utils.graph_utils import create_digraph_new
from logs.checkpoint import filter_counter_by_keys

import os 
import random
import torch
import pickle
import itertools
import numpy as np
import networkx as nx

from torch_geometric.utils import from_networkx
from collections import Counter, defaultdict

def create_dataset(config):
    node_path=""
    edge_path=""

    zero_column=config['species'].name=='mouse'

    G=create_digraph_new(node_path, edge_path, add_in_superbubble_atr=True, zero_column=zero_column, kmers=config['k_mers'], disable_tqdm=True)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Convert to undirected graph
    undirected_G=nx.Graph()
    undirected_G.add_nodes_from(G.nodes(data=True))
    undirected_G.add_edges_from(G.edges())

    pickle.dump(undirected_G, open(f"data/processed/{config['species']}{config['kmers']}{config['fam_type']}.pickle", 'wb'))

    print(f"Number of nodes: {undirected_G.number_of_nodes()}")
    print(f"Creating torch dataset.")

    # Create torch dataset and masks for training and testing
    dataset=data_to_torch(undirected_G, config['kmers'])
    torch.save(dataset, f"data/processed/{config['species']}{config['kmers']}{config['fam_type']}.pt")

    return dataset, undirected_G


def data_to_torch(undirected_G, k_mers):
    g_node_attrs=['weight', 'in_superbubble', 'abundance']

    if k_mers > 0:
        for p in itertools.product(['A','C','G','T'], repeat=k_mers):
            g_node_attrs.append(''.join(p))
            
    g_node_attrs.append('is_dfam')

    data = from_networkx(undirected_G, group_node_attrs=g_node_attrs)

    dataset = data.clone()
    dataset.y = data.x[:, -1].to(torch.float32)
    dataset.x = data.x[:, :-1]
    dataset.x = dataset.x.to(torch.float32)

    columns_to_standardize = [0, 2]

    dataset=standardize_selected_columns(dataset, columns_to_standardize)

    return dataset    

def standardize_selected_columns(data, columns_to_standardize):
    x = data.x  # Node feature matrix (num_nodes x num_features)

    # Compute mean and std for the selected columns
    mean = x[:, columns_to_standardize].mean(dim=0)
    std = x[:, columns_to_standardize].std(dim=0)

    # Avoid division by zero for columns with zero variance
    std[std == 0] = 1.0

    # Standardize only the selected columns
    x[:, columns_to_standardize] = (x[:, columns_to_standardize] - mean) / std

    data.x = x  # Update the node feature matrix
    return data

def dataset_split_by_components(G: nx.Graph, data_full, number_datasets=1):
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
    masks=[]

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
    target_nodes_w_fam_in_test = int(0.2 * nodes_w_fam)

    for i in range(number_datasets):
        print(f"\nDataset {i + 1}:")

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
        num_no_family_test = int(total_nodes*0.2) - len(test_nodes)

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

        masks.append((train_mask, test_mask, train_fams, test_fams))

    filter_counter_by_keys(transposable_e, masks)
    return masks

def count_nodes_with_families(G, nodes_subset):
    count = 0
    for node in nodes_subset:
        families = G.nodes[node].get('lst_dfam_repeats', [])
        if families and families != ['*']:
            count += 1
    return count