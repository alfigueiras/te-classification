import networkx as nx
import itertools
from tqdm import tqdm
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def create_digraph_new(nodes_path="", edges_path="", add_in_superbubble_atr=False, zero_column=True, kmers=0, disable_tqdm=False):
    """
    Creates a directed graph using the nodes and edges in the respective paths, using the mouse dataset.
    """
    G = nx.DiGraph()

    if zero_column:
        node_keys=[("unitig", str), ("zeros", int) , ("weight", int), ("lst_genes", str), ("lst_dfam_repeats", str), ("lst_repbase_repeats", str),
                    ("lst_chromosomes_intergenes", str), ("abundance", int), ("AS_genome", int), ("AS_dfam", int)]
    else:
        node_keys=[("unitig", str), ("weight", int), ("lst_genes", str), ("lst_dfam_repeats", str), ("lst_repbase_repeats", str),
                    ("lst_chromosomes_intergenes", str), ("abundance", int), ("AS_genome", int), ("AS_dfam", int)]
        
    if kmers > 0:
        for p in itertools.product(['A','C','G','T'], repeat=kmers):
            node_keys.append((''.join(p), int))

    if nodes_path:
        with open(nodes_path) as node_file:
            for line in node_file:
                if line[0]!="#":
                    node=re.split(r'\t(?!(?<=;))', line)
                    attrs={}
                    for i, (attr,attr_type) in enumerate(node_keys):
                        if attr[:3]=="lst":
                            lst_vals=node[i+1].split(" ")
                            new_lst_vals=lst_vals
                            if attr=="lst_dfam_repeats":
                                if lst_vals!=["*"]:
                                    attrs["is_dfam"]=1
                                    attrs["is_LTR"]=0
                                    new_lst_vals=[]
                                    for te in lst_vals:
                                        if te!="":
                                            new_lst_vals.append(te)
                                        if "LTR" in te:
                                            attrs["is_LTR"]=1
                                            break
                                else:
                                    attrs["is_dfam"]=0
                                    attrs["is_LTR"]=0

                            attrs[attr] = new_lst_vals
                        else:
                            attrs[attr] = attr_type(node[i+1])
                    attrs["node_type"]="simple"
                    G.add_node(int(node[0]), **attrs)

    if edges_path:
        with open(edges_path) as edge_file:
            for line in edge_file:
                edge = line.split()
                G.add_edge(int(edge[0]), int(edge[1]), orientation=edge[2])

    if add_in_superbubble_atr:
        in_superbubble_attr={node: 0 for node in G.nodes}
        in_superbubble_chain_attr={node: 0 for node in G.nodes}
        is_superbubble_boundary_attr={node: 0 for node in G.nodes}
        
        superbubbles=find_superbubbles_directed(G, disable_tqdm=disable_tqdm)
        superbubble_chains=find_bubble_chains(superbubbles)
        superbubble_chains=superbubble_duplicates_remove(superbubble_chains)

        for s_bubble in superbubbles:
            is_superbubble_boundary_attr[s_bubble[0]]=1
            is_superbubble_boundary_attr[s_bubble[1]]=1
            for node in s_bubble[2]:
                in_superbubble_attr[node]=1

        for s_bubble_chain in superbubble_chains:
            for s_bubble in s_bubble_chain:
                for node in s_bubble[2]:
                    in_superbubble_chain_attr[node]=1

        nx.set_node_attributes(G, in_superbubble_attr, name="in_superbubble")
        nx.set_node_attributes(G, in_superbubble_chain_attr, name="in_superbubble_chain")
        nx.set_node_attributes(G, is_superbubble_boundary_attr, name="is_superbubble_boundary")
        
    return G


def find_superbubbles_directed(graph: nx.DiGraph, disable_tqdm=False):
    """
    Returns a list of tuples (s,t,U), where s, t and U are the entrance, exit and inside of a supperbubble, respectively, given a bi-directed graph.
    """
    superbubbles=[]
    for node in tqdm(graph.nodes(), disable=disable_tqdm):
        for direction in ["F","R"]:
            current_direction=direction
            visited=set()
            nodes_inside = []
            seen=set()
            seen.add(node)
            S={(node, current_direction)}

            while len(S)!=0:
                new_n=S.pop()
                n=new_n[0]
                current_direction=new_n[1]
                visited.add(n)
                nodes_inside.append(n)
                if n in seen:
                    seen.remove(n)
                children_in_direction=[]
                
                for edge in graph.out_edges(n, data=True):
                    if edge[2]["orientation"][0] == current_direction:
                        next_orientation=edge[2]["orientation"][1]
                        children_in_direction.append(edge)
                        u=edge[1]
                        
                        if u==node:
                            # Abort it's a cycle pointing back to n
                            S=set() # To exit outer loop too
                            break

                        seen.add(u)
                        all_parents_visited=True
                        for parent in graph.predecessors(u):
                            #tentar com a outra orientação
                            if graph.get_edge_data(parent, u)["orientation"][1]==next_orientation and parent not in visited: # All parents whose edge will end with the next direction
                                all_parents_visited=False
                        if all_parents_visited:
                            # Also give information according to the end letter of the edge orientation  
                            S.add((u, next_orientation))

                if len(children_in_direction)==0:
                    # Abort it's a tip
                    break

                #if len(S)==1 and len(seen)==0:
                if len(S)==1 and len(seen)==1:
                    t=S.pop()
                    nodes_inside.append(t[0])
                    if len(nodes_inside)==2:
                        # Empty Bubble
                        break

                    if not graph.has_edge(t[0], node):
                        superbubbles.append((node, t[0], nodes_inside))
                    else:
                        # Check if the existing edge is valid (has the correct direction that can leave the node)
                        if graph.get_edge_data(t[0], node)["orientation"][0]==current_direction:
                            # There is a valid edge between t and node and therefore there is a cycle including s
                            break
                        else:
                            # The edge is invalid (has orientation different than the current one)
                            superbubbles.append((node, t[0], nodes_inside))
        
    return superbubbles

def find_bubble_chains(bubbles):
    """
    Given a list of bubbles, finds all bubble chains. A bubble chain is a sequence of bubbles where the end of one bubble is the start of the next.
    """
    used_bubbles=[]
    bubble_chains=[]
    for bubble in bubbles:
        if bubble not in used_bubbles:
            chain_found=True
            current_bubble=bubble
            bubble_chain=[current_bubble]
            while chain_found:
                chain_found=False
                for next_bubble in bubbles:
                    # The end of current bubble is the same as the start of the next bubble
                    if current_bubble[1]==next_bubble[0] and current_bubble[0]!=next_bubble[1] and next_bubble not in bubble_chain:
                        used_bubbles.append(next_bubble)
                        bubble_chain.append(next_bubble)
                        current_bubble=next_bubble
                        chain_found=True
                        break
            if len(bubble_chain)>1:
                bubble_chains.append(bubble_chain)
                    
    return bubble_chains

def superbubble_duplicates_remove(bubbles):
    """
    Given a list of superbubbles, removes duplicates (pairs of bubbles where the sink and source are switched, seen).
    """
    superbubbles=[]
    for bubble in bubbles:
        found=False
        for new_bubble in superbubbles:
            if bubble[0]==new_bubble[1] and bubble[1]==new_bubble[0]:
                found=True
                break
        if not found: 
            superbubbles.append(bubble)
    return superbubbles

def draw_bubble(bubble, graph: nx.DiGraph, get_all_parents=False, get_extremity_neighbors=True, node_size=500, font_size=10):
    bubble_start, bubble_end = bubble[0], bubble[1]
    # Make a local copy to avoid modifying the original list
    nodes_to_draw = set(bubble[2])

    if get_all_parents:
        new_nodes_to_draw = set(nodes_to_draw)
        for node in nodes_to_draw:
            for parent in graph.predecessors(node):
                new_nodes_to_draw.add(parent)
        nodes_to_draw = new_nodes_to_draw

    if get_extremity_neighbors:
        for parent in graph.predecessors(bubble_start):
            if parent not in nodes_to_draw:
                nodes_to_draw.add(parent)
                break

        for parent in graph.predecessors(bubble_end):
            if parent not in nodes_to_draw:
                nodes_to_draw.add(parent)
                break

    subgraph = graph.subgraph(nodes_to_draw)
    node_colors = []
    for node in subgraph.nodes():
        if node == bubble_start or node == bubble_end:
            node_colors.append('red')  # Color for bubble_start or bubble_end
        else:
            node_colors.append('lightblue')  # Default color for other nodes

    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, with_labels=True, node_color=node_colors, edge_color='gray', node_size=node_size, font_size=font_size)
    plt.title("Bubble Visualization")
    plt.show()

def draw_bubble_chain(bubble_chain, graph: nx.DiGraph, get_extremity_neighbors, node_size=500, font_size=10):
    nodes_to_draw=set()
    bubble_boundary=set()
    for bubble in bubble_chain:
        bubble_boundary.add(bubble[0])
        bubble_boundary.add(bubble[1])
        nodes_to_draw.update(set(bubble[2]))
    
    if get_extremity_neighbors:
        for parent in graph.predecessors(bubble_chain[0][0]):
            if parent not in nodes_to_draw:
                nodes_to_draw.add(parent)
                break

        for parent in graph.predecessors(bubble_chain[-1][1]):
            if parent not in nodes_to_draw:
                nodes_to_draw.add(parent)
                break

    subgraph = graph.subgraph(nodes_to_draw)
    node_colors = []
    for node in subgraph.nodes():
        if node in bubble_boundary:
            node_colors.append('red')  # Color for bubble_start
        else:
            node_colors.append('lightblue')  # Default color for other nodes
    
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, with_labels=True, node_color=node_colors, edge_color='gray', node_size=node_size, font_size=font_size)
    plt.title("Bubble Chain")
    plt.show()

def weakly_connected_component_subgraphs(graph: nx.DiGraph):
    """
    Given a directed graph, computes all weakly connected components and outputs their respective subgraphs, 
    the nodes belonging to each component and the size of each component.
    """
    weakly_conn_comp=nx.weakly_connected_components(graph)
    components=[]
    component_lens=[]

    for comp in weakly_conn_comp:
        components.append(set(comp))
        component_lens.append(len(comp))
    
    subgraphs=[graph.subgraph(c).copy() for c in components]

    return subgraphs, components, component_lens

def degree_histogram(graph, direction=False, title="Histogram of Node Degrees"):
    """
    Plots an histogram with the degrees of the nodes of the given graph.

    direction: False / "in" / "out"
    For this type of graphs "in degree" = "out degree"
    """
    if direction=="in":
        degrees=[d for n,d in graph.in_degree()]
    elif direction=="out":
        degrees=[d for n,d in graph.out_degree()]
    else:
        degrees=[d for n,d in graph.degree()]
    
    fig = go.Figure()

    histogram = go.Histogram(
        x=degrees,
        nbinsx=max(degrees) - min(degrees) + 1,
        marker=dict(color='cornflowerblue', line=dict(width=1, color='black')),
        showlegend=False,
        histnorm=None,
    )
    fig.add_trace(histogram)

    counts = [degrees.count(d) for d in sorted(set(degrees))]

    max_count=max(counts)

    for i, count in enumerate(counts):
        fig.add_annotation(
            x=sorted(set(degrees))[i], 
            y=count+max_count/20,
            text=str(count),
            showarrow=False,
            font=dict(size=12, color="black"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Degree",
        yaxis_title="Frequency",
        bargap=0.1,
        template="plotly_white",
        xaxis=dict(dtick=1, range=[-0.5, max(degrees) + 0.5]),
        width=700,
        height=400,
    )

    fig.show()

def kk_core_graph(G: nx.DiGraph, k_in: int, k_out: int):
    """
    Searches for (k_in,k_out)-D-cores in the given graph by iteratitvely removing nodes until we have only the cores.
    """
    graph=nx.DiGraph.copy(G)
    remove_nodes_lst=[0]

    # Iterate until we have only nodes with in_degree >= k_in and out_degree >= k_out
    while len(remove_nodes_lst)>0:
        remove_nodes_lst=[]
        for node in graph.nodes:
            if graph.in_degree(node)<k_in or graph.out_degree(node)<k_out:
                remove_nodes_lst.append(node)
        graph.remove_nodes_from(remove_nodes_lst)
    
    return graph

def k_core_graph(G: nx.Graph, k:int):
    """
    Searches for k-cores in the given graph by iteratitvely removing nodes until we have only the cores.
    """
    graph=nx.Graph.copy(G)
    remove_nodes_lst=[0]

    # Iterate until we have only nodes with in_degree >= k_in and out_degree >= k_out
    while len(remove_nodes_lst)>0:
        remove_nodes_lst=[]
        for node in graph.nodes:
            if graph.degree(node)<k:
                remove_nodes_lst.append(node)
        graph.remove_nodes_from(remove_nodes_lst)
    
    return graph


def connected_component_subgraphs(graph):
    """
    Given a directed graph, computes all connected components and outputs their respective subgraphs,
    nodes belonging to each component and the size of each component."""
    conn_comp=nx.connected_components(graph)
    components=[]
    component_lens=[]

    for comp in conn_comp:
        components.append(set(comp))
        component_lens.append(len(comp))
    
    subgraphs=[graph.subgraph(c).copy() for c in components]

    return subgraphs, components, component_lens

def incremental_conductance(graph: nx.Graph, pagerank: list, npy_file_name: str = "conduct_vals_mouse.npy"):
    """
    Computes the incremental conductance of a graph based on the pagerank values of its nodes.
    """

    # Sort nodes by decreasing pagerank value
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)

    cut_edges = 0
    subset_vol = 0
    conductance_vals=[]

    nodes_dict = {node: False for node in graph.nodes()}

    # Incrementally join nodes to the subset and compute the conductance
    for i, node in enumerate(tqdm(sorted_nodes)):
        subset_vol += graph.degree(node)  # Add the node's degree to subset volume

        nodes_dict[node] = True
        # Update cut edges: edges connecting to nodes outside the subset
        for neighbor in graph.neighbors(node):
            if not nodes_dict[neighbor]:  # If the neighbor is not in the subset
                cut_edges += 1
            else:  # If the neighbor is in the subset, remove this as it was a "cut edge"
                cut_edges -= 1
        
        if i%50000==0 and i!=0:
            with open(npy_file_name, 'wb') as f:
                np.save(f,np.array(conductance_vals))

        # Compute conductance
        conductance = cut_edges / subset_vol
        conductance_vals.append(conductance)

    return conductance_vals