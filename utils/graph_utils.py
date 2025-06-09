import networkx as nx
import itertools
import tqdm
import re

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
        boundary_superbubble_attr={node: 0 for node in G.nodes}
        superbubbles=find_superbubbles_directed(G, disable_tqdm=disable_tqdm)
        for s_bubble in superbubbles:
            boundary_superbubble_attr[s_bubble[0]]=1
            boundary_superbubble_attr[s_bubble[1]]=1
            for node in s_bubble[2]:
                in_superbubble_attr[node]=1

        nx.set_node_attributes(G, in_superbubble_attr, name="in_superbubble")
        nx.set_node_attributes(G, boundary_superbubble_attr, name="boundary_superbubble")
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