import argparse
import pickle
import json
import networkx as nx


def read_files(args):
    with open(args.graph_path, 'rb') as f:
        graph = pickle.load(f)
    with open(args.entities, 'r') as f:
        entities = json.load(f)
    with open(args.relations, 'r') as f:
        rel_dict = json.load(f)
    return graph, entities, rel_dict


def save_graph(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)


def filter_graph(args):
    graph, entities, rel_dict = read_files(args)
    edges_to_add = []
    nodes_to_add = []
    nodes_set = set()
    for u, v, e in graph.edges(data=True):
        if e['pid'] in rel_dict and u in entities:
            if 'entity' in  rel_dict[e['pid']]:
                if v in entities:
                    edges_to_add.append((u,v,e))
                    if not u in nodes_set:
                        nodes_to_add.append((u, graph.nodes[u]))
                        nodes_set.add(u)
                    if not v in nodes_set:
                        nodes_to_add.append((v, graph.nodes[v]))
                        nodes_set.add(v)
            else:
                if not v.startswith("Q"):
                    edges_to_add.append((u,v,e))
                    if not u in nodes_set:
                        nodes_to_add.append((u, graph.nodes[u]))
                        nodes_set.add(u)
                    if not v in nodes_set:
                        nodes_to_add.append((v, graph.nodes[v]))
                        nodes_set.add(v)
        else:
            continue

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes_to_add)
    graph.add_edges_from(edges_to_add)
    save_graph(graph, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, help='Path to unprocessed graph')
    parser.add_argument('--save_path', type=str, help='Path where to save the processed graph')
    parser.add_argument('--entities', type=str, help='Path where entities are stored')
    parser.add_argument('--relations', type=str, help='Path where relations dictionary is stored')
    args, unknown = parser.parse_known_args()

    filter_graph(args)