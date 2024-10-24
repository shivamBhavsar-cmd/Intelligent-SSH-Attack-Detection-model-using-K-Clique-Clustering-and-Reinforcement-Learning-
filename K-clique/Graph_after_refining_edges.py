def remove_unnecessary_edges(G, clusters):
    for cluster in clusters:
        subgraph = G.subgraph(cluster)
        for edge in list(subgraph.edges):
            if not nx.has_path(subgraph, edge[0], edge[1]):
                G.remove_edge(edge[0], edge[1])
    return G

G_refined = remove_unnecessary_edges(G, clusters)

print("Graph after refining edges:")
print(G_refined.edges(data=True))