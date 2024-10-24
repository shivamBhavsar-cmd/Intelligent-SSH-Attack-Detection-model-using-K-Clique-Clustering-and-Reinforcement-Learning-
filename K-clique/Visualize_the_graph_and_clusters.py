import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Preprocess SSH logs from file
def preprocess_logs(logfile):
    with open(logfile, 'r') as f:
        logs = f.readlines()
    
    pattern = re.compile(r'Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)')
    parsed_logs = []
    for log in logs:
        match = pattern.search(log)
        if match:
            username, ip = match.groups()
            parsed_logs.append((username, ip))
    return parsed_logs

# Create graph from parsed logs
def build_graph(parsed_logs):
    G = nx.Graph()
    for username, ip in parsed_logs:
        G.add_node(ip)
        for other_username, other_ip in parsed_logs:
            if ip != other_ip and username == other_username:
                G.add_edge(ip, other_ip)
    return G

# Find k-cliques
def find_k_cliques(G, k):
    cliques = list(nx.find_cliques(G))
    k_cliques = [clique for clique in cliques if len(clique) >= k]
    return k_cliques

# Apply k-clique percolation
def k_clique_percolation(G, k):
    k_cliques = find_k_cliques(G, k)
    clusters = []
    while k_cliques:
        current_clique = k_cliques.pop()
        current_cluster = set(current_clique)
        for clique in k_cliques[:]:
            if len(set(clique) & current_cluster) >= k - 1:
                current_cluster.update(clique)
                k_cliques.remove(clique)
        clusters.append(current_cluster)
    return clusters

# Remove unnecessary edges
def remove_unnecessary_edges(G, clusters):
    for cluster in clusters:
        subgraph = G.subgraph(cluster)
        for edge in list(subgraph.edges):
            if not nx.has_path(subgraph, edge[0], edge[1]):
                G.remove_edge(edge[0], edge[1])
    return G

# Visualize the graph and clusters
def visualize_graph(G, clusters):
    pos = nx.spring_layout(G)  # Positions for all nodes
    
    plt.figure(figsize=(12, 8))
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw clusters with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    for cluster, color in zip(clusters, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=cluster, node_color=[color], node_size=700)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title("SSH Brute-Force Attack Log Clusters")
    plt.show()

# Main execution
if __name__ == '__main__':
    parsed_logs = preprocess_logs('ssh_logs.txt')
    G = build_graph(parsed_logs)
    k = 3  # Example value of k
    clusters = k_clique_percolation(G, k)
    
    print("Clusters found using k-clique percolation:")
    for i, cluster in enumerate(clusters):
        print(f"Community {i + 1}: {cluster}")
    
    G_refined = remove_unnecessary_edges(G, clusters)
    
    print("Graph after refining edges:")
    print(G_refined.edges(data=True))
    
    visualize_graph(G_refined, clusters)