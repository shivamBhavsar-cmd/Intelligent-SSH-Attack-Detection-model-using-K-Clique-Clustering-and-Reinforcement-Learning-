import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Function to read CSV data
def read_csv(filepath):
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

# Function to preprocess logs and create a graph
def preprocess_logs_and_build_graph(data):
    G = nx.Graph()
    for entry in data:
        ip = entry['Content'].split(' ')[-1]  # Extracting IP from Content field
        username = entry['Content'].split(' ')[3]  # Extracting username from Content field
        # Add nodes for IPs and usernames
        G.add_node(ip, type='IP')
        G.add_node(username, type='Username')
        # Create edges based on username and IP
        G.add_edge(ip, username)
    return G

# Function to visualize the graph in various forms
def visualize_graph(G):
    plt.figure(figsize=(14, 10))
    
    # 1. Spring layout with nodes colored by type
    plt.subplot(221)
    pos = nx.spring_layout(G)
    node_colors = {'IP': 'skyblue', 'Username': 'lightcoral'}
    node_color = [node_colors[G.nodes[node]['type']] for node in G.nodes]
    nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color=node_color, edge_color='gray')
    plt.title('Spring Layout with Node Types')
    
    # 2. Circular layout with community detection
    plt.subplot(222)
    communities = list(nx.community.greedy_modularity_communities(G))
    cmap = plt.cm.get_cmap('tab20', max(len(communities), 1))
    colors = [cmap(i % len(communities)) for i in range(len(G))]
    pos = nx.circular_layout(G)
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_color=colors[i], node_size=700, cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Circular Layout with Community Detection')
    
    # 3. Kamada-Kawai layout with central nodes highlighted
    plt.subplot(223)
    pos = nx.kamada_kawai_layout(G)
    degree_dict = dict(G.degree(G.nodes()))
    nx.draw_networkx_nodes(G, pos, node_size=[v * 50 for v in degree_dict.values()], node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    central_nodes = [node for node, degree in degree_dict.items() if degree >= 4]  # Example threshold for central nodes
    nx.draw_networkx_nodes(G, pos, nodelist=central_nodes, node_color='salmon', node_size=[v * 50 for v in degree_dict.values() if v >= 4])
    plt.title('Kamada-Kawai Layout with Central Nodes Highlighted')
    
    # 4. Random layout with labels
    plt.subplot(224)
    pos = nx.random_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color='lightgreen', edge_color='gray')
    plt.title('Random Layout with Labels')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    csv_filepath = 'OpenSSH_2k.log_structured(1).csv'  # Replace with your actual CSV file path
    data = read_csv(csv_filepath)
    if data:
        G = preprocess_logs_and_build_graph(data)
        
        # Visualize the graph in various forms
        visualize_graph(G)