import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
        G.add_node(ip)
        G.add_node(username)
        # Create edges based on username and IP
        G.add_edge(ip, username)
    return G

# Function to visualize the graph
def visualize_graph(G):
    pos = nx.spring_layout(G)  # Positions for all nodes
    
    plt.figure(figsize=(12, 8))
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title("SSH Brute-Force Attack Log Network Graph")
    plt.show()

# Main execution
if __name__ == '__main__':
    csv_filepath = 'OpenSSH_2k.log_structured(1).csv'  # Replace with your actual CSV file path
    data = read_csv(csv_filepath)
    if data:
        G = preprocess_logs_and_build_graph(data)
        
        # Visualize the graph
        visualize_graph(G)