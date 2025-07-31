import re
import argparse
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def parse_ssh_logs(log_file):
    """
    Parses an SSH log file to extract failed login attempts.

    Args:
        log_file (str): Path to the SSH log file.

    Returns:
        list: A list of tuples, where each tuple contains a username and an IP address
              from a failed password attempt.
    """
    # Regex to find failed password attempts for a user from an IP address
    pattern = re.compile(r'Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)')
    parsed_logs = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    username, ip = match.groups()
                    parsed_logs.append((username, ip))
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file}")
        return []
    return parsed_logs

def build_graph_efficiently(parsed_logs):
    """
    Builds a graph from parsed SSH logs. Nodes are IP addresses, and an edge
    exists between two IPs if they were used in failed login attempts with the
    same username.

    Args:
        parsed_logs (list): A list of (username, ip) tuples.

    Returns:
        networkx.Graph: The constructed graph.
    """
    G = nx.Graph()
    # Group IPs by username
    user_to_ips = defaultdict(set)
    for username, ip in parsed_logs:
        user_to_ips[username].add(ip)

    # Add edges between IPs that share a username
    for username, ips in user_to_ips.items():
        ip_list = list(ips)
        for i in range(len(ip_list)):
            for j in range(i + 1, len(ip_list)):
                G.add_edge(ip_list[i], ip_list[j])
    return G


def find_k_clique_clusters(G, k):
    """
    Finds k-clique communities in the graph.

    Args:
        G (networkx.Graph): The graph to analyze.
        k (int): The size of the clique.

    Returns:
        list: A list of sets, where each set is a community (cluster).
    """
    return list(nx.algorithms.community.k_clique_communities(G, k))


def visualize_clusters(G, clusters, output_file):
    """
    Visualizes the graph and its clusters, saving the plot to a file.

    Args:
        G (networkx.Graph): The graph to visualize.
        clusters (list): A list of clusters (communities).
        output_file (str): The path to save the visualization image.
    """
    pos = nx.spring_layout(G, iterations=50)
    plt.figure(figsize=(15, 15))

    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)

    # Highlight clusters with different colors
    cmap = plt.colormaps.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(clusters))]
    for i, cluster in enumerate(clusters):
        nx.draw_networkx_nodes(G, pos, nodelist=list(cluster), node_color=[colors[i]], node_size=500)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"SSH Attack Clusters (k={len(clusters)})")
    plt.savefig(output_file)
    plt.close()
    print(f"Graph visualization saved to {output_file}")


def main():
    """
    Main function to run the SSH log analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze SSH logs to find attack clusters using k-clique percolation.")
    parser.add_argument("logfile", help="Path to the SSH log file (e.g., ssh_logs.txt).")
    parser.add_argument("-k", type=int, default=3, help="The value of k for k-clique clustering (default: 3).")
    parser.add_argument("-o", "--output", default="clusters.png", help="Output file for the graph visualization (default: clusters.png).")
    args = parser.parse_args()

    # 1. Parse logs
    parsed_logs = parse_ssh_logs(args.logfile)
    if not parsed_logs:
        return

    # 2. Build graph
    G = build_graph_efficiently(parsed_logs)

    # 3. Find clusters
    clusters = find_k_clique_clusters(G, args.k)

    print(f"Found {len(clusters)} clusters with k={args.k}:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster}")

    # 4. Visualize clusters
    if clusters:
        visualize_clusters(G, clusters, args.output)
    else:
        print("No clusters to visualize.")

if __name__ == "__main__":
    main()
