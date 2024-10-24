import re
import networkx as nx

# Step 2: Preprocessing
def parse_logs(logfile):
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

# Step 3: Graph Construction
def build_graph(parsed_logs):
    G = nx.Graph()
    for username, ip in parsed_logs:
        G.add_node(ip)
        for other_username, other_ip in parsed_logs:
            if ip != other_ip and username == other_username:
                G.add_edge(ip, other_ip)
    return G

# Step 4: k-Clique Percolation
def k_clique_percolation(G, k):
    return list(nx.algorithms.community.k_clique_communities(G, k))

# Main execution
if __name__ == '__main__':
    parsed_logs = parse_logs('ssh_logs.txt')
    G = build_graph(parsed_logs)
    k = 3  # Adjust the value of k as needed
    communities = k_clique_percolation(G, k)
    
    for i, community in enumerate(communities):
        print(f"Community {i + 1}: {community}")