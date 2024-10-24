# Create graph from parsed logs
def build_graph(parsed_logs):
    G = nx.Graph()
    for username, ip in parsed_logs:
        G.add_node(ip)
        for other_username, other_ip in parsed_logs:
            if ip != other_ip and username == other_username:
                G.add_edge(ip, other_ip)
    return G

G = build_graph(parsed_logs)
print("Graph created with nodes and edges:")
print(G.nodes(data=True))
print(G.edges(data=True))