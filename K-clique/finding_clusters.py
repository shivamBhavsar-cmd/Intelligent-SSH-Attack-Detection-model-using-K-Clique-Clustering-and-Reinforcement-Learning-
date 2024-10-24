def find_k_cliques(G, k):
    cliques = list(nx.find_cliques(G))
    k_cliques = [clique for clique in cliques if len(clique) >= k]
    return k_cliques

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

k = 3  # Example value of k
clusters = k_clique_percolation(G, k)

print("Clusters found using k-clique percolation:")
for cluster in clusters:
    print(cluster)