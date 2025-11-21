import networkx as nx
from itertools import islice

def build_sample_graph():
    """
    Creates a simple Diamond Topology:
      h1 -> s1 -> s2 -> h2 (Fast but congested?)
      h1 -> s3 -> s4 -> h2 (Slow but reliable?)
    With attributes: 'capacity' (Mbps), 'base_delay' (ms)
    """
    G = nx.DiGraph()
    
    # Define edges: (u, v, capacity_mbps, delay_ms)
    links = [
        # Path A (Top): Short but standard
        ('h1', 's1', 10.0, 1.0), 
        ('s1', 's2', 10.0, 1.0), 
        ('s2', 'h2', 10.0, 1.0),

        # Path B (Bottom): High delay
        ('h1', 's3', 10.0, 5.0), 
        ('s3', 's4', 10.0, 5.0), 
        ('s4', 'h2', 10.0, 5.0),
        
        # Cross link
        ('s1', 's3', 5.0, 2.0)
    ]
    
    for u, v, cap, delay in links:
        G.add_edge(u, v, capacity=cap, base_delay=delay)
        
    return G

def k_shortest_paths(G, source, target, k=1, weight=None):
    """
    Finds the k-shortest paths between source and target.
    Args:
        k (int): The number of paths to return.
        weight (str): The edge attribute to use as weight (default None = hop count).
    """
    try:
        # nx.shortest_simple_paths returns a generator of paths ordered by length
        generator = nx.shortest_simple_paths(G, source, target, weight=weight)
        # Slice the generator to get the top k paths
        return list(islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
