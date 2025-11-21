from topology.graph import build_sample_graph, k_shortest_paths
from collections import defaultdict

class Simulator:
    def __init__(self, G):
        self.G = G
        self.link_load = defaultdict(float)
        self.flows = {}

    def get_link_load(self, u, v):
        """Safe accessor for link load."""
        return self.link_load.get((u,v), 0.0)

    def install_flow(self, flow_id, path, demand_mbps):
        if flow_id in self.flows:
            raise ValueError(f"flow {flow_id} already installed")
        for u, v in zip(path[:-1], path[1:]):
            if not self.G.has_edge(u, v):
                raise ValueError(f"edge {u}->{v} not in graph")
            self.link_load[(u,v)] += demand_mbps
        self.flows[flow_id] = {'path': list(path), 'demand': float(demand_mbps)}

    def remove_flow(self, flow_id):
        if flow_id not in self.flows:
            # Ideally raise error, but for robust training logs, we can just return
            return
        meta = self.flows.pop(flow_id)
        path = meta['path']
        d = meta['demand']
        for u, v in zip(path[:-1], path[1:]):
            self.link_load[(u,v)] -= d
            if self.link_load[(u,v)] < 1e-9:
                self.link_load[(u,v)] = 0.0

    def probe_latency(self, path):
        total_latency = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if not self.G.has_edge(u, v): return float('inf')
            cap = self.G[u][v]['capacity']
            base = self.G[u][v]['base_delay']
            load = self.link_load[(u,v)]
            if cap <= 0: return float('inf')
            utilization = load / cap
            if utilization >= 0.999: return float('inf')
            edge_latency = base / (1.0 - utilization)
            total_latency += edge_latency
        return total_latency
