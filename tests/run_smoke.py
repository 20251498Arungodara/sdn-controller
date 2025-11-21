# quick test: import modules, build graph, run agent selection
from topology.graph import build_sample_graph, k_shortest_paths
from rl.agent import MBEQAgent

G = build_sample_graph()
print("Graph nodes:", list(G.nodes()))
paths = k_shortest_paths(G, 'h1', 'h2')
print("Sample paths from h1 to h2:", paths)

agent = MBEQAgent()
state = (0,0,0)  # example compact state
actions = ['p1','p2']
choice = agent.select_action(state, actions)
print("Agent chose:", choice)
