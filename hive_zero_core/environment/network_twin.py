import networkx as nx
import numpy as np
from typing import Dict, Any, List, Set

class NetworkDigitalTwin:
    """
    Simulates a realistic corporate network topology (Web -> App -> DB).
    """
    def __init__(self, num_nodes: int = 20):
        self.graph = nx.DiGraph()
        self.num_nodes = num_nodes
        self._build_topology()

        # State tracking
        self.compromised_nodes: Set[int] = set()
        self.patched_nodes: Set[int] = set()
        self.alert_level = 0.0

    def _build_topology(self):
        for i in range(self.num_nodes):
            tier = "dmz" if i < 5 else "app" if i < 15 else "data"
            vuln_score = 0.8 if tier == "dmz" else 0.5 if tier == "app" else 0.2
            self.graph.add_node(i, tier=tier, vuln_score=vuln_score, compromised=False)

        for i in range(5):
            targets = np.random.choice(range(5, 15), 3, replace=False)
            for t in targets:
                self.graph.add_edge(i, t, proto="http")

        for i in range(5, 15):
            targets = np.random.choice(range(15, 20), 2, replace=False)
            for t in targets:
                self.graph.add_edge(i, t, proto="sql")

    def get_logs(self) -> List[Dict[str, Any]]:
        logs = []
        for u, v, data in self.graph.edges(data=True):
            if np.random.rand() > 0.8:
                logs.append({
                    'src_ip': f"192.168.1.{u}",
                    'dst_ip': f"192.168.1.{v}",
                    'port': 80 if data['proto']=='http' else 3306,
                    'proto': 6
                })
        return logs

    def apply_action(self, action_vector: np.ndarray) -> float:
        target_idx = np.random.randint(0, self.num_nodes)
        node = self.graph.nodes[target_idx]

        attack_strength = np.mean(np.abs(action_vector))
        success_prob = attack_strength * node['vuln_score']

        reward = 0.0

        if np.random.rand() < success_prob:
            if not node['compromised']:
                node['compromised'] = True
                self.compromised_nodes.add(target_idx)
                reward += 10.0 * (3.0 if node['tier'] == 'data' else 1.0)

        self.alert_level += attack_strength * 0.5
        return reward

class BlueTeamAgent:
    def __init__(self, env: NetworkDigitalTwin):
        self.env = env

    def step(self):
        if np.random.rand() > 0.7:
            target = np.random.randint(0, self.env.num_nodes)
            self.env.graph.nodes[target]['vuln_score'] *= 0.5
            self.env.patched_nodes.add(target)

        if self.env.alert_level > 5.0:
            if self.env.compromised_nodes:
                target = list(self.env.compromised_nodes)[0]
                self.env.graph.nodes[target]['compromised'] = False
                self.env.compromised_nodes.remove(target)
                self.env.alert_level *= 0.5
