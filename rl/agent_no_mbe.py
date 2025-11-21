"""
rl/agent_no_mbe.py
CAMBEAgent variant that uses epsilon-greedy (no Boltzmann softmax exploration).
"""
import random
import pickle
from collections import defaultdict
from typing import Iterable, Hashable, List, Tuple, Optional

class CAMBEAgent:
    def __init__(self, alpha=0.6, gamma=0.9, eps=0.2, tau=0.8,
                 eps_min=0.01, tau_min=0.2, eps_decay=0.9995, tau_decay=0.9995, seed: Optional[int]=None):
        if seed is not None:
            random.seed(seed)
        self.alpha = float(alpha); self.gamma = float(gamma)
        self.eps = float(eps)
        self.eps_min = float(eps_min); self.eps_decay = float(eps_decay)
        self.Q = defaultdict(float)

    @staticmethod
    def _state_key(state):
        if isinstance(state, tuple): return state
        try: return tuple(state)
        except: return (str(state),)

    def get_q(self, state, action):
        return float(self.Q[(self._state_key(state), action)])
    def set_q(self, state, action, value):
        self.Q[(self._state_key(state), action)] = float(value)

    def _argmax_actions(self, state, actions: Iterable[Hashable]):
        s = self._state_key(state)
        best = None; best_actions = []
        for a in actions:
            q = self.Q[(s, a)]
            if (best is None) or (q > best + 1e-12):
                best = q; best_actions = [a]
            elif abs(q - best) <= 1e-12:
                best_actions.append(a)
        return best_actions

    def select_action(self, state, Avalid: List[Hashable]):
        if not Avalid: raise ValueError("Avalid empty")
        if random.random() < self.eps:
            return random.choice(Avalid)
        bests = self._argmax_actions(state, Avalid)
        return random.choice(bests)

    def update(self, state, action, reward, next_state, next_actions: Iterable[Hashable]):
        s = self._state_key(state); s2 = self._state_key(next_state); a = action
        q = self.Q[(s,a)]
        if next_actions:
            q_next = max([self.Q[(s2,a2)] for a2 in next_actions])
        else:
            q_next = 0.0
        target = reward + self.gamma * q_next
        self.Q[(s,a)] = float(q + self.alpha * (target - q))
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        with open(path, 'wb') as fh: pickle.dump({'alpha':self.alpha,'gamma':self.gamma,'eps':self.eps,'Q':dict(self.Q)}, fh)
    def load(self, path):
        with open(path, 'rb') as fh: data=pickle.load(fh)
        self.alpha=data.get('alpha', self.alpha); self.gamma=data.get('gamma', self.gamma)
        self.eps=data.get('eps', self.eps); self.Q=defaultdict(float, data.get('Q', {}))
