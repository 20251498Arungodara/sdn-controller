import random
import pickle
import os
import numpy as np
from collections import defaultdict

class CAMBEAgent:
    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1, tau=0.1, eps_decay=0.99, tau_decay=0.99, seed=42):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps
        self.tau = tau
        self.eps_decay = eps_decay
        self.tau_decay = tau_decay
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Q-table: state -> {action: value}
        self.q_table = defaultdict(lambda: defaultdict(float))

    def select_action(self, state, valid_actions):
        """Select an action from valid_actions using Epsilon-Greedy."""
        if not valid_actions:
            return 0 # Default fallback
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for valid actions only
        q_vals = {a: self.q_table[state][a] for a in valid_actions}
        
        # Find max Q
        if not q_vals:
            return random.choice(valid_actions)
            
        max_q = max(q_vals.values())
        # Break ties randomly among best actions
        best_actions = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_valid_actions):
        """Standard Q-Learning update."""
        current_q = self.q_table[state][action]
        
        # Calculate Max Q for next state (over valid actions)
        if next_valid_actions:
            next_max_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            next_max_q = 0.0
            
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay exploration
        self.epsilon *= self.eps_decay

    def save(self, filepath):
        """Save agent state (Q-table) to pickle."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), data)
