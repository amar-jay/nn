# created entirely by GPT, for learning purposes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

class GridWorld:
    """4x4 Grid World Environment with terminal states"""
    def __init__(self, size: int = 4):
        self.size = size
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.terminal_states = {(0, 0), (size-1, size-1)}  # Top-left and bottom-right corners
        self.actions = ['up', 'down', 'left', 'right']
        
        # Initialize transition probabilities and rewards
        self.transition_probs = self._create_transition_probs()
        self.rewards = self._create_rewards()
        
    def _create_transition_probs(self) -> Dict:
        """Create transition probability dictionary"""
        probs = {}
        for s in self.states:
            if s in self.terminal_states:
                continue
                
            for a in self.actions:
                next_state = self._get_next_state(s, a)
                probs[(s, a)] = [(1.0, next_state)]  # Deterministic transitions
        return probs
    
    def _create_rewards(self) -> Dict:
        """Create reward dictionary"""
        rewards = {}
        for s in self.states:
            if s in self.terminal_states:
                continue
            for a in self.actions:
                next_state = self._get_next_state(s, a)
                # Reward of +1 for reaching terminal state, -0.1 for other moves
                rewards[(s, a, next_state)] = 1.0 if next_state in self.terminal_states else -0.1
        return rewards
    
    def _get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next state given current state and action"""
        i, j = state
        if action == 'up':
            i = max(0, i-1)
        elif action == 'down':
            i = min(self.size-1, i+1)
        elif action == 'left':
            j = max(0, j-1)
        elif action == 'right':
            j = min(self.size-1, j+1)
        return (i, j)

def iterative_policy_evaluation(env: GridWorld, policy: Dict, theta: float = 0.001, gamma: float = 0.9) -> Dict:
    """
    Iterative Policy Evaluation algorithm
    
    Args:
        env: GridWorld environment
        policy: Dictionary mapping states to action probabilities
        theta: Convergence threshold
        gamma: Discount factor
    
    Returns:
        Dictionary of state values
    """
    # Initialize state values
    V = {state: 0 for state in env.states}
    iteration = 0
    
    while True:
        delta = 0
        iteration += 1
        
        # Update each state
        for s in env.states:
            if s in env.terminal_states:
                continue
                
            v = V[s]
            new_v = 0
            
            # Calculate new state value using Bellman equation
            for a in env.actions:
                action_prob = policy[s][a]
                
                for prob, s_prime in env.transition_probs[(s, a)]:
                    reward = env.rewards[(s, a, s_prime)]
                    new_v += action_prob * prob * (reward + gamma * V[s_prime])
            
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            print(f"Converged after {iteration} iterations")
            break
    
    return V

def create_uniform_policy(env: GridWorld) -> Dict:
    """Create a uniform random policy"""
    return {
        s: {a: 1.0/len(env.actions) for a in env.actions}
        for s in env.states if s not in env.terminal_states
    }

def visualize_values(V: Dict, env_size: int):
    """Visualize state values as a heatmap"""
    values = np.zeros((env_size, env_size))
    for (i, j), v in V.items():
        values[i, j] = v
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(values, annot=True, fmt='.2f', cmap='RdYlBu_r')
    plt.title('State Values under Random Policy')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

def analyze_policy(V: Dict, env: GridWorld):
    """Analyze and print insights about the learned values"""
    # Find states with highest and lowest values
    non_terminal_states = [s for s in env.states if s not in env.terminal_states]
    max_state = max(non_terminal_states, key=lambda s: V[s])
    min_state = min(non_terminal_states, key=lambda s: V[s])
    
    print("\nPolicy Analysis:")
    print(f"Highest value state: {max_state} with value {V[max_state]:.2f}")
    print(f"Lowest value state: {min_state} with value {V[min_state]:.2f}")
    
    # Analyze value gradients
    print("\nValue gradients from center:")
    center = (env.size//2, env.size//2)
    adjacent = [(center[0]-1, center[1]), (center[0]+1, center[1]),
                (center[0], center[1]-1), (center[0], center[1]+1)]
    
    print(f"Center state {center} value: {V[center]:.2f}")
    for adj in adjacent:
        if adj in V:
            diff = V[adj] - V[center]
            print(f"Gradient to {adj}: {diff:.2f}")

# Run the example
def main():
    # Create environment and policy
    env = GridWorld(size=4)
    policy = create_uniform_policy(env)
    
    # Run policy evaluation
    V = iterative_policy_evaluation(env, policy)
    
    # Visualize and analyze results
    visualize_values(V, env.size)
    analyze_policy(V, env)
    
    return V, env

if __name__ == "__main__":
    V, env = main()
    np.save('game.npy', V)
