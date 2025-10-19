### NOTE: faulty
def iterative_policy_evaluation(policy, states, theta, gamma, transition_probs, rewards, terminal_states={}):
    """
    Iterative Policy Evaluation algorithm to estimate state values V ≈ v_π
    
    Parameters:
    - policy: Dictionary mapping states to probability distribution over actions
    - states: List of states in the environment
    - theta: Small threshold determining accuracy of estimation
    - gamma: Discount factor
    - transition_probs: Dictionary (state, action) -> list of (probability, next_state)
    - rewards: Dictionary (state, action, next_state) -> reward
    
    Returns:
    - V: Dictionary of estimated state values
    """
    
    # Initialize V(s) arbitrarily for all states except terminal
    V = {state: 0 for state in states}  # Initialize all states to 0
    steps = 0
    while True:
        delta = 0  # Initialize delta for tracking maximum change
        
        # Loop through all states
        for s in states:
            v = V[s]  # Store current state value
            
            # If terminal state, continue (as V(terminal) = 0)
            if s in terminal_states:
                continue
            
            # Calculate new state value using Bellman equation
            new_v = 0
            for a in policy[s]:  # For each action possible in state s
                action_prob = policy[s][a]  # π(a|s)
                
                # Sum over all possible next states
                for prob, s_prime in transition_probs[(s, a)]:
                    reward = rewards[(s, a, s_prime)]
                    new_v += action_prob * prob * (reward + gamma * V[s_prime])
            
            V[s] = new_v  # Update state value
            
            # Track maximum change in value
            delta = max(delta, abs(v - V[s]))
        
        # Check if we can stop
        if delta < theta:
            break
    
        steps+=1

        if steps%100 == 0:
            print("Step: ", steps, " Delta: ", delta)
    return V

# Example usage:
def example_gridworld():
    # Define a simple 3x3 gridworld
    states = [(i, j) for i in range(3) for j in range(3)]
    terminal_states = {(2, 2)}  # Bottom-right corner is terminal
    
    # Simple uniform random policy
    policy = {
        s: {
            'up': 0.25, 'down': 0.25, 
            'left': 0.25, 'right': 0.25
        } for s in states if s not in terminal_states
    }
    
    # Define transition probabilities (deterministic in this case)
    transition_probs = {}
    for s in states:
        if s in terminal_states:
            continue
            
        for a in ['up', 'down', 'left', 'right']:
            i, j = s
            next_i, next_j = i, j
            
            if a == 'up' and i > 0:
                next_i -= 1
            elif a == 'down' and i < 2:
                next_i += 1
            elif a == 'left' and j > 0:
                next_j -= 1
            elif a == 'right' and j < 2:
                next_j += 1
                
            # Probability 1 of reaching next state
            transition_probs[(s, a)] = [(1.0, (next_i, next_j))]
    
    # Define rewards (-1 for each move except reaching terminal)
    rewards = {
        (s, a, s_prime): -1 if s_prime not in terminal_states else 0
        for s in states if s not in terminal_states
        for a in ['up', 'down', 'left', 'right']
        for s_prime in [transition_probs[(s, a)][0][1]]
    }

    # Run policy evaluation
    V = iterative_policy_evaluation(
        policy=policy,
        states=states,
        theta=0.001,
        gamma=0.9,
        transition_probs=transition_probs,
        rewards=rewards,
        terminal_states=terminal_states
    )

    return V

if __name__ == "__main__":
    import numpy as np
    print("Example grid world")
    V = example_gridworld()
    for state, value in V:
        print("state: ", state, "\t value: ", value)
    v = np.array(zip(V))
    np.save('value_fn.npy', v)
