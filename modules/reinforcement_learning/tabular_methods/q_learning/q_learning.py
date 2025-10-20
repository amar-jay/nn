import time
import gymnasium as gym
import numpy as np

def train():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0  # START HIGH for exploration
    epsilon_decay = 0.995
    epsilon_min = 0.01
    n_episodes = 10000
    max_steps = 100
    
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_steps = 0
        episode_reward = 0
        
        for _ in range(max_steps):
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # FIXED Q-learning update
            if done:
                td_target = reward  # No future rewards for terminal states
            else:
                best_next_action = np.argmax(q_table[next_state])
                td_target = reward + gamma * q_table[next_state, best_next_action]
            
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            
            total_steps += 1
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Steps: {total_steps:.3f}")
    
    np.save("frozenlake_q_table.npy", q_table)
    print("\nTraining complete! Q-table saved.")

def test():
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    q_table = np.load("frozenlake_q_table.npy")
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.1)
    
    print(f"\nTotal reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    import argparse
    # train / test
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true", help="Train the Q-learning agent")
    argparser.add_argument("--test", action="store_true", help="Test the Q-learning agent")
    args = argparser.parse_args()
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")

