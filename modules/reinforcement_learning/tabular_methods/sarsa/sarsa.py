import time
import gymnasium as gym
import numpy as np

def train():
    num_episodes = 10000
    epsilon = 0.5 # epsilon-greedy factor, exploration vs exploitation
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_steps = 100
    gamma = 0.99 # discount factor
    alpha = 0.1 # learning rate

    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        steps = 0
        # Choose INITIAL action using epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            if done:
                td_target = float(reward)
            else:
                td_target = float(reward) + gamma * q_table[next_state, next_action]

            q_table[state, action] = (1- alpha) * q_table[state, action] + alpha * td_target
            steps += 1

            state, action = next_state, next_action
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 1000 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Steps: {steps}")

    np.save("frozenlake_sarsa.npy", q_table)
    print("\nTraining complete! Q-table for SARSA saved.")

def test():
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    q_table = np.load("frozenlake_sarsa.npy")
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
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
    if args.train and args.test:
        print("Please specify only one of --train or --test")
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")
