import gymnasium as gym
import numpy as np

def train():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    num_episodes = 10000
    epsilon = 0.5
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    alpha = 0.1

    max_steps = 100

    for episode in range(num_episodes):
        state, _ = env.reset()
        steps = 0

        for _ in range(max_steps):
            steps += 1

            # epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = float(reward)
            else:
                # Expected SARSA: build policy and compute expectation
                policy_s = np.ones(n_actions) * (epsilon / n_actions)
                best_action = np.argmax(q_table[next_state])
                policy_s[best_action] += (1.0 - epsilon)

                expected_value = np.dot(policy_s, q_table[next_state])
                td_target = float(reward) + gamma * expected_value

            # Update rule
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            if done:
                break

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} | Steps: {steps}")

    # Save once
    np.save("frozenlake_expected_sarsa.npy", q_table)
    env.close()
    print("\nTraining complete! Q-table for Expected SARSA saved.")


def test():
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    q_table = np.load("frozenlake_expected_sarsa.npy")
    
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

    print(f"Total reward during test: {total_reward}")
    env.close()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--test", action="store_true")
    args = argparser.parse_args()

    if args.train and args.test:
        print("Please specify only one of --train or --test")
    elif args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")
