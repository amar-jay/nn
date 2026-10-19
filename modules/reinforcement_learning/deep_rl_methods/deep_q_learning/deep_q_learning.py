import time
import random
from torch import nn
import gymnasium as gym
import torch
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class QNetwork(nn.Module):
    """
    This is just a simple classical feedforward dense neural network.
    """
    def __init__(self, state_dim, action_dim, device=device):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128, device=device)
        self.fc2 = nn.Linear(128, 128, device=device)
        self.fc3 = nn.Linear(128, action_dim, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train():
    #env = gym.make("FrozenLake-v1", is_slippery=False)
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_network = QNetwork(n_states, n_actions, device=device)
    target_network = QNetwork(n_states, n_actions, device=device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.AdamW(q_network.parameters(), lr=1e-3)

    buffer = deque(maxlen=10000)

    num_episodes = 10000
    epsilon = 0.5
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    max_steps = 100
    target_update_per_eps = 10

    I = torch.eye(n_states, device=device)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_steps = 0

        for step in range(max_steps):
            total_steps = step

            # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_onehot = I[state].unsqueeze(0)
                    # so this implies that its not backpropagating during action selection
                    action = q_network(s_onehot).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            # training step
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s, a, r, ns, d = zip(*batch)

                s      = torch.tensor(s,  device=device)
                a      = torch.tensor(a,  device=device, dtype=torch.long) # actions are deterministic indices
                r      = torch.tensor(r,  device=device, dtype=torch.float32)
                ns     = torch.tensor(ns, device=device)
                d      = torch.tensor(d,  device=device, dtype=torch.float32) # float32 are best for broadcasting

                s_onehot  = I[s]
                ns_onehot = I[ns]

                # indexing the q_values with the actions taken
                q_values = q_network(s_onehot).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_network(ns_onehot).max(1)[0]
                    target = r + gamma * max_next_q * (1 - d)

                loss = F.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_per_eps == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 1000 == 0:
            print(f"Episode {episode}, epsilon={epsilon:.3f}, Total Steps: {total_steps}")

    env.close()
    #torch.save(q_network.state_dict(), "deep_q_learning_frozenlake.pth")
    torch.save(q_network.state_dict(), "deep_q_learning_cartpole.pth")

def test():
    #env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    env = gym.make("CartPole-v1", render_mode="human")
    print(env.observation_space, env.observation_space.shape)
    print(env.action_space)

    weights = torch.load("deep_q_learning_frozenlake.pth")

    q_network = QNetwork(env.observation_space, env.action_space, device=device)
    q_network.load_state_dict(weights)
    q_network.eval()

    I = torch.eye(env.observation_space.n, device=device)

    state, _ = env.reset()
    total_reward = 0
    done = False

    with torch.no_grad():
        while not done:
            s = int(state)
            q_values = q_network(I[s])
            action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            print(state)
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
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")

