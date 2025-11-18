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
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
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

    # in cartpole, state is already represented as a vector
    # I = torch.eye(n_states, device=device) 

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_steps = 0

        state = torch.tensor(state, device=device)
        for step in range(max_steps):
            total_steps = step

            # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # so this implies that its not backpropagating during action selection
                    action = q_network(state.unsqueeze(0)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, device=device)

            buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            # training step
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s, a, r, ns, d = zip(*batch)

                s      = torch.stack(s)
                a      = torch.tensor(a,  device=device, dtype=torch.long) # actions are deterministic indices
                r      = torch.tensor(r,  device=device, dtype=torch.float32)
                ns     = torch.stack(ns)
                d      = torch.tensor(d,  device=device, dtype=torch.float32) # float32 are best for broadcasting

                q_values = q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_network(ns).max(1)[0]
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
    env = gym.make("CartPole-v1", render_mode="human")
    print(env.observation_space, env.observation_space.shape)
    print(env.action_space)

    weights = torch.load("deep_q_learning_cartpole.pth")
    if env.observation_space.shape is not None:
        n_states = env.observation_space.shape[0]
    else:
        raise NotImplementedError("This environment is not supported yet.")

    q_network = QNetwork(n_states, env.action_space.n, device=device)
    q_network.load_state_dict(weights)
    q_network.eval()


    state, _ = env.reset()
    total_reward = 0
    done = False

    with torch.no_grad():
        while not done:
            s = torch.tensor(state, device=device).unsqueeze(0)
            q_values = q_network(s)
            action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            print(state)
            total_reward += float(reward)
            
            done = terminated or truncated
            time.sleep(0.05)

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

