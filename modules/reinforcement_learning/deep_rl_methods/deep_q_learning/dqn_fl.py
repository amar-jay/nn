import random
import torch.nn.functional as F
import torch
import numpy as np
import gymnasium as gym
from collections import deque

def ql_train():
    env = gym.make("CartPole-v1")

    num_episodes = 1000
    max_steps = 500
    epsilon = 0.5
    epsilon_decay = 0.995
    epsilon_min = 0.01
    logging_per_eps = 50
    alpha = 0.1 # learning rate
    gamma = 0.95 # discount factor

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_steps = 0
        for step in range(max_steps):
            # epsilon-greedy action selection
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                best_next_action = np.argmax(q_table[next_state, :])
                td_target = float(reward) + gamma * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] = q_table[state, action] + alpha * td_error
            state = next_state

            if done: # update the last step before termination
                total_steps = 1 + step
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % logging_per_eps == 0:
            print(f"Episide: {episode}, Total Steps: {total_steps}, Epsilon: {epsilon:.3f}")

class QNetwork(torch.nn.Module):
    def __init__(self, n_states, n_actions, device=torch.device('cpu')):
        super(QNetwork, self).__init__()
        self.net = torch.nn.ModuleList([
            torch.nn.Linear(n_states, 128, device=device),
            torch.nn.Linear(128, 128, device=device),
        ])
        self.fc_head = torch.nn.Linear(128, n_actions, device=device)
    def forward(self, x):
        for net in self.net:
            x = torch.relu(net(x))
        return self.fc_head(x)

def dqn_train():
    env = gym.make("FrozenLake-v1", is_slippery=False)

    num_episodes = 1000
    max_steps = 500
    epsilon = 0.5
    epsilon_decay = 0.995
    epsilon_min = 0.01
    logging_per_eps = 50
    gamma = 0.95 # discount factor
    target_update_per_step = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(env.observation_space.n, env.action_space.n, device=device)
    target_network = QNetwork(env.observation_space.n, env.action_space.n, device=device)
    target_network.load_state_dict(q_network.state_dict())
    
    I = torch.eye(env.observation_space.n, device=device)

    optimizer = torch.optim.AdamW(q_network.parameters(), lr=1e-3)

    # replay buffer
    buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_steps = 0
        state = torch.tensor(state, device=device)
        for step in range(max_steps):
            # epsilon-greedy action selection
            if np.random.rand() > epsilon:
                action = q_network(I[state].unsqueeze(0)).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = torch.tensor(next_state, device=device)
            buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done: # update the last step before termination
                total_steps = 1 + step
                break

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s, a, r, ns, d = zip(*batch)
                s = torch.stack(s)
                a = torch.tensor(a, device=device, dtype=torch.long)
                r = torch.tensor(r, device=device, dtype=torch.float32)
                ns = torch.stack(ns)
                d = torch.tensor(d, device=device, dtype=torch.float32)

                s_onehot = torch.eye(env.observation_space.n, device=device)[s]
                ns_onehot = torch.eye(env.observation_space.n, device=device)[ns]
                q_values = q_network(s_onehot).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    target_q_values = target_network(ns_onehot).max(1)[0]
                    td_targets = r + gamma * target_q_values * (1 - d)

                loss = F.mse_loss(q_values, td_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if step % target_update_per_step == 0:
                target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % logging_per_eps == 0:
            print(f"Episide: {episode}, Total Steps: {total_steps}, Epsilon: {epsilon:.3f}")
dqn_train()
