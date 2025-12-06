import random
import torch
from torch import nn
from collections import deque
import gymnasium as gym

# seeding to ensure reproducibility of results
SEED = 21
random.seed(SEED)      # Python's built-in random
torch.manual_seed(SEED)  # PyTorch (CPU)
torch.cuda.manual_seed_all(SEED)  # PyTorch (GPU, if used)

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0] # State is represented by a vector and its continuous
    action_dim = env.action_space.n # the action is discrete with two possible actions: left (0) and right (1)

    q_net = QNet(state_dim, action_dim)
    target_net = QNet(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_update_per_eps = 10

    buffer = deque(maxlen=10000) # to store experiences for experience replay
    num_episodes = 100
    epsilon = 0.5 # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99 # discount factor

    optimizer = torch.optim.AdamW(q_net.parameters(), lr=1e-3)
    batch_size = 64

    for episode in range(num_episodes):
        state, _ = env.reset()

        done = False
        while not done:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample() # explore
            else:
                s_tensor = torch.FloatTensor(state).unsqueeze(0) # [1, C]
                action = torch.argmax(q_net(s_tensor)).item() # exploit


            # A state within CartPole-v1 is represented by a 4-dimensional vector
            # 1. Cart Position 2. Cart Velocity 3. Pole Angle 4. Pole Velocity At Tip
            # So the shape of state is (4,)
            # The action on the other hand is either 0 (push cart to the left) or 1 (push cart to the right)
            # And the reward, as usual is a scalar value
            # In cartpole, no individual action is inherently better, just the overall sequence of actions that keep the pole balanced,
            # Every time step that the pole is balanced, a reward of +1 is given.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # put in experience replay buffer,
            # in on-policy variants, this is less commin since it only learns from the most recent trajectory and updating immediately
            # However, DQN is off-policy, so we can learn from older experiences, improving sample efficiency and stability, 
            # by randomly samping and breaking correlation between consecutive samples
            buffer.append((state, action, reward, next_state, done))

            state = next_state

            # DQN update
            # The Q Network is updated frequently by sampling mini-batches from the replay buffer, ie., every few time steps (batch_size).
            # However the target network is updated less frequently to provide stable targets during training, and prevent network from chasing its own tail (a moving target)
            if len(buffer) > batch_size:
                batch = random.sample(buffer, batch_size)
                s, a, r, ns, d = zip(*batch)

                state_tensor = torch.FloatTensor(s) # [B, C]
                action_tensor = torch.LongTensor(a).unsqueeze(1) #[B, 1]
                reward_tensor = torch.FloatTensor(r).unsqueeze(1) # [B, 1]
                next_state_tensor = torch.FloatTensor(ns) # [B, C]
                done_tensor = torch.FloatTensor(d).unsqueeze(1) # [B, 1] float is better for calculations, masking and broadcasting

                # indexing the q values with the actions taken
                pred_q_values = q_net(state_tensor).gather(1, action_tensor).squeeze(1) # [B]

                with torch.no_grad():
                    next_q_values = target_net(next_state_tensor).max(1)[0].unsqueeze(1) #[B, 1]
                    actual_q_values = reward_tensor + (1 - done_tensor) * gamma * next_q_values #[B,1]
                    actual_q_values = actual_q_values.squeeze(1) #[B]

                loss = torch.nn.functional.mse_loss(pred_q_values, actual_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # update target network
        if episode % target_update_per_eps == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)






main()
