import random
import torch
from torch import nn
import gymnasium as gym
import numpy as np

# Seeding
SEED = 21
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ANet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Pendulum specific: Action range is [-2, 2]
        # We use tanh to bound it to [-1, 1], then multiply by 2
        mean = torch.tanh(self.mean(x)) * 2.0 
        
        # Exponentiate log_std to get positive std deviation
        std = torch.exp(self.log_std)
        return mean, std

class VNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.v(x)

def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_net = ANet(state_dim, action_dim)
    value_net = VNet(state_dim)

    # Lower Learning rate helps stability in A2C
    actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=1e-4)
    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=1e-3)
    
    num_episodes = 500 # Increased episodes to see convergence
    gamma = 0.99

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 1. Convert state to tensor for the network
            # Keep 'state' variable as numpy for the loop logic
            state_tensor = torch.FloatTensor(state).unsqueeze(0) 

            # 2. Get Action Distribution
            mean, std = actor_net(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            
            # 3. Sample Action
            # rsample allows reparameterization trick, but standard sample is fine for vanilla A2C
            action = dist.sample()
            
            # Clamp action just for environment safety (network already outputs mostly in range)
            action_np = torch.clamp(action, -2.0, 2.0).detach().numpy()[0]
            
            # Calculate log_prob for the gradient
            # sum() is needed if action_dim > 1 (joint probability)
            log_prob = dist.log_prob(action).sum(axis=-1)

            # 4. Step Environment
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

            # 5. Prepare Target Data
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            
            # 6. Calculate Value Loss (Critic)
            # V(s)
            pred_value = value_net(state_tensor).squeeze(-1) # Remove extra dim
            # V(s')
            next_value = value_net(next_state_tensor).squeeze(-1).detach()
            
            # TD Target: r + gamma * V(s') * (1-done)
            td_target = reward_tensor + gamma * next_value * (1 - int(done))
            
            # Critic Loss: MSE(Target, Predicted)
            critic_loss = nn.functional.mse_loss(pred_value, td_target)

            # 7. Calculate Actor Loss
            # TD Error (Advantage): Target - V(s)
            # We detach because we don't want to update Critic via Actor loss
            td_error = td_target - pred_value.detach() 
            
            # Policy Gradient: -log_prob * Advantage
            actor_loss = -log_prob * td_error

            # 8. Update Networks
            loss = actor_loss + 0.5 * critic_loss

            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            value_optimizer.step()

            # 9. Update State
            state = next_state
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

main()
