import numpy as np
import random
import torch
from torch import nn
import gymnasium as gym

# seeding to ensure reproducibility of results
SEED = 21
np.random.seed(SEED)
random.seed(SEED)      # Python's built-in random
torch.manual_seed(SEED)  # PyTorch (CPU)
torch.cuda.manual_seed_all(SEED)  # PyTorch (GPU, if used)

# Actor Network
# Unlike in DQN, the actor network in A2C outputs a probability distribution over actions rather than Q-values.
class ANet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # mean of Gaussian
        self.mean = nn.Linear(128, action_dim)
        
        # std is fixed here
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Pendulum specific: action range is [-2, 2]
        # We use tanh to bound it to [-1, 1], then multiply by 2
        mean = torch.nn.functional.tanh(self.mean(x)) * 2.0  # Scale to action range [-2, 2]
        std = torch.exp(self.log_std)
        return mean, std

# Value Network
class VNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.v(x) #.squeeze(-1)  # [B]

# select action from a probability distribution based on the actor network's output
def select_action(mean, std):
    # state = torch.FloatTensor(state).unsqueeze(0)
    # mean, std = actor(state)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    action = torch.clamp(action, -2.0, 2.0).detach()
    logprob = dist.log_prob(action).sum(axis=-1)
    return action.squeeze(0).detach().numpy(), logprob


def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0] # State is represented by a vector and its continuous
    action_dim = env.action_space.shape[0] # Action is also continuous. 


    # In A2C, we have two separate networks: the actor network and the value network.
    # A2C uses these two networks to learn both the policy (actor) and the value function (critic) simultaneously and both are independent and updated frequently.
    # It is well known for producing differentiable actions, which is particularly useful in continuous action spaces.
    actor_net = ANet(state_dim, action_dim)
    value_net = VNet(state_dim)

    num_episodes = 500
    gamma = 0.99

    actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=1e-3)
    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0) # [1, C]

        done = False
        action = env.action_space.sample() # initial random action
        while not done:
            # A state within Pendulum-v1 is represented by a 3-dimensional vector
            # 1. cos(theta): The cosine of the angle of the pendulum from the upright position.
            # 2. sin(theta): The sine of the angle of the pendulum from the
            # 3. theta_dot: The angular velocity of the pendulum.
            # Similarly, The action is continuous but a dimensional vector, representing the torque applied to the pendulum between -2 and +2
            # And the reward, as continuous value, is designed to encourage the pendulum to stay upright and minimize energy usage.
            # the reawrd function is defined as: reward = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
            # where theta is the angle of the pendulum from the upright position, theta_dt is the angular velocity, and action is the torque applied.
            # so in essence, the reward is higher when the pendulum is upright (theta close to 0) and when less torque is used (action close to 0).
            # The worst reward is -16.273604400000003 when the pendulum is hanging straight down and no action is taken.
            # And the optimal reward is 0 when the pendulum is perfectly upright and no action is taken.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # action selection 
            # sampling the next action is done discre
            reward = torch.FloatTensor(np.array(reward)).unsqueeze(0) # [1, 1]
            next_state = torch.FloatTensor(next_state).unsqueeze(0) # [1, C]
            done = torch.FloatTensor([done]).unsqueeze(0) # [1, 1] float is best for calculations
            mean, std = actor_net(state)
            action, logprob = select_action(mean, std)

            value_reward = value_net(state)
            value_next_reward = value_net(next_state)

            # td_target (pred_value_next_reward) the predicted reward of the next state based on RL Bellman equation 
            td_target = reward + gamma * value_next_reward * (1 - done)
            td_error = td_target - value_reward

            # actor loss = -log π(a|s) * A(s,a)
            actor_loss = (-logprob * td_error.detach()).view(-1)

            value_reward = value_reward.view(-1)
            td_target = td_target.view(-1)

            # critic loss = (R - V(s))^2
            critic_loss = torch.mean((value_reward - td_target) ** 2)


            loss = actor_loss + .5 * critic_loss


            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            value_optimizer.step()

            state = next_state
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {reward.item():.2f}")



main()
