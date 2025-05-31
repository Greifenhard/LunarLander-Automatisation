import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.autograd import Variable

# Hyperparameters
model_path = './Model/PL_Model.pt'
env_name = 'LunarLander-v3'

num_episodes = 2500
batch_size = 16
gamma = 0.99
learning_rate = 0.01

# Define the neural network model
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax((self.fc3(x)),dim=-1)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Define the training function
def train_policy_gradient():
    env = gym.make(env_name, gravity=-10.0, render_mode='rgb_array')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    policy_net.apply(init_weights)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state)
        reward_all = 0
        
        # Batch History
        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
        
        for _ in range(1000):
            
            probs = policy_net(state)
            m = torch.multinomial(probs, 1).item()
        
            action = m.sample()
            
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward_all += reward 
            
            if terminated or truncated:
                reward = 0
            
            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)
            
            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)
            
            steps += 1
            
            if terminated or truncated:
                break
        
        # Update policy
        if episode > 0 and episode % batch_size == 0:
            
            # Compute the discounted rewards
            R = 0
            for r in reversed(range(steps)):
                if reward_pool[r] == 0:
                    R = 0
                else:
                    R = R * gamma + reward_pool[r]
                    reward_pool[r] = R
                
            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = action_pool[i]
                reward = reward_pool[i]

                probs = policy_net(state)
                m = Categorical(probs)
                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                loss.backward()

            optimizer.step()
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward_all:.2f}")
    
    env.close()
    
    torch.save(policy_net.state_dict(), model_path)
    
train_policy_gradient()







# # Load the model for evaluation
# def load_model(model_path):
#     model = PolicyNetwork(input_size, output_size)
#     model.load_state_dict(torch.load(model_path))

# # Funktion zum Testen und Bewerten des trainierten Modells
# def evaluate_policy(env, model, episodes=20):
#     model.eval()
#     total_rewards = []
#     for ep in range(episodes):
#         state, _ = env.reset()
#         done = False
#         ep_reward = 0
#         while not done:
#             state_tensor = torch.from_numpy(state)
#             with torch.no_grad():
#                 action_probs = model(state_tensor)
#             action = torch.argmax(action_probs, dim=1).item()
#             state, reward, done, _, _ = env.step(action)
#             ep_reward += reward
#         total_rewards.append(ep_reward)
    
#     avg_reward = np.mean(total_rewards)
#     std_reward = np.std(total_rewards)
#     min_reward = np.min(total_rewards)
#     max_reward = np.max(total_rewards)
#     print(f"Evaluation over {episodes} episodes:")
#     print(f"Average Reward: {avg_reward:.2f}")
#     print(f"Std Reward: {std_reward:.2f}")
#     print(f"Min Reward: {min_reward:.2f}")
#     print(f"Max Reward: {max_reward:.2f}")
    
#     return total_rewards

# # Modell laden und bewerten
# eval_env = gym.make('LunarLander-v3', gravity=-10.0, render_mode='rgb_array')
# model.load_state_dict(torch.load(model_path))
# evaluate_policy(eval_env, model, episodes=20)
# eval_env.close()
