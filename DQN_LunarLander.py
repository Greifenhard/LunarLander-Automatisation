import gymnasium as gym         # OpenAI Gym
import numpy as np              # NumPy
import torch                    # PyTorch
import torch.nn as nn           # PyTorch
import torch.optim as optim     # PyTorch
from collections import deque   # Deque for experience replay
import random                   # Random number generator
import matplotlib.pyplot as plt # Matplotlib
from tqdm import tqdm           # Progress bar


# Metadata
env_name = 'LunarLander-v3'             # Environment name
Model_Path = './Model/DQN_Model.pt'     # Path to save the model

# Hyperparameters
num_episodes = 2500                     # Number of episodes to train
learning_rate = 0.0005                  # Learning rate for the optimizer
gamma = 0.99                            # Discount factor
epsilon_start = 1.0                     # Initial exploration rate
epsilon_min = 0.05                      # Minimum exploration rate
epsilon_decay = 0.999                   # Decay rate for exploration
batch_size = 128                        # Size of the batch for training
memory_size = 200000                    # Size of the replay memory
update_target_every = 5                 # Number of episodes to update the target network


# Q-Network
class QNetwork(nn.Module):
    """Q-Network for Double DQN."""
    def __init__(self, input_dim, output_dim):
        """Initialize the Q-Network."""
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayMemory:
    """Replay Memory for storing transitions."""
    def __init__(self, memory):
        """Initialize the replay memory."""
        self.memory = deque(maxlen=memory)

    def push(self, transition):
        """Push a transition into the memory."""
        self.memory.append(transition)

    def sample(self, batch_size):
        """Sample a batch of transitions from the memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)

# Doppelte DQN-Agent mit Target Network
class DoubleDQNAgent:
    """Double DQN Agent with Target Network."""
    def __init__(self, obs_dim, n_actions):
        """Initialize the Double DQN Agent."""
        self.q_network = QNetwork(obs_dim, n_actions)
        self.target_network = QNetwork(obs_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma

    def update_target_network(self):
        """Update the target network with the current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def optimize_model(self, batch_size):
        """Optimize the Q-network using a batch of transitions."""
        
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch of transitions from the replay memory
        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
                
        # Convert to tensors
        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        actions_t = torch.tensor(actions).unsqueeze(1)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32))
        next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones_t = torch.from_numpy(np.array(dones, dtype=np.float32))
        
        # Compute the current and target Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            # Use the target network to compute the next Q-values
            next_q_actions = self.q_network(next_states_t).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states_t).gather(1, next_q_actions)

        # Compute the target Q-values
        target_q_values = rewards_t.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones_t.unsqueeze(1)))
        
        # Compute the loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
def deep_q_learning_with_double_dqn():
    """Train a Double DQN agent on the LunarLander-v3 environment."""
    env = gym.make(env_name, gravity=-10.0, render_mode='rgb_array')
    agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)
    epsilon = epsilon_start
    episode_stats = dict(rewards=[], epsilon=[], episodes=[])
    all_rewards = []

    print('Training started...')
    print(f"\n{'Episodes'.center(11)} | {'Mean Reward'.center(10)} | {'Epsilon'.center(18)}")
    print('-' * 45)
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(1000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state)
                    q_values = agent.q_network(state_t)
                    action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.memory.push((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

            agent.optimize_model(batch_size)

        all_rewards.append(total_reward)
        episode_stats['rewards'].append(total_reward)
        episode_stats['epsilon'].append(epsilon)
        episode_stats['episodes'].append(episode + 1)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % update_target_every == 0:
            agent.update_target_network()

        if (episode + 1) % 100 == 0:
            print(f"{episode_stats['episodes'][0]:>4} - {episode_stats['episodes'][-1]:<4} | {np.mean(episode_stats['rewards']):11.5f} | {episode_stats['epsilon'][0]:>5.4f} - {episode_stats['epsilon'][-1]:<5.4f}")
            print('-' * 45)
            episode_stats = dict(rewards=[], epsilon=[], episodes=[])
            
    torch.save(agent.q_network.state_dict(), Model_Path)

    print('\nPlotting results...')
    plt.figure(figsize=(12, 6))
    plt.title('Train Results')
    plt.plot(all_rewards, label='Scores')
    plt.fill_between(range(num_episodes), np.array(all_rewards) - np.std(all_rewards), np.array(all_rewards) + np.std(all_rewards), alpha=0.2)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid()
    plt.show()

def test_agent(num_episodes=100, render_mode='human'):
    env = gym.make(env_name, gravity=-10.0, render_mode=render_mode)
    agent = QNetwork(env.observation_space.shape[0], env.action_space.n)
    agent.load_state_dict(torch.load(Model_Path))
    agent.eval()
    all_rewards = []
    all_steps = []
    
    print("Testing the agent...")
    for _ in tqdm(range(num_episodes), desc="Testing Agent", unit="episode"):
        state, _ = env.reset()
        total_reward, steps = 0, 0

        for _ in range(1000):
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32)
                q_values = agent(state_t)
                action = torch.argmax(q_values).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        all_steps.append(steps)

    env.close()
    
    print('\nTest Results:')
    print(f"\n{'Metric':<30} {'Value':<10}")
    print('-' * 40)
    print(f"{'Average Reward':<30} {np.mean(all_rewards):.4f}")
    print(f"{'Median Reward':<30} {np.median(all_rewards):.4f}")
    print(f"{'Standard Deviation':<30} {np.std(all_rewards):.4f}")
    print(f"{'Minimum Reward':<30} {np.min(all_rewards):.4f}")
    print(f"{'Maximum Reward':<30} {np.max(all_rewards):.4f}")
    print(f"{'Success Rate (>= 200)':<30} {np.mean(np.array(all_rewards) >= 200)*100:.2f} %")
    print('-' * 40)
    print(f"{'Total Episodes':<30} {num_episodes}")
    print(f"{'Total Steps':<30} {sum(all_steps)}")
    print(f"{'Average Steps per Episode':<30} {np.mean(all_steps):.2f}")
    print(f"{'Max Steps':<30} {np.max(all_steps)}")
    print(f"{'Min Steps':<30} {np.min(all_steps)}")
    print('-' * 40)
    print('Test completed.')
    
    print('\nPlotting results...')
    plt.figure(figsize=(12, 6))
    plt.title('Test Results')
    plt.plot(all_rewards, label='Scores')
    plt.plot([np.mean(all_rewards[:x]) for x in range(1, num_episodes+1)], label='Mean Score')
    plt.plot([np.median(all_rewards[:x]) for x in range(1, num_episodes+1)], label='Median Score')
    plt.fill_between(range(num_episodes), np.array(all_rewards) - np.std(all_rewards), np.array(all_rewards) + np.std(all_rewards), alpha=0.2)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid()
    plt.show()


if __name__ == "__main__":

    deep_q_learning_with_double_dqn()

    # Test des trainierten Netzwerks
    test_agent(num_episodes=1000,render_mode='rgb_array')