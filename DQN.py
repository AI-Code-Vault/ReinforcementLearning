
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

        # Compute Q target
        with torch.no_grad():
            max_next_q_value = self.target_network(next_state).max(dim=1, keepdim=True)[0]
            q_target = reward + (self.gamma * max_next_q_value * (1 - done))

        # Compute current Q value
        q_value = self.q_network(state).gather(1, action)

        # Compute loss
        loss = self.criterion(q_value, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
