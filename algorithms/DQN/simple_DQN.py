
import torch                                                                                # Imports the PyTorch library.
import torch.nn as nn                                                                       # Imports neural network modules from PyTorch.
import torch.optim as optim                                                                 # Imports optimization algorithms from PyTorch.
import numpy as np                                                                          # Imports NumPy for numerical operations.
import random                                                                               # Imports random module for random number generation.

class QNetwork(nn.Module):                                                                  ### Defines a function approximator neural network that inherits from nn.Module.
    def __init__(self, state_size, action_size) -> None:                                    ## Constructor method fits input and output sizes to state and action spaces.
        """ 
            Initialize the neural network for Q-Learning.
                Args:-------state_size:     int, number of state variables
                            action_size:    int, number of actions
                Returns:----None    
        """
        super(QNetwork, self).__init__()                                                    # Calls nn.Module() class constructor (mandatory).
        self.fc1 = nn.Linear(state_size, 64)                                                # Creates a fully connected layer with state_size inputs and 64 outputs.
        self.fc2 = nn.Linear(64, 64)                                                        # Creates a fully connected layer with 64 inputs and 64 outputs.
        self.fc3 = nn.Linear(64, action_size)                                               # Creates a fully connected layer with 64 inputs and action_size outputs.

    def forward(self, x) -> torch.Tensor:                                                   ## Defines the forward pass method (assign Q-values to actions in this state).
        """ 
            Perform forward pass on Q-network with input x.
                Args:-------state_size:     int, number of state variables
                Returns:----self.fc3(x):    torch.Tensor, output of the third layer 
        """
        x = torch.relu(self.fc1(x))                                                         # Applies ReLU activation to first layer output.
        x = torch.relu(self.fc2(x))                                                         # Applies ReLU activation to second layer output.
        return self.fc3(x)                                                                  # Returns the output of the third layer.


class DQNAgent:                                                                             ### Defines the DQNAgent class that will act and update its network.
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3) -> None:               ## Constructor with default parameters.
        """
            Initialize the DQNAgent class.
                Args:-------state_size:     int, number of state variables
                            action_size:    int, number of actions
                            gamma:          float, discount factor
                            lr:             float, learning rate
                Returns:----None
        """
        self.state_size = state_size                                                        # Stores state size as instance variable.
        self.action_size = action_size                                                      # Stores action size as instance variable.
        self.gamma = gamma                                                                  # Stores gamma value as instance variable.

        self.q_network = QNetwork(state_size, action_size)                                  # Creates main Q-network.
        self.target_network = QNetwork(state_size, action_size)                             # Creates target Q-network.
        self.target_network.load_state_dict(self.q_network.state_dict())                    # Copies weights from Q-network to target network.
        self.target_network.eval()                                                          # Sets target network to evaluation mode (turn off dropout and other when evaluating).

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)                     # Creates Adam optimizer.
        self.criterion = nn.MSELoss()                                                       # Creates Mean Squared Error loss function.

        self.epsilon = 1.0                                                                  # Initializes epsilon value.
        self.epsilon_decay = 0.995                                                          # Sets epsilon decay rate.
        self.epsilon_min = 0.01                                                             # Sets minimum epsilon value.

    def act(self, state) -> int:                                                            ## Method for selecting actions.
        """
            Select an action based on an epsilon-greedy policy.
                Args:-------state:          list, current state
                Returns:----action:         int, action to take
        """
        if random.random() < self.epsilon:                                                  # Checks if random number is less than epsilon.
            return random.randint(0, self.action_size - 1)                                  # Returns if yes, explore: select a random action.
        else:                                                                               # Otherwise, exploit: select the best action.                      
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)         # Converts state to PyTorch tensor
            with torch.no_grad():                                                           # Disables gradient calculation
                q_values = self.q_network(state)                                            # Gets Q-values from network
            return torch.argmax(q_values).item()                                            # Returns action with highest Q-value

    def train_step(self, state, action, reward, next_state, done) -> None:                  ## Training method
        """
            Perform a single training step on the Q-network.
                Args:-------state:          list, current state
                            action:         int, action taken
                            reward:         float, reward received
                            next_state:     list, next state
                            done:           bool, terminal state indicator
                Returns:----None
        """
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)             # Converts inputs to PyTorch tensors and adds batch dimension.
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0)   # ... for state and next state.
        action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)                     # ... for action.
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)                   # ... for reward.
        done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)                       # ... for done (boolean representing end of episode).

        with torch.no_grad():                                                               # Disables gradient calculation (we are just making inference).
            max_next_q_value = self.target_network(next_state).max(dim=1, keepdim=True)[0]  # Gets maximum Q-value for next state.
            q_target = reward + (self.gamma * max_next_q_value * (1 - done))                # Calculates target Q-value.

        q_value = self.q_network(state).gather(1, action)                                   # Gets current Q-value for taken action.
        loss = self.criterion(q_value, q_target)                                            # Calculates loss.

        self.optimizer.zero_grad()                                                          # Zeros out gradients.
        loss.backward()                                                                     # Performs backpropagation on Q-network.
        self.optimizer.step()                                                               # Updates Q-network weights.

    def update_target_network(self) -> None:                                                ## Method to update target network.
        """
            Update the target network with the Q-network weights.
                Args:-------None
                Returns:----None
        """
        self.target_network.load_state_dict(self.q_network.state_dict())                    # Copies weights from Q-network to target network.

    def decay_epsilon(self) -> None:                                                        ## Method to decay epsilon.
        """
            Decay the epsilon value.
                Args:-------None
                Returns:----None
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)             # Updates epsilon value with decay
