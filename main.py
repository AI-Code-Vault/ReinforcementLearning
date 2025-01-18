
import gym                                                                      # Imports the Gym library
from DQN import DQNAgent                                                        # Imports the DQNAgent class

import warnings                                                                 # Imports warnings module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")    # Suppresses gym deprecation warnings

env = gym.make("CartPole-v1")                                                   # Creates CartPole environment
state_size = env.observation_space.shape[0]                                     # Gets size of state space
action_size = env.action_space.n                                                # Gets size of action space

agent = DQNAgent(state_size, action_size)                                       # Creates DQNAgent instance
episodes = 100                                                                  # Sets number of episodes

for episode in range(episodes):                                                 # Loops through episodes
    state, _ = env.reset()                                                      # Resets environment and gets initial state
    total_reward = 0                                                            # Initializes total reward counter

    while True:                                                                 # Continuous loop for each episode
        action = agent.act(state)                                               # Gets action from agent
        next_state, reward, terminated, truncated, _ = env.step(action)         # Takes step in environment
        done = bool(terminated or truncated)                                    # Determines if episode is done

        agent.train_step(state, action, reward, next_state, done)               # Trains the agent

        state = next_state                                                      # Updates current state
        total_reward += reward                                                  # Adds reward to total

        if done:                                                                # Checks if episode is done
            agent.update_target_network()                                       # Updates target network
            agent.decay_epsilon()                                               # Decays epsilon value
            print(f"Episode: {episode},\
Total Reward: {total_reward},\
Epsilon: {agent.epsilon:.3f}")                                                  # Prints episode results
            break                                                               # Breaks the loop

env.close()                                                                     # Closes the environment
