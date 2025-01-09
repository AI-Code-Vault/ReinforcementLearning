
import gym
from DQN import DQNAgent

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
episodes = 100

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        agent.train_step(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            agent.update_target_network()
            agent.decay_epsilon()
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            break

env.close()
