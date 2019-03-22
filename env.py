import gym
from ai import DQN

brain = DQN(2, 10000, 50, 0.1)

env = gym.make('CartPole-v0')
reward = 0
for i_episode in range(100000):
    observation = env.reset()
    for t in range(100):
        env.render()
        observation = [observation.tolist()]
        print(observation)
        action = brain.update(observation, reward)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
