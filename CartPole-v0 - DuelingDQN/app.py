import gym
import numpy as np
from DuelingDQN import DuelingDQN

env = gym.make('CartPole-v0')


def train(num = 2000):
    agent = DuelingDQN(env.observation_space.shape[0], env.action_space.n, e = 0.01)
    batch_size = 32
    # agent.load_model()
    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = env.action_space.sample()
        done = False
        step = 0
        while not done:
            step = step + 1
            # env.render()
            observation, reward, done, info = env.step(old_action)
            if done:
                reward = -200
            
            agent.store_train_data(np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action, reward, np.reshape(observation, [1, env.observation_space.shape[0]]), done)
            if len(agent.memory) > batch_size:
                    agent.train(batch_size)

            old_observation = observation
            old_action = agent.get_action(np.reshape(observation, [1, env.observation_space.shape[0]]))

            if done:
                steps.append(step)
                print("{}:{} steps".format(i_episode, step))
                agent.copy_weights()
                agent.save_model()
                break
        # if the average steps of consecutive 100 games is lower than a standard
        # we consider the method passes the game
        if len(steps) > 200 and sum(steps[-200:])/200 >=195:
            print(sum(steps[-200:])/200)
            break


def test(num = 500):
    agent = DuelingDQN(env.observation_space.shape[0], env.action_space.n, e = 0)
    batch_size = 32
    agent.load_model()
    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = env.action_space.sample()
        done = False
        step = 0
        while not done:
            step = step + 1
            # env.render()
            observation, reward, done, info = env.step(old_action)
            if done:
                reward = -200

            old_observation = observation
            old_action = agent.get_action(np.reshape(observation, [1, env.observation_space.shape[0]]))

            if done:
                steps.append(step)
                print("{}:{} steps".format(i_episode, step))
                break
        # if the average steps of consecutive 100 games is lower than a standard
        # we consider the method passes the game
        if len(steps) > 200 and sum(steps[-200:])/200 >=195:
            print(sum(steps[-200:])/200)
            break

if __name__ == "__main__":
    test()