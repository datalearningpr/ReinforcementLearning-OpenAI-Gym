import gym
import numpy as np
from ActorCritic import ActorCritic

env = gym.make('CartPole-v0')


def train(num = 2000):
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    # agent.load_model()

    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0

        while not done:
            step = step + 1
            # env.render()
            observation, reward, done, info = env.step(old_action)
            
            if done:
                reward = - 20

            td_error = agent.train_critic(reward, np.reshape(old_observation, [1, env.observation_space.shape[0]]), np.reshape(observation, [1, env.observation_space.shape[0]]))
            agent.train_actor(td_error, np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action)

            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))

            if done:
                steps.append(step)
                print("{}:{} steps".format(i_episode, step))
                agent.save_model()
                break



def test(num = 500):
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    agent.load_model()

    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0

        while not done:
            step = step + 1
            # env.render()
            observation, reward, done, info = env.step(old_action)
            
            if done:
                reward = - 20

            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))

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