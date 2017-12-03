import gym
import numpy as np
from ActorCritic import ActorCritic

env = gym.make('Pendulum-v0')
A_BOUND = env.action_space.high


def train(num = 500):
    agent = ActorCritic(env.observation_space.shape[0], [-A_BOUND, A_BOUND])
    # agent.load_model()

    steps = []
    RENDER = False
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0
        ep_r = 0
        while not done:
            step = step + 1
            if RENDER:
                env.render()
            observation, reward, done, info = env.step(old_action)
            
            reward /= 10

            td_error = agent.train_critic(reward, np.reshape(old_observation, [1, env.observation_space.shape[0]]), np.reshape(observation, [1, env.observation_space.shape[0]]))
            agent.train_actor(td_error, np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action)

            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
            ep_r += reward
            if done:
                print("{} {}".format(i_episode, ep_r))
                if ep_r > -50:
                    RENDER = True
                break

if __name__ == "__main__":
    train()