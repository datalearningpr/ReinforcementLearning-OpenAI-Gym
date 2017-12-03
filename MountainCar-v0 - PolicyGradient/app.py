import gym
import numpy as np
from PolicyGradient import PolicyGradient

env = gym.make('MountainCar-v0')

env.seed(2017)
env = env.unwrapped

def train(num = 3000):
    agent = PolicyGradient(env.observation_space.shape[0], env.action_space.n)
    batch_size = 32
    # agent.load_model()
    steps = []

    
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0

        states = []
        actions = []
        rewards = []
        while not done:
            step = step + 1
            # env.render()

            observation, reward, done, info = env.step(old_action)
            states.append(old_observation)
            actions.append(old_action)
            rewards.append(reward)

            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))

            if done:
                steps.append(step)
                print("{}:{} steps: {}".format(i_episode, step, reward))
                agent.train(np.array(rewards), np.array(actions), np.array(states))
                agent.save_model()
                break
        
        # if the average steps of consecutive 100 games is lower than a standard
        # we consider the method passes the game
        if len(steps) > 100 and sum(steps[-100:])/100 < 110:
            print("---------------------------------------")
            print("done")
            break
 



if __name__ == "__main__":
    train()