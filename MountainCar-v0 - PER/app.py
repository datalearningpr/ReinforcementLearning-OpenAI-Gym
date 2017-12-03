import gym
import numpy as np
from PER import PER

env = gym.make('MountainCar-v0')
env = env.unwrapped

def train(num = 500):
    agent = PER(env.observation_space.shape[0], env.action_space.n, e = 0.01)
    batch_size = 32
    # agent.load_model()
    train_start = 0
    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0
        
        while not done:
            step = step + 1
            train_start += 1
            # env.render()
            observation, reward, done, info = env.step(old_action)

            if done:
                reward = 10

            agent.store_train_data(np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action, reward, np.reshape(observation, [1, env.observation_space.shape[0]]), done)
            if train_start > 2000:
                agent.train(batch_size)

            old_observation = observation
            old_action = agent.get_action(np.reshape(observation, [1, env.observation_space.shape[0]]))
            
         
            if done:
                agent.copy_weights()
                steps.append(step)
                print("{}:{} steps: {}".format(i_episode, step, reward))
                agent.save_model()
                break
        
        # if average of steps of consecutive 100 games are lower than a standard, we consider the game passed
        if len(steps) > 100 and sum(steps[-100:])/100 < 110:
            print("---------------------------------------")
            print("done")
            break
        
if __name__ == "__main__":
    train()