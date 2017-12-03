import gym
import numpy as np
from DDQN import DDQN

env = gym.make('Acrobot-v1')

def train(num = 500):
    agent = DDQN(env.observation_space.shape[0], env.action_space.n, e = 0.01)
    batch_size = 32
    # agent.load_model()
    rewards = []
    steps = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))

        step = 0
        while True:
            step = step + 1
            # env.render() # render the game will make the process slow
            observation, reward, done, info = env.step(old_action)

            if done:
                if reward == 0:
                    reward = 100
            
            agent.store_train_data(np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action, reward, np.reshape(observation, [1, env.observation_space.shape[0]]), done)
            if len(agent.memory) > batch_size:
                    agent.train(batch_size)

            old_observation = observation
            old_action = agent.get_action(np.reshape(observation, [1, env.observation_space.shape[0]]))

            if done:                
                print("{}:{} steps: {}".format(i_episode, step, reward))
                rewards.append(1 if reward == 0 else 0)
                steps.append(step)
                agent.copy_weights()
                agent.save_model()
                break
        
        # if the average steps of consecutive 100 games is lower than a standard
        # we consider the method passes the game
        
        score = sum(steps[-100:]) / 100           
        if len(steps) >= 100 and score < 275:
            print("---------------------------------------------------------------")
            print("done")
            break            

if __name__ == "__main__":
    train()