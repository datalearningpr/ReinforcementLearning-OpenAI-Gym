import gym
import numpy as np
from DDPG import DDPG
import tensorflow as tf

env = gym.make('Pendulum-v0')

a_bound = env.action_space.high
batch_size = 32
copy_counter = 0

def train(num = 500):
    RENDER = False
    agent = DDPG(env.observation_space.shape[0], a_bound)
    tf.summary.FileWriter("logs/", agent.sess.graph)
    # agent.load_model()

    steps = []
    var = 3         # make the exploration big at the beginning
    for i_episode in range(num):

        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]), env)
        old_action = np.clip(np.random.normal(old_action, var), -2, 2) 

        done = False
        step = 0
        ep_reward = 0

        while not done:
            
            step = step + 1
            if RENDER:          # do not show the game at the begining, too slow if show
                env.render()

            observation, reward, done, info = env.step(old_action)
            reward /= 10
            agent.store_train_data(np.reshape(old_observation, [1, env.observation_space.shape[0]]), old_action, reward, np.reshape(observation, [1, env.observation_space.shape[0]]), done)
            
            if len(agent.memory) > batch_size:
                var *= .9995    # as time passes by, stop big exploration
                agent.train_critic(batch_size)
                agent.train_actor(batch_size)

                if copy_counter % 500 == 0:
                    agent.copy_weights()    # copy weights like we did for DQN, to speed up converage

        
            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]), env)
            old_action = np.clip(np.random.normal(old_action, var), -2, 2) 

            ep_reward += reward
            if done:
                print('{}: {} {}'.format(i_episode, int(ep_reward), var))
                if ep_reward > -30:
                    # if the model is playing the game well, show the game
                    RENDER = True
                break


if __name__ == "__main__":
    train()