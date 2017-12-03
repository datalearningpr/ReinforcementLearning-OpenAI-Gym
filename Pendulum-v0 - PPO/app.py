import gym
import numpy as np
from PPO import PPO
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
A_BOUND = env.action_space.high


batch_size = 32
def train(num = 250):
    agent = PPO(env.observation_space.shape[0], [-A_BOUND, A_BOUND])    
    # agent.load_model()

    steps = []
    e_r_list = []
    for i_episode in range(num):
        old_observation = env.reset()
        old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))
        done = False
        step = 0

        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        while not done:
            step = step + 1
            env.render()

            observation, reward, done, info = env.step(old_action)
            
            buffer_s.append(old_observation)
            buffer_a.append(old_action)
            buffer_r.append((reward+8)/8)   # normalisation seems help, not a MUST

            ep_r += reward
            if step % batch_size == 0 or done:
                s_v = agent.critic.predict(np.reshape(observation, [1, env.observation_space.shape[0]]))[0][0]
                d_r = []
                for r in buffer_r[::-1]:
                    s_v = r + agent.g * s_v
                    d_r.append(s_v)
                d_r.reverse()
                
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(d_r)
                buffer_s, buffer_a, buffer_r = [], [], []
                agent.train(bs, np.squeeze(ba), br)
            
            old_observation = observation
            old_action = agent.get_action(np.reshape(old_observation, [1, env.observation_space.shape[0]]))

        print("{} {}".format(i_episode,ep_r))
        e_r_list.append(ep_r)

    plt.plot(np.arange(len(e_r_list)), e_r_list)
    plt.show()

if __name__ == "__main__":
    train()