import gym
from ReinforcementLearning import QLearing

# actions of this game
actions = []
for i in range(2):
    for j in range(2):
        for k in range(5):
            actions.append((i, j, k))

agent = QLearing(actions, e = 0)
# agent.loadQ()

# load Copy-v0 environment
env = gym.make('Copy-v0')


result = []
for i_episode in range(10000):
    # reset environment
    old_observation = env.reset()
    old_action = agent.getA(old_observation)
    # old_action = (1, 1, old_observation)
    env.render()

    total_reward = 0
    while True:
        
        observation, reward, done, info = env.step(old_action)
        agent.updateQ(old_observation, old_action, observation, reward)
        env.render()

        agent.saveQ()
        old_observation = observation
        action = agent.getA(observation)
        old_action = action
        
        total_reward = total_reward + reward
        if done:
            result.append(total_reward)
            print(i_episode)
            break
    # if the average steps of consecutive 100 games is higher than a standard
    # we consider the method passes the game
    if i_episode > 100 and sum(result[-100:]) / 100 >=25:
        break







