
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential   
from keras.layers import Dense 
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils

# PolicyGradient object
class PolicyGradient():
    def __init__(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size
        
        self.decay = 0.99

        self.model = self._build_model()
        self.training_function = self._build_training_fuction()


    # use normal 1 hidden layer NN, number of nodes can be changed
    def _build_model(self):
        model = Sequential()
        model.add(Dense(15, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(self.a_size, activation = 'softmax'))
        return model

    # the major difference from DQN is that there is no target and therefore
    # loss has to be defined yourself
    def _build_training_fuction(self):
        action_output = self.model.output
        action_onehot_output = backend.placeholder(shape=(None, self.a_size), name="action")
        discount_reward_input = backend.placeholder(shape=(None,), name="discount_reward")
        action_prob = backend.sum(action_output * action_onehot_output, axis=1)
        log_action_prob = backend.log(action_prob)
        loss = - log_action_prob * discount_reward_input
        loss = backend.mean(loss)
        adam = Adam(lr=0.01)
        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.model.input,
                                           action_onehot_output,
                                           discount_reward_input], outputs=[], updates=updates)


    # an array of rewards, discount based on the factor
    def get_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        reward = 0
        for i in range(len(rewards) - 1, -1, -1):

            reward = reward * self.decay + rewards[i]
            discounted_rewards[i] = reward

        discounted_rewards -= discounted_rewards.mean() 
        discounted_rewards /= discounted_rewards.std()

        return discounted_rewards

    # method to train the model
    def train(self, rewards, actions, states):
        discounted_rewards = self.get_discounted_rewards(rewards)
        action_onehot_output = np_utils.to_categorical(actions, num_classes = self.a_size)
        self.training_function([states, action_onehot_output, discounted_rewards])

    # method to get the action based on one state
    # different from Q learning is that all are probability
    def get_action(self, state):
        action_predict = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.a_size), p = action_predict)


    # save model
    def save_model(self):
        self.model.save('model.h5')

    # load model from archive
    def load_model(self):
        self.model = load_model('model.h5')

