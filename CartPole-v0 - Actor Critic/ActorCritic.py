
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential   
from keras.layers import Dense 
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils

# ActorCritic object
class ActorCritic():
    def __init__(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size
        
        self.g = 0.9

        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()
        self.actor_training_function = self._build_actor_training_fuction()
        self.critic_training_function = self._build_critic_training_fuction()



    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(self.a_size, activation = 'softmax'))
        return model


    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(1, activation = None))
        return model


    # training actor will need the td_error from the critic training
    # so that actor knows what direction(loss) to train the model to
    def _build_actor_training_fuction(self):
        action_output = self.actor.output
        td_error = backend.placeholder(shape=(None,), name="td")
        action = backend.placeholder(shape=(None,), name="action", dtype="int32")

        log_action_prob = backend.log(action_output[:,action[0]])
        loss = - log_action_prob * td_error
        loss = backend.mean(loss)
        adam = Adam(lr=0.001)
        updates = adam.get_updates(params=self.actor.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.actor.input,
                                           action,
                                           td_error], outputs=[], updates=updates)


    # training critic is clear, just minimize the td error
    def _build_critic_training_fuction(self):
        old_reward = self.critic.output
        reward_holder = backend.placeholder(shape=(None,), name="r")
        new_state_holder = backend.placeholder(shape=(None, self.s_size), name="new_state")
        new_reward_holder = backend.placeholder(shape=(None,), name="new_r")

        td_error = reward_holder + self.g * new_reward_holder - old_reward
        loss = backend.square(td_error) 

        adam = Adam(lr=0.001)
        updates = adam.get_updates(params=self.critic.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.critic.input,
                                           reward_holder,
                                           new_reward_holder], outputs=[td_error], updates=updates)


    # method to train the model
    def train_actor(self, td, old_state, action):
        return self.actor_training_function([old_state, [action], td[0][0]])
    
    def train_critic(self, rewards, old_state, new_state):
        p = self.critic.predict(new_state)
        return self.critic_training_function([old_state, [rewards], p[0]])


    # method to get the action based on one state
    # different from Q learning is that all are probability
    def get_action(self, state):
        action_predict = np.squeeze(self.actor.predict(state))
        return np.random.choice(np.arange(self.a_size), p = action_predict)


    # save model
    def save_model(self):
        self.actor.save('actor_model.h5')
        self.critic.save('critic_model.h5')

    # load model from archive
    def load_model(self):
        self.actor = load_model('actor_model.h5')
        self.critic = load_model('critic_model.h5')

