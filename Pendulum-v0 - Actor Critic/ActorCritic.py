
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils

# ActorCritic object
class ActorCritic():
    def __init__(self, s_size, action_range):
        self.s_size = s_size
        self.action_range = action_range
        
        self.g = 0.9

        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()

        self.actor_training_function = self._build_actor_training_fuction()
        self.critic_training_function = self._build_critic_training_fuction()


    # if the output is continuous, then we need u and sigma to get a normal distribution output to mimic it
    def _build_actor_model(self):
        
        inputs = Input(shape=(self.s_size,))
        a = Dense(30, activation = 'relu')(inputs)

        u = Dense(1, activation = 'tanh')(a)
        sigma = Dense(1, activation = 'softplus')(a)
        return Model(inputs = inputs, outputs=[u, sigma])


    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(1, activation = None))
        return model


    # training actor will need the td_error from the critic training
    # so that actor knows what direction(loss) to train the model to
    def _build_actor_training_fuction(self):
        u = backend.squeeze(self.actor.output[0] * self.action_range[1], 0)
        sigma = backend.squeeze(self.actor.output[1] + 0.1, 0)

        pi = 3.141592653589793
        td_error = backend.placeholder(shape=(None,), name="td")
        action = backend.placeholder(shape=(None,), name="action", dtype="float32")

        density = backend.exp(-0.5 * (action - u)**2 / sigma**2) / ((2 * pi * sigma**2)**0.5)
        log_action_prob = backend.log(density)
        exp = log_action_prob * td_error
        entropy = 0.5 * (backend.log(2 * pi * sigma * sigma)+1)
        exp += 0.01 * entropy
        loss = - exp
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
        return self.actor_training_function([old_state, action, td[0][0]])
    
    def train_critic(self, rewards, old_state, new_state):
        p = self.critic.predict(new_state)
        return self.critic_training_function([old_state, [rewards], p[0]])


    # method to get the action based on one state
    # different from Q learning is that all are probability
    def get_action(self, state):
        output = self.actor.predict(state)
        u = output[0] * self.action_range[1]
        sigma = output[1] + 0.1

        return np.clip(np.random.normal(u[0][0],sigma[0][0],1), self.action_range[0], self.action_range[1])


    # save model
    def save_model(self):
        self.actor.save('actor_model.h5')
        self.critic.save('critic_model.h5')

    # load model from archive
    def load_model(self):
        self.actor = load_model('actor_model.h5')
        self.critic = load_model('critic_model.h5')

