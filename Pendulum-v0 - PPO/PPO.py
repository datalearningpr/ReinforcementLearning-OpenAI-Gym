
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils

# PPO object
class PPO():
    def __init__(self, s_size, action_range):
        self.s_size = s_size
        self.action_range = action_range
        
        self.g = 0.9
        self.e = 0.2

        self.actor = self._build_actor_model()
        self.actor_old = self._build_actor_model()
        self.critic = self._build_critic_model()

        self.actor_training_function = self._build_actor_training_fuction()
        self.critic_training_function = self._build_critic_training_fuction()
        self.critic_advantage_function = self._build_critic_advantage_fuction()

        self.a_steps = 10
        self.c_steps = 10


    def _build_actor_model(self):
        
        inputs = Input(shape=(self.s_size,))

        a = Dense(100, activation = 'relu')(inputs)
        u = Dense(1, activation = 'tanh')(a)
        sigma = Dense(1, activation = 'softplus')(a)
        return Model(inputs = inputs, outputs=[u, sigma])


    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(1))
        return model


    # this is using the OpenAI version, not google version
    def _build_actor_training_fuction(self):
        u = backend.flatten(self.actor.output[0] * 2)
        sigma = backend.flatten(self.actor.output[1])

        u_old = backend.flatten(self.actor_old.output[0] * 2)
        sigma_old = backend.flatten(self.actor_old.output[1])

        pi = 3.141592653589793
        advantage = backend.placeholder(shape=(None,), name="td")
        action = backend.placeholder(shape=(None,), name="action", dtype="float32")

        density = backend.exp(-0.5 * (action - u)**2 / sigma**2) / ((2 * pi * sigma**2)**0.5)
        density_old = backend.exp(-0.5 * (action - u_old)**2 / sigma_old**2) / ((2 * pi * sigma_old**2)**0.5)

        # this is the tricky part of PPO
        # it will balance the advantage and step size of the training
        # if advantage is bigger, we train bigger step but it will have a clip so that it will not go too far away
        # model training can benefit from this balanced framework
        ratio = density / density_old
        surrogate = ratio * advantage
        loss = - backend.mean(backend.minimum(surrogate, backend.clip(ratio, 1-self.e, 1+self.e)*advantage))


        adam = Adam(lr=0.0001)
        updates = adam.get_updates(params=self.actor.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.actor.input,
                                        self.actor_old.input,
                                           action,
                                           advantage], outputs=[], updates=updates)


    # normal td error verison of training critic
    def _build_critic_training_fuction(self):
        old_reward = backend.flatten(self.critic.output)
        reward_holder = backend.placeholder(shape=(None,), name="r")

        advantage = reward_holder - old_reward
        loss = backend.mean(backend.square(advantage))

        adam = Adam(lr=0.0002)
        updates = adam.get_updates(params=self.critic.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.critic.input,
                                           reward_holder], outputs=[advantage], updates=updates)


    # this function will give the advantage that actor training needs
    # not a MUST, you can calculate while training
    def _build_critic_advantage_fuction(self):
        old_reward = backend.flatten(self.critic.output)
        reward_holder = backend.placeholder(shape=(None,), name="r")

        advantage = reward_holder - old_reward
        return backend.function(inputs=[self.critic.input,
                                           reward_holder], outputs=[advantage])

    
    # method to train the model
    def train(self, old_state, action, reward):
        self.actor_old.set_weights(self.actor.get_weights())

        # r = self.critic.predict(old_state)
        # adv = reward - np.squeeze(r) 

        adv = self.critic_advantage_function([old_state, reward])[0]    # using function is not a MUST, can use the code above

        for i in range(self.a_steps):
            self.actor_training_function([old_state, old_state, action, adv])

        for i in range(self.c_steps):
            self.critic_training_function([old_state, reward])



    def get_action(self, state):
        output = self.actor.predict(state)
        u = output[0] * 2
        sigma = output[1]
        return np.clip(np.random.normal(u[0][0],sigma[0][0],1), self.action_range[0], self.action_range[1])


    # save model
    def save_model(self):
        self.actor.save('actor_model.h5')
        self.critic.save('critic_model.h5')

    # load model from archive
    def load_model(self):
        self.actor = load_model('actor_model.h5')
        self.actor_old = load_model('actor_model.h5')
        self.critic = load_model('critic_model.h5')

