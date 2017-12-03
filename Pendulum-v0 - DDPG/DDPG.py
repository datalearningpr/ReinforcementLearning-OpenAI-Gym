
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Input, Lambda, merge, Activation
from keras.layers.merge import Add, Multiply, concatenate
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils
from keras import initializers
import tensorflow as tf


# DDPG object
class DDPG():
    def __init__(self, s_size, a_bound, memory_size = 2000):
        self.s_size = s_size
        self.a_bound = a_bound
        self.sess = tf.Session()
        backend.set_session(self.sess)  # this is very important! if tf and keras are using at the same time
        self.memory = deque(maxlen = memory_size)

        self.g = 0.9

        self.actor_eval = self._build_actor_model()
        self.actor_target = self._build_actor_model()
        self.critic_eval = self._build_critic_model()
        self.critic_target = self._build_critic_model()
        self.critic_grads = tf.gradients(self.critic_eval.output, self.critic_eval.input[1])

        # this is the tricky point of DDPG
        # training actor is using the gradient of value over action
        # then with the gradient above, actor can then know how to change the weights to output better action for a state to get higher value

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, 1])
        self.actor_grads = tf.gradients(self.actor_eval.output, self.actor_eval.trainable_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, self.actor_eval.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def store_train_data(self, old_state, old_action, reward, state, is_done):
        self.memory.append((old_state, old_action, reward, state, is_done))

    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(1, activation = 'tanh'))
        model.add(Lambda(lambda x: x * self.a_bound))
        return model


    def _build_critic_model(self):
        state_input = Input(shape=(self.s_size,))
        action_input = Input(shape=(1,))
        s1 = Dense(32)(state_input)
        a1 = Dense(32)(action_input)
        merged = Add()([s1, a1])        # tricky part of critic, add them together is good 
        merged_h1 = Dense(16, activation='relu')(merged)

        output = Dense(1, activation='linear')(merged_h1)
        model  = Model(inputs=[state_input,action_input], outputs=output)

        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model
    

    # method to train the model
    def train_critic(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        old_state = np.array([x for x, old_action, reward, state, is_done in minibatch])
        old_action = np.array([x for old_state, x, reward, state, is_done in minibatch])
        state = np.array([x for old_state, old_action, reward, x, is_done in minibatch])
        reward = np.array([x for old_state, old_action, x, state, is_done in minibatch])

        action_ = self.actor_target.predict(np.reshape(state,[32,self.s_size]))
        q_ = self.critic_target.predict([np.reshape(state,[32,self.s_size]), action_])
        q_target = reward + self.g * q_

        self.critic_eval.fit([np.reshape(old_state,[32,self.s_size]), np.reshape(old_action, [32, 1])], np.reshape(np.array([q_target]), [32, 1]), batch_size=32, verbose = 0)
    


    def train_actor(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        old_state = np.array([x for x, old_action, reward, state, is_done in minibatch])
        action = self.actor_eval.predict(np.reshape(old_state,[32,self.s_size]))
        
        # to train the actor, we need to get the gradient first
        temp_grad = self.sess.run(self.critic_grads, feed_dict = {
            self.critic_eval.input[1]: action,
            self.critic_eval.input[0]: np.reshape(old_state,[32,self.s_size])
        })[0]
  
        self.sess.run(self.optimize, feed_dict = {
            self.actor_critic_grad: temp_grad,
            self.actor_eval.input: np.reshape(old_state,[32,self.s_size])
        })


    # special point of DDPG, when choosing the action, it is deterministic
    def get_action(self, state, env):
        return self.actor_eval.predict(state)

    
    def copy_weights(self):
        self.actor_target.set_weights(self.actor_eval.get_weights())
        self.critic_target.set_weights(self.critic_eval.get_weights())


    # save model
    def save_model(self):
        self.actor_eval.save('actor_eval.h5')
        self.critic_eval.save('critic_eval.h5')
        
    # load model from archive
    def load_model(self):
        self.actor_eval = load_model('actor_eval.h5')
        self.actor_target = load_model('actor_eval.h5')

        self.critic_eval = load_model('critic_eval.h5')
        self.critic_target = load_model('critic_eval.h5')

