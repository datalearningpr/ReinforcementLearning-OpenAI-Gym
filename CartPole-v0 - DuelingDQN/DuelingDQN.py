
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Lambda, Input, Activation
from keras.optimizers import Adam
from keras import backend

# DuelingDQN object
class DuelingDQN():
    def __init__(self, s_size, a_size, e = 0.01, g = 0.95, memory_size = 2000, learning_rate = 0.001):
        self.s_size = s_size
        self.a_size = a_size
        self.e = e
        self.e_decay = 0.99

        self.g = g
        self.memory = deque(maxlen = memory_size)
        self.learning_rate = learning_rate

        self.q_model = self._build_model()
        self.target_model = self._build_model()
        self.copy_weights()

    # the model is the main difference between normal DQN and DuelingDQN
    def _build_model(self):

        state_input = Input(shape=(self.s_size,))
        l1 = Dense(32, activation='relu')(state_input)

        l2 = Dense(16)(l1)
        l3 = Dense(16)(l1)

        # for a state input
        # we will have a value output for all the actions
        # and advantage output of all the different specific action

        v = Dense(1)(l2)
        a = Dense(self.a_size)(l3)

        # output is the advantage of a action over the mean of all actions
        # plus the value of all actions
        policy = Lambda(lambda x: x[0]-backend.mean(x[0])+x[1], output_shape = (self.a_size,))([a, v])

        model  = Model(inputs=state_input, outputs=policy)

        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model


    # after certain trainings, we copy the weights from q_model to target model
    # this is one of the tricks DQN uses to stablize the model training process
    def copy_weights(self):
        self.target_model.set_weights(self.q_model.get_weights())

    # store the training data to a memory with fixed size, so that old memory dies out
    # when training data, randomly get data from the memory
    # this is one of the tricks DQN uses to break the correlation between the training data 
    def store_train_data(self, old_state, old_action, reward, state, is_done):
        self.memory.append((old_state, old_action, reward, state, is_done))

    # method to train the model
    def train(self, batch_size):
        # randomly get batch size data from the memory
        minibatch = random.sample(self.memory, batch_size)
        
        for old_state, old_action, reward, state, is_done in minibatch:
            target = self.q_model.predict(old_state)
            if is_done:
                target[0][old_action] = reward
            else:
                # a = self.q_model.predict(state)[0]
                t = self.target_model.predict(state)[0]
                # target[0][old_action] = reward + self.g * t[np.argmax(a)]
                target[0][old_action] = reward + self.g * np.max(t)
            self.q_model.fit(old_state, target, epochs = 1, verbose = 0)

    # get action based on the max Q
    def get_action(self, state):
        if random.random() < self.e:
            return random.choice(range(self.a_size))
        act = self.q_model.predict(state)  
        return np.argmax(act[0])

    # save model
    def save_model(self):
        s = self.q_model.get_weights()
        pickle.dump(s, open("q_model.txt", "wb"))

    # load model from archive
    def load_model(self):
        self.q_model.set_weights(pickle.load(open("q_model.txt", "rb")))
        self.target_model.set_weights(pickle.load(open("q_model.txt", "rb")))
