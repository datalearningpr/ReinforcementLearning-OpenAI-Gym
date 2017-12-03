
import numpy as np
import random
import pickle
import random
from collections import deque
from keras.models import load_model, Sequential   
from keras.layers import Dense 
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils
import keras

# sumtree will make the weighting update faster
class SumTree:
    pos = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)


    def add(self, p, data):
        idx = self.pos + self.capacity - 1

        self.data[self.pos] = data
        self.update(idx, p)

        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0


    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change


    def get(self, s):
        idx = 0

        while True:
            left_child_idx = 2 * idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                break
            else:
                if s <= self.tree[left_child_idx]:
                    idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    idx = right_child_idx

        data_index = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_index]


    def total(self):
        return self.tree[0]

    

# since PER are using weighted samples
# we have to use new structures to get weighted samples    
class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    max_p = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)


    def add(self, error, sample):
        p = (error + self.e) ** self.a
        self.tree.add(p, sample)
    

    def add(self, sample):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.max_p
        self.tree.add(max_p, sample) 

    
    def sample(self, n):

        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            prob = p / self.tree.total()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights


    def update(self, idx, error):
        error += self.e  
        clipped_errors = np.minimum(error, self.max_p)
        ps = np.power(clipped_errors, self.a)
        for ti, p in zip(idx, ps):
            self.tree.update(ti, p)


# Prioritized Experience Replay DQN object
class PER():
    def __init__(self, s_size, a_size, e = 0.01, g = 0.95, memory_size = 2000, learning_rate = 0.001):
        self.s_size = s_size
        self.a_size = a_size
        self.e = e
        self.e_decay = 0.99

        self.g = g
        self.memory = Memory(memory_size)
        self.learning_rate = learning_rate

        self.abs_errors = []

        self.q_model = self._build_model()
        self.target_model = self._build_model()
        self.training_function = self._build_training_fuction()
        self.copy_weights()
        

    # the model is just a normal 2 layer NN
    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim = self.s_size, activation = 'relu'))
        model.add(Dense(self.a_size, activation = 'linear'))
        return model


    def _build_training_fuction(self):
        
        # this is the key part of PER
        # train the model with samples having different weights

        q_eval = self.q_model.output
        q_target = backend.placeholder(shape=(None, self.a_size), name="q_target")
        ISWeights = backend.placeholder(shape=(None, 1), name="ISWeights")

        # the errors will be used to update the weights of the samples in the memory
        self.abs_errors.append(backend.sum((q_target - q_eval), axis=1))
        loss = backend.mean(ISWeights * backend.square((q_target - q_eval)))
        adam = Adam(lr=0.001)
        updates = adam.get_updates(params=self.q_model.trainable_weights,
                                   loss=loss)

        return backend.function(inputs=[self.q_model.input,
                                           q_target,
                                           ISWeights], outputs=[], updates=updates)



    # after certain trainings, we copy the weights from q_model to target model
    # this is one of the tricks DQN uses to stablize the model training process
    def copy_weights(self):
        self.target_model.set_weights(self.q_model.get_weights())

    # store the training data to a memory with fixed size, so that old memory dies out
    # when training data, randomly get data from the memory
    # this is one of the tricks DQN uses to break the correlation between the training data 
    def store_train_data(self, old_state, old_action, reward, state, is_done):
        self.memory.add(np.array([old_state, old_action, reward, state, is_done]))

    # method to train the model
    def train(self, batch_size):
        # randomly get batch size data from the memory

        # in PER, the samples from memory are having different weights
        idx, minibatch, ISWeights = self.memory.sample(batch_size)
        # for DDQN, use the q_model to get the action and target model to get the max Q
        self.abs_errors = []
        for old_state, old_action, reward, state, is_done in minibatch:
            target = self.q_model.predict(old_state)
            if is_done:
                target[0][old_action] = reward
            else:
                a = self.q_model.predict(state)[0]
                t = self.target_model.predict(state)[0]
                target[0][old_action] = reward + self.g * t[np.argmax(a)]
            self.training_function([old_state, target, ISWeights])
        
        # after training, we use the errors to update the weights of the samples
        self.memory.update(idx, np.array(self.abs_errors))

    # get action based on the max Q
    def get_action(self, state):
        if random.random() < self.e:
            return random.choice(range(self.a_size))
        act = self.q_model.predict(state)  
        return np.argmax(act[0])

    # save model
    def save_model(self):
        self.q_model.save('q_model.h5')

    # load model from archive
    def load_model(self):
        self.q_model = load_model('q_model.h5')
        self.target_model = load_model('q_model.h5')
