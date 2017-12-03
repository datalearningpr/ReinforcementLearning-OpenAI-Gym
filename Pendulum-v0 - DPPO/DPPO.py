
import pickle
import random
import numpy as np
from collections import deque
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend
from keras import utils as np_utils
import gym, threading, queue
import tensorflow as tf


# this DPPO is not like A3C
# the distributed concept here is just having different thread playing games collecting training data
# the pure PPO only has one agent playing
# there is no concept here like A3C, training is happening in the worker, updating is to the same "father" model

# A3C can also use the "collecting different training data" concept to build



# DPPO object
class DPPO():
    def __init__(self, s_size, action_range):


        self.sess = tf.Session()
        backend.set_session(self.sess)
        # PPO is pruly Keras
        # but we are using threading to train
        # only tf will provide the threads handling, so we have to use tf session

        self.s_size = s_size
        self.action_range = action_range
        
        self.g = 0.9
        self.e = 0.2

        self.actor = self._build_actor_model()
        self.actor._make_predict_function()

        self.actor_old = self._build_actor_model()
        self.actor_old._make_predict_function()
        
        self.critic = self._build_critic_model()
        self.critic._make_predict_function()

        self.actor_training_function = self._build_actor_training_fuction()
        self.critic_training_function = self._build_critic_training_fuction()
        self.critic_advantage_function = self._build_critic_advantage_fuction()

        self.a_steps = 10
        self.c_steps = 10
        
        self.graph = tf.get_default_graph() # we have to use the graph here, otherwise in different thread, they cannot tell which graph to use
        self.sess.run(tf.global_variables_initializer())

    def _build_actor_model(self):
        
        inputs = Input(shape=(self.s_size,))

        a = Dense(100, activation = 'relu')(inputs)
        u = Dense(1, activation = 'tanh')(a)
        sigma = Dense(1, activation = 'softplus')(a)
        return Model(inputs = inputs, outputs=[u, sigma])


    def _build_critic_model(self):
        inputs = Input(shape=(self.s_size,))

        a = Dense(100, activation = 'relu')(inputs)
        v = Dense(1)(a)
        return Model(inputs = inputs, outputs = v)


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



    def _build_critic_advantage_fuction(self):
        old_reward = backend.flatten(self.critic.output)
        reward_holder = backend.placeholder(shape=(None,), name="r")

        advantage = reward_holder - old_reward
        return backend.function(inputs=[self.critic.input,
                                           reward_holder], outputs=[advantage])

    

    # method to train the model, will be called by the different threads
    def train(self):

        global GLOBAL_update_count  # how many times all the 4 workers train in total
        while not COORD.should_stop():
            if GLOBAL_step < 500:
                UPDATE_EVENT.wait()                   

                # copy weights, has to use the graph, otherwise error
                with self.graph.as_default():
                    self.actor_old.set_weights(self.actor.get_weights())

                # get the data 
                data = [TRAINING_QUEUE.get() for i in range(TRAINING_QUEUE.qsize())]     
                old_state = [i[0] for i in data]
                action = [i[1] for i in data]
                reward = [i[2] for i in data]

                # as long as the batch size is not too small
                # there should not be case the queue is empty
                old_state = np.vstack(old_state)
                reward = np.hstack(reward)
                action = np.squeeze(np.vstack(action))
                
                # make the training data suitable for the model to fit before training
                adv = self.critic_advantage_function([old_state, reward])[0]
                for i in range(self.a_steps):
                    self.actor_training_function([old_state, old_state, action, adv])

                for i in range(self.c_steps):
                    self.critic_training_function([old_state, reward])

                # stop update model
                UPDATE_EVENT.clear()        
                GLOBAL_update_count = 0
                # restart collecting data  
                ROLLING_EVENT.set() 



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





# typical workers in different threads
class Worker(object):
    def __init__(self, id, father):
        self.id = id
        self.env = gym.make('Pendulum-v0')
        self.agent = father # this is the father PPO

    def work(self):
        global GLOBAL_step, GLOBAL_update_count
        while not COORD.should_stop():

            old_observation = self.env.reset()
            ep_r = 0
            done = False
            buffer_s, buffer_a, buffer_r = [], [], []
            while not done:
                # if collecting data mode is disabled, then wait 
                if not ROLLING_EVENT.is_set():                  
                    ROLLING_EVENT.wait()                       
                    buffer_s, buffer_a, buffer_r = [], [], []  
                
                # else, start collecting data
                old_action = self.agent.get_action(np.reshape(old_observation, [1, 3]))
                observation, reward, done, info = self.env.step(old_action)

                buffer_s.append(old_observation) 
                buffer_a.append(old_action)
                buffer_r.append((reward+8)/8)   # normalisation seems help
                
                old_observation = observation
                ep_r += reward

                GLOBAL_update_count += 1    

                if done or GLOBAL_update_count >= 32:    
                    # collecting data like normal PPO playing
                    v_s_ = self.agent.critic.predict(np.reshape(observation, [1, 3]))[0][0]
                    discounted_r = []                           
                    for r in buffer_r[::-1]:
                        v_s_ = r + 0.9 * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # key point of DPPO, workers collects data but only put to queue for father model to train
                    TRAINING_QUEUE.put([bs, ba, br])    
                    
                    # batch size reached, then train mode
                    if GLOBAL_update_count >= 32:
                        ROLLING_EVENT.clear()       
                        UPDATE_EVENT.set()          

                    # if max game play reached, then stop playing
                    if GLOBAL_step >= 300:
                        COORD.request_stop()
                        break

            GLOBAL_step += 1
            print("{} {} {}".format(self.id, ep_r, GLOBAL_step))




if __name__ == '__main__':
   
    father = DPPO(3, [-2, 2])

    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()

    UPDATE_EVENT.clear()    # this one triggers father model to train
    ROLLING_EVENT.set()     # this one triggers workers to collect data

    workers = [Worker(i, father) for i in range(4)]

    GLOBAL_step, GLOBAL_update_count = 0, 0

    COORD = tf.train.Coordinator()
    TRAINING_QUEUE = queue.Queue()      
    threads = []

    for worker in workers:        
        t = threading.Thread(target=worker.work, args=())
        t.start()       
        threads.append(t)
    
    # add father thread into pool, so that update can happen
    threads.append(threading.Thread(target=father.train,))
    threads[-1].start()
    COORD.join(threads)


    # if training is done, see how good it plays
    env = gym.make('Pendulum-v0')
    while True:
        old_observation = env.reset()
        for t in range(50):
            env.render()
            old_observation = env.step(father.get_action(np.reshape(old_observation, [1, 3])))[0]