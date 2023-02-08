import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from config import *

class CriticNetwork(keras.Model):
    """
    Critic network sees all the states and actions and outputs a Q value
    """
    def __init__(self,name):
        super(CriticNetwork, self).__init__()
        self.critic_dense1 = CRITIC_DENSE1
        self.critic_dense2 = CRITIC_DENSE2

        self.model_name = name
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_MADDPG')

        self.fc1 = Dense(self.critic_dense1, activation=CRITIC_ACTIVATION_HIDDEN, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT)
        self.fc2 = Dense(self.critic_dense2, activation=CRITIC_ACTIVATION_HIDDEN, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT)
        self.q = Dense(1, activation=CRITIC_ACTIVATION_OUTPUT, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT)

    def call(self, inputs):
        """
        Centralised Critic takes the all the states from each Agent and corresponding actions from each Agent and gives a Q value
        """
        state, action = inputs
        q_network = self.fc1(tf.concat([state, action], axis=1))
        q_network = self.fc2(q_network)

        q_value = self.q(q_network)

        return q_value


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.actors_dense1 = ACTORS_DENSE1
        self.actors_dense2 = ACTORS_DENSE2
        #This is the number of all possible actions per agent we need to give a value for each. Environment expects it.
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_MADDPG')

        self.fc1 = Dense(self.actors_dense1, activation=ACTORS_ACTIVATION_HIDDEN, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT)
        self.fc2 = Dense(self.actors_dense2, activation=ACTORS_ACTIVATION_HIDDEN, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT)

        #using sigmoid as the environment expects action values between 0 and 1 
        self.mu = Dense(self.n_actions, activation=ACTORS_ACTIVATION_OUTPUT, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT) 

    def call(self, state):
        """
        Actor network takes the state of the agent and outputs continous value for each action
        """
        continuous_action_temp = self.fc1(state)
        continuous_action_temp = self.fc2(continuous_action_temp)
        #output: [0.1, 0.2, 0.3, 0.4, 0.5]
        mu = self.mu(continuous_action_temp)

        return mu