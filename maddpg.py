import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.optimizers import Adam
#locals
from networks import ActorNetwork, CriticNetwork
from config import *
from buffer import ReplayBuffer
from utils import OrnsteinUhlenbeckActionNoise


class MADDPG:
    """
    PettingZoo object structured as: ["agent_0", "agent_1", ...], agent title is required for various methods 
    as the items are shared in dictionaries such as rewards: {"agent_0": 2, "agent_1": 5, ...}

    in Continuous environments, the action space is a Box, which is a tuple of (low, high) values the environment 
    expects the agent to give a value between low and high for each action type. Hence the Actor Network should give us action 
    for each action type

    """
    def __init__(self, agents_env, n_actions, actions_low, actions_high, actors_shape, critic_shape, alpha, beta):
        #List of our Agent objects
        self.agents = []
        self.agents_env = agents_env

        self.n_actions = n_actions
        self.gamma = GAMMA
        self.actor_shape = actors_shape
        self.critic_shape = critic_shape
        self.memory = ReplayBuffer(MEMORY_SIZE, actors_shape, critic_shape, self.agents_env, n_actions, BATCH_SIZE)

        for agent_id, agent_title in enumerate(self.agents_env):
            self.agents.append(ActorAgents(n_actions[agent_id], actions_low, actions_high, agent_title, alpha=alpha))

        self.critic_agent = CriticAgent(beta=beta)

    def store_transition(self, obs, actions, obs_, reward, done):
        self.memory.store_transition(obs, actions, obs_, reward, done)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
        self.critic_agent.save_models()

    def load_checkpoint(self):
        """
        Mo need to load critic agent, as it is only used for learning
        """
        print('... loading checkpoint ...')
        for idx, agent in enumerate(self.agents):
            agent.load_models(self.actor_shape[idx])

    def update_network_parameters(self, tau):
        for agent in self.agents:
            agent.update_network_parameters(tau)
        self.critic_agent.update_network_parameters(tau)

    def choose_action(self, obs, agents):
        """
        Environment takes parallel actions, so we need to return a dictionary of agents in the environment
        return such that {"agent_0": [action_0, action_1 ... action_n], "agent_1": [action_0, action_1 ... action_n], ...}
        """

        actions = {}
        for agent_id, agent_title in enumerate(agents):
            actions[agent_title] = self.agents[agent_id].choose_action(obs[agent_title])

        return actions

    def learn(self):

        if self.memory.mem_cntr < BATCH_SIZE:
            return 0, 0

        critic_obs, critic_obs_, obs_np, obs_np_, rewards, actions_np, dones = self.memory.sample_buffer()

        critic_obs = tf.convert_to_tensor(critic_obs, dtype=tf.float32)
        critic_obs_ = tf.convert_to_tensor(critic_obs_, dtype=tf.float32)
        obs = []
        obs_ = []
        actions = []

        for agent_id, _ in enumerate(self.agents_env):
            obs.append(tf.convert_to_tensor(obs_np[agent_id], dtype=tf.float32))
            obs_.append(tf.convert_to_tensor(obs_np_[agent_id], dtype=tf.float32))
            actions.append(tf.convert_to_tensor(actions_np[agent_id], dtype=tf.float32))

        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        new_pi = []
        pi = []
        target = []
        critic_loss = []

        with tf.GradientTape() as tape:
            for agent_id, _ in enumerate(self.agents_env):
                new_pi.append(self.agents[agent_id].target_actor(obs_[agent_id]))
            old_actions = tf.concat(actions, axis=1)
            new_actions = tf.concat(new_pi, axis=1)
            critic_value_ = tf.squeeze(self.critic_agent.target_critic((critic_obs_, new_actions)), 1)
            critic_value = tf.squeeze(self.critic_agent.critic((critic_obs, old_actions)), 1)

            for agent_id, _ in enumerate(self.agents_env):
                target.append(rewards[:, agent_id] + self.gamma * critic_value_ * (1 - dones[:, agent_id]))
            target = tf.reduce_mean(target, axis=0)

            #OpenAI implementation uses huber loss, but after testing, MSE works better
            critic_loss = keras.losses.MSE(target, critic_value)


        critic_network_gradient = tape.gradient(critic_loss, self.critic_agent.critic.trainable_variables)
        self.critic_agent.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic_agent.critic.trainable_variables))
        
        with tf.GradientTape(persistent= True) as tape2:
            for agent_id, _ in enumerate(self.agents_env):  
                pi.append(self.agents[agent_id].actor(obs[agent_id]))

            pi = tf.concat(pi, axis=1)
            actors_loss = -tf.squeeze(self.critic_agent.critic((critic_obs, pi)),1)
            actors_loss = tf.reduce_mean(actors_loss, axis=0)

        for agent_id, _ in enumerate(self.agents_env):  
            actor_network_gradient = tape2.gradient(actors_loss, self.agents[agent_id].actor.trainable_variables)
            self.agents[agent_id].actor.optimizer.apply_gradients(zip(actor_network_gradient, self.agents[agent_id].actor.trainable_variables))


        del tape2
        self.update_network_parameters(TAU) 

        return critic_loss.numpy(), actors_loss.numpy()


class ActorAgents:
    """
    Class of the actor agent
    """
    def __init__(self, n_actions, actions_low, actions_high, agent_title, alpha=0.001):
        
        self.n_actions = n_actions
        #for action selection: clipping / scaling the action to be between low and high
        self.actions_low = actions_low
        self.actions_high = actions_high

        #per OpenAI paper, Ornstein-Uhlenbeck process for action noise is the best to introduce exploration
        self.noise = OrnsteinUhlenbeckActionNoise(mu= np.zeros(self.n_actions))

        self.actor = ActorNetwork(n_actions=n_actions, name=agent_title + 'actor')
        self.target_actor = ActorNetwork(n_actions=n_actions, name= agent_title+ 'target_actor')

        self.actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        actions = self.actor(state)
        #add noise for exploration
        actions += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        #clip the action to be between low and high otherwise environment will do it for you but 
        #it will affect performance and gives warning message
        actions = tf.clip_by_value(actions, self.actions_low, self.actions_high)
        return actions[0].numpy()

    def save_models(self):
        print('... saving {}  models ...' .format(self.actor.model_name))
        self.actor.save_weights(self.actor.checkpoint_file, save_format='h5')
        print('... saving {}  models ...' .format(self.target_actor.model_name))
        self.target_actor.save_weights(self.target_actor.checkpoint_file, save_format='h5')


    def load_models(self, actor_shape):
        print('... loading {}  models ...'.format(self.actor.model_name))
        self.actor.build((BATCH_SIZE, actor_shape))
        self.actor.load_weights(self.actor.checkpoint_file)
        print('... loading {}  models ...' .format(self.target_actor.model_name))
        self.target_actor.build((BATCH_SIZE, actor_shape))
        self.target_actor.load_weights(self.target_actor.checkpoint_file)


class CriticAgent():
    """
    Class of the critic agent
    """
    def __init__(self, beta):

        self.critic = CriticNetwork(name='Centralised critic')
        self.target_critic = CriticNetwork(name='Centralised target_critic')

        self.critic.compile(optimizer=Adam(learning_rate=beta, clipnorm=1))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta, clipnorm=1))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def save_models(self):
        print('... saving {} models ...'.format(self.critic.model_name))
        self.critic.save_weights(self.critic.checkpoint_file, save_format='h5')
        print('... saving {} models ...'.format(self.target_critic.model_name))
        self.target_critic.save_weights(self.target_critic.checkpoint_file, save_format='h5')

    def load_models(self):
        print('... loading {} models ...'.format(self.critic.model_name))
        self.critic.load_weights(self.critic.checkpoint_file)
        print('... loading {} models ...'.format(self.target_critic.model_name))
        self.update_network_parameters(TAU)

   