import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from bufferDDPG import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from config import *


class DDPG_AGENT:
    def __init__(self, input_dims, n_actions, actions_low, actions_high, agent_title):
        self.gamma = GAMMA
        self.tau = TAU
        self.memory = ReplayBuffer(MEMORY_SIZE, input_dims, n_actions)
        self.batch_size = BATCH_SIZE
        self.n_actions = n_actions
        self.noise = NOISE
        self.actions_high = actions_high
        self.actions_low = actions_low
        self.agent_title = agent_title

        self.actor = ActorNetwork(n_actions=n_actions, name=agent_title + 'actor')
        self.target_actor = ActorNetwork(n_actions=n_actions, name= agent_title+ 'target_actor')
        self.critic = CriticNetwork(name='Centralised critic')
        self.target_critic = CriticNetwork(name='Centralised target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=ALPHA))
        self.target_actor.compile(optimizer=Adam(learning_rate=ALPHA))

        self.critic.compile(optimizer=Adam(learning_rate=BETA))
        self.target_critic.compile(optimizer=Adam(learning_rate=BETA))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def store_transition(self, obs, actions, obs_, reward, done):
        self.memory.store_transition(obs, actions, obs_, reward, done)

    def save_checkpoint(self):
        print('... saving DDPG {} models ...' .format(self.agent_title))
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_checkpoint(self):
        print('... loading DDPG {} models ...' .format(self.agent_title))
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        #min max is 2, -2; network gives -1, 1
        actions = self.actor(state)
        actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.actions_low, self.actions_high)


        return {self.agent_title : actions.numpy()[0]}

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 0, 0

        obs, action, reward, obs_, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(obs, dtype=tf.float32)
        states_ = tf.convert_to_tensor(obs_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)


        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
        return critic_loss.numpy(), actor_loss.numpy()