import numpy as np

class ReplayBuffer:
    """
    Replay buffer for the agents, seperate buffer for different teams to wall the information between teams. 
    """
    def __init__(self, mem_size, actors_shape, critic_shape, agents_env, n_actions, batch_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.n_agents = len(agents_env)
        self.agents_env = agents_env

        self.actors_shape = actors_shape
        self.critic_shape = critic_shape
        self.n_actions = n_actions
        self.batch_size = batch_size

        #Global storage units
        self.reward_memory = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=np.bool)

        #Centralised Critic stroage units
        self.critic_obs_memory = np.zeros((self.mem_size, self.critic_shape))
        self.critic_new_obs_memory = np.zeros((self.mem_size, self.critic_shape))

        #Decentralised Actor storage units
        self.actor_obs_memory = []
        self.actor_new_obs_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_obs_memory.append(np.zeros((self.mem_size, self.actors_shape[i])))
            self.actor_new_obs_memory.append(np.zeros((self.mem_size, self.actors_shape[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, obs, action, obs_, reward, done):
        i = self.mem_cntr % self.mem_size

        critic_obs, critic_obs_ = self.flatten(obs, obs_) 
        
        #Global storage
        for agent_id, agent_title in enumerate(self.agents_env):
            self.reward_memory[i][agent_id] = reward[agent_title]
            self.terminal_memory[i][agent_id] = done[agent_title]

        #Critic storage
        self.critic_obs_memory[i] = critic_obs
        self.critic_new_obs_memory[i] = critic_obs_
        
        #Actor storage
        for agent_id, agent_title in enumerate(self.agents_env):
            self.actor_obs_memory[agent_id][i] = obs[agent_title]
            self.actor_new_obs_memory[agent_id][i] = obs_[agent_title]
            self.actor_action_memory[agent_id][i] = action[agent_title]

        self.mem_cntr += 1

    def flatten(self, obs, obs_):
        critic_obs = []
        critic_obs_ = []
        for agent_title in self.agents_env:
            critic_obs.append(obs[agent_title])
            critic_obs_.append(obs_[agent_title])
        
        critic_obs = np.concatenate(critic_obs, axis=-1)
        critic_obs_ = np.concatenate(critic_obs_, axis=-1)
        return critic_obs, critic_obs_

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        #Global sample
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        #Critic sample
        critic_obs = self.critic_obs_memory[batch]
        critic_obs_ = self.critic_new_obs_memory[batch]

        #Actor sample
        obs = []
        obs_ = []
        actions = []
        for agent_id in range(self.n_agents):
            obs.append(self.actor_obs_memory[agent_id][batch])
            obs_.append(self.actor_new_obs_memory[agent_id][batch])
            actions.append(self.actor_action_memory[agent_id][batch])

        return critic_obs, critic_obs_, obs, obs_, rewards, actions, dones