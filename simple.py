import pettingzoo.mpe as mpe
import os
from utils import *

#set video device dummy
os.environ["SDL_VIDEODRIVER"] = "dummy"


env = mpe.simple_adversary_v2.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode='human')
observation = env.reset()
print(env.observation_space(env.agents[0]))
print(env.action_space(env.agents[0]))
print(env.action_space(env.agents[1]))
print(env.action_space(env.agents[2]))
print(env.action_space(env.agents[1]).shape[0])
print(env.agents, "asd")
print(len(env.agents), "asd")
print(env.action_space(env.agents[0]).low[0])
print(env.action_space(env.agents[0]).high[0])

#for agent in env.agent_iter():
for i in range(10):
    frame = []
    while env.agents:
        #action = env.action_space(agent).sample()
        env.render()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
        #print(actions)
        #print(env.step(actions))
        observation_, reward, terminated, truncated, info = env.step(actions)
        #frame.append(env.render())
        print('action: ', actions)
        print('observation: ', observation_)
        print('reward: ', reward)
        print('terminated: ', terminated)
        print('truncated: ', truncated)
        print('info: ', info)

        if terminated or truncated:
            #save_frames_as_gif(frame, i)
            env.reset()
            print('reset: ', i)
            break
    