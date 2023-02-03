import numpy as np
from maddpg import MADDPG
from utils import plot_learning_curve, save_frames_as_gif
from config import *
import pettingzoo.mpe as mpe


if __name__ == '__main__':
    
    env = mpe.simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    #env = mpe.simple_adversary_v2.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode='rgb_array')
    obs = env.reset()
    agents_env = env.agents

    best_score, avg_score, n_steps = -np.inf,-np.inf, 0
    score_history = np.empty((len(agents_env), N_GAMES))
    actors_shape = []
    n_actions = []
   
    for agent in agents_env:
        actors_shape.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_space(agent).shape[0])

    critic_shape = sum(actors_shape)

    agents = MADDPG(agents_env, n_actions, env.action_space(agents_env[0]).low[0], env.action_space(agents_env[0]).high[0], actors_shape, critic_shape, ALPHA, BETA)

    for i in range(N_GAMES):
        obs = env.reset()
        done = dict(zip(agents_env, [False]*len(agents_env)))
        score = dict(zip(agents_env, [0]*len(agents_env)))
        while not any(done.values()):

            actions = agents.choose_action(obs,agents_env)
            obs_, reward, done, truncated, info = env.step(actions)
            
            #append score for each agent
            for agent in agents_env:
                score[agent] += reward[agent]
            #store transition
            agents.store_transition(obs, actions, obs_, reward, done)

            #update network parameters every 100 steps, except for the frist 1024 steps
            if n_steps % UPDATE_EVERY == 0:
                agents.learn()

            #new obs is now the current
            obs = obs_
            n_steps += 1
            #if truncated, end episode
            if any(truncated.values()):
                done.update((a, True) for a in done)
            

        #end of each game
        for agent_id, agent_title in enumerate(agents_env):
            score_history[agent_id][i] = score[agent_title]
            #score_history.append(score[agent_title])
        score_history[score_history == 0] = np.nan
        avg_score = np.nanmean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agents.save_checkpoint()

        print('episode ', i, 'score ', score, 'avg score %.1f' % avg_score)
    

    x = [i+1 for i in range(N_GAMES)]
    plot_learning_curve(x, score_history[0], FIGURE_FILE)
