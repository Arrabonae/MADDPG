import numpy as np
from maddpg import MADDPG
from utils import plot_learning_curve
from config import *
import pettingzoo.mpe as mpe
import time


if __name__ == '__main__':
    
    env = mpe.simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    obs = env.reset()
    agents_env = env.agents
    best_score, n_steps = -np.inf, 0

    actors_shape = []
    n_actions = []
    for agent in agents_env:
        actors_shape.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_space(agent).shape[0])
    critic_shape = sum(actors_shape)
    
    agents = MADDPG(agents_env, n_actions, 
                    env.action_space(agents_env[0]).low[0], 
                    env.action_space(agents_env[0]).high[0], 
                    actors_shape, critic_shape, ALPHA, BETA)

    if LOAD_CHECKPOINT:
        agents.load_models()

    for i in range(N_GAMES):
        obs = env.reset()
        done = dict(zip(agents_env, [False]*len(agents_env)))
        score = dict(zip(agents_env, [0]*len(agents_env)))
        while not any(done.values()):
            if LOAD_CHECKPOINT:
                env.render()
                time.sleep(0.4)

            actions = agents.choose_action(obs,agents_env)
            if actions is not None:
                obs_, reward, done, truncated, info = env.step(actions)
            
                #append score for each agent
                for agent in agents_env:
                    s = (score[agent] + reward[agent])/2
                    score[agent] = s
                #store transition
                if not LOAD_CHECKPOINT:
                    agents.store_transition({key: obs[key] for key in agents_env},
                                                {key: actions[key] for key in agents_env},
                                                {key: obs_[key] for key in agents_env},
                                                {key: reward[key] for key in agents_env},
                                                {key: done[key] for key in agents_env})

                #new obs is now the current
                obs = obs_
                n_steps += 1
                #if truncated, end episode
                if any(truncated.values()):
                    done.update((a, True) for a in done)

            #update network parameters every 100 steps, except for the first 1024 steps
            if n_steps % UPDATE_EVERY == 0 and not LOAD_CHECKPOINT:
                critic_loss, actors_loss = agents.learn()
                CRITIC_LOSS.append(critic_loss)
                ACTORS_LOSS.append(actors_loss)
                UPDATE_EPISODES.append(i)

        #end of each game
        s = score[agents_env[0]] + score[agents_env[1]] + score[agents_env[2]]
        SCORES_HISTORY.append(s)
        AVG_SCORE = np.mean(SCORES_HISTORY[-100:])

        if AVG_SCORE > best_score and not LOAD_CHECKPOINT:
            best_score = AVG_SCORE
            agents.save_checkpoint()

        print('episode ', i, 'avg score MADDPG %.1f ' % AVG_SCORE, 'best score so far %.1f ' %best_score)
        print('episode ', i, 'team episode mean score ', s)

    plot_learning_curve()
