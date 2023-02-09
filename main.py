import numpy as np
from maddpg import MADDPG
from utils import plot_learning_curve
from config import *
import pettingzoo.mpe as mpe
import time


if __name__ == '__main__':
    
    env = mpe.simple_adversary_v2.parallel_env(max_cycles=25, continuous_actions=True)
    obs = env.reset()
    agents_env = env.agents

    adv = agents_env[0]
    good = agents_env[1:]

    best_score_good, best_score_adv, n_steps = -np.inf, -np.inf, 0

    actors_shape = []
    n_actions = []
    for agent in good:
        actors_shape.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_space(agent).shape[0])
    critic_shape = sum(actors_shape)
    
    agents = MADDPG(good, n_actions, 
                    env.action_space(agents_env[0]).low[0], 
                    env.action_space(agents_env[0]).high[0], 
                    actors_shape, critic_shape, ALPHA, BETA)
    adv_agent = MADDPG([adv], [env.action_space(adv).shape[0]], 
                    env.action_space(agents_env[0]).low[0], 
                    env.action_space(agents_env[0]).low[0], 
                    [env.observation_space(adv).shape[0]], env.observation_space(adv).shape[0], ALPHA, BETA)



    if LOAD_CHECKPOINT:
        agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        done = dict(zip(agents_env, [False]*len(agents_env)))
        good_score = 0
        adv_score = 0
        while not any(done.values()):
            if LOAD_CHECKPOINT:
                env.render()
                time.sleep(0.4)
            actions= adv_agent.choose_action(obs,[adv])
            actions.update(agents.choose_action(obs,good))

            if actions is not None:
                obs_, reward, done, truncated, info = env.step(actions)

                #for simplicity the good agents are rewarded with the same reward
                good_score += reward[agents_env[1]]
                adv_score += reward[agents_env[0]]

                #store transition
                if not LOAD_CHECKPOINT:
                    agents.store_transition({key: obs[key] for key in good},
                                                 {key: actions[key] for key in good},
                                                 {key: obs_[key] for key in good},
                                                 {key: reward[key] for key in good},
                                                 {key: done[key] for key in good})
                    adv_agent.store_transition({key: obs[key] for key in [adv]},
                                                 {key: actions[key] for key in [adv]},
                                                 {key: obs_[key] for key in [adv]},
                                                 {key: reward[key] for key in [adv]},
                                                 {key: done[key] for key in [adv]})

                #new obs is now the current obs
                obs = obs_
                n_steps += 1
                #if truncated, end episode
                if any(truncated.values()):
                    done.update((a, True) for a in done)

            #update network parameters every 100 steps, except for the first 1024 steps
            if n_steps % UPDATE_EVERY == 0 and n_steps > BATCH_SIZE and not LOAD_CHECKPOINT:
                critic_loss_good, actors_loss_good = agents.learn()
                critic_loss_adv, actors_loss_adv = adv_agent.learn()
                CRITIC_LOSS_GOOD.append(critic_loss_good)
                ACTORS_LOSS_GOOD.append(actors_loss_good)
                CRITIC_LOSS_ADV.append(critic_loss_adv)
                ACTORS_LOSS_ADV.append(actors_loss_adv)
                UPDATE_EPISODES.append(i)

        #end of each game
        SCORES_HISTORY_GOOD.append(good_score/25)
        SCORES_HISTORY_ADV.append(adv_score/25)
        AVG_SCORE_GOOD = np.mean(SCORES_HISTORY_GOOD[-100:])
        AVG_SCORE_ADV = np.mean(SCORES_HISTORY_ADV[-100:])

        if AVG_SCORE_GOOD > best_score_good and n_steps > BATCH_SIZE:
            best_score_good = AVG_SCORE_GOOD
            if not LOAD_CHECKPOINT:
                agents.save_checkpoint()

        if AVG_SCORE_ADV > best_score_adv:
            best_score_adv = AVG_SCORE_ADV
            if not LOAD_CHECKPOINT:
                adv_agent.save_checkpoint()    
    
        if i % 100 == 0 and not LOAD_CHECKPOINT:
            print('episode ', i, 'avg score Good MADDPG %.1f ' % AVG_SCORE_GOOD, 'best score so far %.1f ' %best_score_good)
            print('episode ', i, 'avg score Adv DDPG %.1f ' % AVG_SCORE_ADV, 'best score so far %.1f ' %best_score_adv)

    plot_learning_curve()
