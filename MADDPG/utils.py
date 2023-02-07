import numpy as np
import matplotlib.pyplot as plt
from config import *

def plot_learning_curve():

    x = [i+1 for i in range(N_GAMES)]        
    running_avg = np.zeros(len(SCORES_HISTORY))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(SCORES_HISTORY[max(0, i-100):(i+1)])
    fig, ax1 = plt.subplots()
    ax1.plot(x, running_avg, label = 'Mean episode reward', color= 'green')
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Episodes")
    ax1.legend()
    plt.title('Performance of the agents')
    plt.savefig(FIGURE_FILE)
    plt.clf()

    fig, ax2 = plt.subplots()
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS, label='Critic loss Team 1', color='blue')
    ax2.plot(UPDATE_EPISODES, ACTORS_LOSS, label='Actor loss Team 1', color='green')

    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.savefig(FIGURE_FILE2)
    
    file = open("score.txt", 'w+')
    file.write(str(SCORES_HISTORY))
    file.close()
