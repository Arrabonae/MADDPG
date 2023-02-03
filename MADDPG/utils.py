import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from config import *

def plot_learning_curve():

    x = [i+1 for i in range(N_GAMES)]
    running_avg = np.zeros(len(SCORES_HISTORY))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(SCORES_HISTORY[max(0, i-100):(i+1)])

    fig, ax1 = plt.subplots()
    ax1.plot(x, running_avg, label = 'running avg', color= 'green')
    ax1.set_ylabel("Score", color='green')
    ax1.set_xlabel("Episodes")
    ax1.legend()
    #plt.plot(x, running_avg, label = 'running avg', color= 'green')
    plt.title('Performance of the agents')
    plt.savefig(FIGURE_FILE)

    ax2 = ax1.twinx() 
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS, label='Critic loss', color='red')
    ax2.plot(UPDATE_EPISODES, ACTORS_LOSS, label='Actor loss', color='blue')
    ax2.set_ylabel("Loss", color='red')
    ax2.legend()
    plt.savefig(FIGURE_FILE2)

def plot_loss_curves(c_loss, a_loss, figure_file):

    ax2 = ax1.twinx() 
    plt.plot(c_loss[1], c_loss[0], label='Critic loss', color='red')
    plt.plot(a_loss[1], a_loss[0], label='Actor loss', color='blue')
    plt.title('Critic and Actor losses')
    plt.legend()
    plt.savefig(figure_file)

def save_frames_as_gif(frames, episode, path = 'gifs/'):
    filename = 'episode_' +str(episode) +'.gif'

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1]/24, frames[0].shape[0]/24 ))#, dpi = 100)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=100)
    anim.save(path + filename, writer='pillow', fps=100)
    plt.close()