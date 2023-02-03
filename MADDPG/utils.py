import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
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