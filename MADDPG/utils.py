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
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS, label='Critic loss', color='blue')
    ax2.plot(UPDATE_EPISODES, ACTORS_LOSS, label='Actor loss', color='green')

    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.savefig(FIGURE_FILE2)
    
    file = open("score.txt", 'w+')
    file.write(str(SCORES_HISTORY))
    file.close()


class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu):
        self.theta = THETA
        self.mu = mu
        self.sigma = SIGMA
        self.dt = DT
        self.x0 = X0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)