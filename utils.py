import numpy as np
import matplotlib.pyplot as plt
from config import *

def plot_learning_curve():
    x = [i+1 for i in range(N_GAMES)]        
    running_avg_good = np.zeros(len(SCORES_HISTORY_GOOD))
    running_avg_adv = np.zeros(len(SCORES_HISTORY_ADV))
    for i in range(len(running_avg_good)):
        running_avg_good[i] = np.mean(SCORES_HISTORY_GOOD[max(0, i-100):(i+1)])
        running_avg_adv[i] = np.mean(SCORES_HISTORY_ADV[max(0, i-100):(i+1)])
    _, ax1 = plt.subplots()
    ax1.plot(x, running_avg_good, label = 'Mean episode reward Good', color= 'green')
    ax1.plot(x, running_avg_adv, label = 'Mean episode reward Adv', color= 'blue')
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Episodes")
    ax1.legend()
    plt.title('Performance of the agents')
    plt.savefig(FIGURE_FILE)
    plt.clf()

    _, ax2 = plt.subplots()
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS_GOOD, label='Critic loss', color='blue')
    ax2.plot(UPDATE_EPISODES, ACTORS_LOSS_GOOD, label='Actor loss', color='green')
    ax2.plot(UPDATE_EPISODES, CRITIC_LOSS_ADV, label='Critic loss Adv', color='red')
    ax2.plot(UPDATE_EPISODES, ACTORS_LOSS_ADV, label='Actor loss Adv', color='yellow')
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Episodes")
    ax2.legend()
    plt.savefig(FIGURE_FILE2)

class OrnsteinUhlenbeckActionNoise():
    """
    OpenAI baselines implementation of Ornstein-Uhlenbeck process
    """
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