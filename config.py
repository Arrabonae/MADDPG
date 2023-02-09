#General
N_GAMES = 60
FIGURE_FILE = 'plots/score.png'
FIGURE_FILE2 = 'plots/critic_actor_loss.png'
CHECKPOINT_DIR = 'checkpoint/'
LOAD_CHECKPOINT = False

#Memory
MEMORY_SIZE = 10**6
BATCH_SIZE = 1024

#Network hyper pars
ALPHA = 0.01
BETA = 0.02
TAU = 0.01
GAMMA = 0.95
UPDATE_EVERY = 100

#Network architecture
WEIGHT_INIT = 'he_normal'
BIAS_INIT = 'he_normal'
CRITIC_DENSE1 = 128
CRITIC_DENSE2 = 128
ACTORS_DENSE1 = 128
ACTORS_DENSE2 = 128

CRITIC_ACTIVATION_HIDDEN = 'leaky_relu'
CRITIC_ACTIVATION_OUTPUT = None
ACTORS_ACTIVATION_HIDDEN = 'leaky_relu'
ACTORS_ACTIVATION_OUTPUT = 'softmax'

#Ornstein-Uhlenbeck noise parameters for exploration
THETA = 0.15
SIGMA = 0.3
DT = 1e-2
X0 = None


#for logging purposes
CRITIC_LOSS_GOOD = []
ACTORS_LOSS_GOOD = []
CRITIC_LOSS_ADV = []
ACTORS_LOSS_ADV = []
UPDATE_EPISODES = []
SCORES_HISTORY_GOOD = []
SCORES_HISTORY_ADV = []
AVG_SCORE_GOOD = 0
AVG_SCORE_ADV = 0