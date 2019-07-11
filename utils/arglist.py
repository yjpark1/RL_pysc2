import numpy as np
import torch
from pysc2.lib import actions

DEVICE = torch.device('cuda:0')

SEED = 1234
FEAT2DSIZE = 64
NUM_ACTIONS = len(actions.FUNCTIONS)
EPS = np.finfo(np.float32).eps.item()

action_shape = {'categorical': (NUM_ACTIONS,),
                'screen1': (1, FEAT2DSIZE, FEAT2DSIZE),
                'screen2': (1, FEAT2DSIZE, FEAT2DSIZE)}

observation_shape = {'minimap': (7, FEAT2DSIZE, FEAT2DSIZE),
                     'screen': (17, FEAT2DSIZE, FEAT2DSIZE),
                     'nonspatial': (NUM_ACTIONS,)}


# DDPG parameters
class DDPG:
    GAMMA = 0.99
    TAU = 0.001
    LEARNINGRATE = 0.001
    BatchSize = 5
    memory_limit = int(1e2)


# PPO parameters
class PPO:
    gamma = 0.99
    lamda = 0.98
    hidden = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    BatchSize = 5
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2
    memory_limit = int(1e2)


# ACER parameters
class ACER:
    GAMMA = 0.99
    TAU = 0.001
    LEARNINGRATE = 0.001
    BatchSize = 5
    memory_limit = int(1e2)
    REPLAY_BUFFER_SIZE = 25
    TRUNCATION_PARAMETER = 10
    DISCOUNT_FACTOR = 0.99
    REPLAY_RATIO = 4
    MAX_EPISODES = 200
    MAX_STEPS_BEFORE_UPDATE = 20
    NUMBER_OF_AGENTS = 12
    OFF_POLICY_MINIBATCH_SIZE = 16
    TRUST_REGION_CONSTRAINT = 1.
    TRUST_REGION_DECAY = 0.99
    ENTROPY_REGULARIZATION = 1e-3
    MAX_REPLAY_SIZE = 200
    ACTOR_LOSS_WEIGHT = 0.1

