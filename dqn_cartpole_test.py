import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import SequentialMemory
from rl.callbacks import *


ENV_NAME = 'LunarLander-v2'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# print("observation space: {0!r}".format(env.observation_space)); quit()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# do = None
do = [0]
if do is not None:
    logger = TrainEpisodeLogger('saves2/')
else:
    logger = TrainEpisodeLogger('saves/')

dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
dqn.fit(env, nb_steps=1000000, visualize=False, callbacks=[logger], log_interval=1000, do=do)

# After training is done, we save the final weights.
if do is None:
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)