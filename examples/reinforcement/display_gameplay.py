import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from paiutils import neural_network as nn
from paiutils import reinforcement as rl
from paiutils import reinforcement_agents as ra


# see if using GPU and if so enable memory growth
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

genv = gym.make('MountainCar-v0')
max_steps = genv._max_episode_steps
print(max_steps)
print(genv.observation_space, genv.action_space)

env = rl.GymWrapper(genv)

inputs = keras.layers.Input(shape=env.state_shape)
x = nn.dense(64)(inputs)
x = nn.dense(64)(x)
outputs = nn.dense(env.action_size, activation='softmax',
                   batch_norm=False)(x)
amodel = keras.Model(inputs=inputs,
                     outputs=outputs)
amodel.compile(optimizer=keras.optimizers.Adam(.001),
                loss='mse')
amodel.summary()

inputs = keras.layers.Input(shape=env.state_shape)
x = nn.dense(64)(inputs)
x = nn.dense(64)(x)
outputs = keras.layers.Dense(1)(x)

cmodel = keras.Model(inputs=inputs,
                     outputs=outputs)
cmodel.compile(optimizer=keras.optimizers.Adam(.001),
               loss='mse')
cmodel.summary()

discounted_rate = .99
lambda_rate = 0.95
agent = ra.A2CAgent(
    amodel, cmodel, discounted_rate,
    lambda_rate=lambda_rate,
    create_memory=lambda shape, dtype: rl.Memory(20000)
)


agent.load('trained_a2cagent_mountaincarv0', load_data=False)

agent.set_playing_data(training=False,
                       memorizing=False)
while True:
    step, total_reward = env.play_episode(
        agent, max_steps,
        verbose=True, render=True
    )
    print(total_reward)
