"""
Author: Travis Hammond
Version: 12_7_2020
"""

import pytest

from paiutils.reinforcement import *
from paiutils.reinforcement_agents import *


def test_dqn_pg_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    qmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    qmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    amodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    amodel.compile(optimizer='adam', loss='mse')

    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99)
    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99, enable_target=True)
    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99, enable_per=True)
    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99, enable_double=True)

    assert agent.uses_dqn_method == True
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.use_pg()
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, tau=.5)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99, enable_per=True)
    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    x1 = keras.layers.Dense(128)(x)
    x2 = keras.layers.Dense(128)(x)
    outputs = DQNAgent.get_dueling_output_layer(5,
                                                dueling_type='avg')(x1, x2)
    qmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    qmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    agent = DQNPGAgent(GreedyPolicy(), qmodel, amodel, .99, enable_target=True)
    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()
    agent.learn(batch_size=2, epochs=5)


def test_a2c_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    amodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    amodel.compile(optimizer='adam', loss='mse')

    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(1)(x)
    cmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    cmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    agent = A2CAgent(amodel, cmodel, .97)

    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    agent = A2CAgent(amodel, cmodel, .97, lambda_rate=.95)

    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)


def test_ppo_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    amodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    amodel.compile(optimizer='adam', loss='mse')

    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(1)(x)
    cmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    cmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    agent = PPOAgent(amodel, cmodel, .97)

    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    agent = PPOAgent(amodel, cmodel, .97, lambda_rate=.95, clip_ratio=.4)

    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.randint(0, 5),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)


def test_td3_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(1, activation='softmax')(x)
    amodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    amodel.compile(optimizer='adam', loss='mse')

    state_inputs = keras.layers.Input((512,))
    action_inputs = keras.layers.Input((1,))
    x1 = keras.layers.Dense(64)(state_inputs)
    x2 = keras.layers.Dense(64)(action_inputs)
    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(128)(x)
    outputs1 = keras.layers.Dense(1)(x)

    x1 = keras.layers.Dense(64)(state_inputs)
    x2 = keras.layers.Dense(64)(action_inputs)
    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(128)(x)
    outputs2 = keras.layers.Dense(1)(x)

    cmodel = keras.models.Model(inputs=[state_inputs, action_inputs],
                                outputs=[outputs1, outputs2])
    cmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    policy = NoisePolicy(ExponentialDecay(1.0, .001, .1), .001, (-1, 1))
    agent = TD3Agent(policy, amodel, cmodel, .97)

    for _ in range(10):
        assert -1 <= agent.select_action(np.random.random(512),
                                         training=True) <= 1
    for _ in range(10):
        assert -1 <= agent.select_action(np.random.random(512),
                                         training=False) <= 1

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.random((1,)),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.random((1,)),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, tau=.1)
    agent.learn(batch_size=2, epochs=5, policy_noise_std=.3,
                policy_noise_clip=.9, actor_update_infreq=4)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    state_inputs = keras.layers.Input((512,))
    action_inputs = keras.layers.Input((1,))
    x1 = keras.layers.Dense(64)(state_inputs)
    x2 = keras.layers.Dense(64)(action_inputs)
    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(128)(x)
    outputs1 = keras.layers.Dense(1)(x)
    cmodel = keras.models.Model(inputs=[state_inputs, action_inputs],
                                outputs=[outputs1])
    cmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    with pytest.raises(ValueError):
        agent = TD3Agent(policy, amodel, cmodel, .97)


def test_pgc_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    mean = keras.layers.Dense(1, activation='tanh')(x)
    mean = PGCAgent.scale(0, 1, name='scale_mean')(mean)
    std = keras.layers.Dense(1, activation='tanh')(x)
    std = PGCAgent.scale(1e-5, 1, name='scale_std')(std)
    action = PGCAgent.sample(name='sample')([mean, std])
    action = PGCAgent.clip(0, 1)(action)

    amodel = keras.Model(inputs=inputs,
                         outputs=[mean, std, action])
    amodel.compile(optimizer=keras.optimizers.Adam(.001),
                   loss='mse')

    agent = PGCAgent(amodel, .97)

    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True)[0] <= 1
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False)[0] <= 1

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.random((1,)),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.random((1,)),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)


def test_a2cc_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    mean = keras.layers.Dense(2, activation='tanh')(x)
    mean = A2CCAgent.scale(0, 1, name='scale_mean')(mean)
    std = keras.layers.Dense(2, activation='tanh')(x)
    std = A2CCAgent.scale(1e-5, 1, name='scale_std')(std)
    action = A2CCAgent.sample(name='sample')([mean, std])
    action = A2CCAgent.clip(0, 1)(action)
    amodel = keras.Model(inputs=inputs,
                         outputs=[mean, std, action])
    amodel.compile(optimizer='adam', loss='mse')

    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(1)(x)
    cmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    cmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    agent = A2CCAgent(amodel, cmodel, .97)

    for _ in range(10):
        actions = agent.select_action(np.random.random(512),
                                      training=True)
        assert 0 <= actions[0] <= 1
        assert 0 <= actions[1] <= 1
    for _ in range(10):
        actions = agent.select_action(np.random.random(512),
                                      training=False)
        assert 0 <= actions[0] <= 1
        assert 0 <= actions[1] <= 1

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.random(2),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.random(2),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    agent = A2CCAgent(amodel, cmodel, .97, lambda_rate=.95)

    for _ in range(10):
        actions = agent.select_action(np.random.random(512),
                                      training=True)
        assert 0 <= actions[0] <= 1
        assert 0 <= actions[1] <= 1
    for _ in range(10):
        actions = agent.select_action(np.random.random(512),
                                      training=False)
        assert 0 <= actions[0] <= 1
        assert 0 <= actions[1] <= 1

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    for _ in range(10):
        for _ in range(10):
            agent.add_memory(np.random.random(512), np.random.random(2),
                             np.random.random(512), np.random.random(), False)
        agent.add_memory(np.random.random(512), np.random.random(2),
                         np.random.random(512), np.random.random(), True)
        agent.end_episode()

    agent.learn(batch_size=2, epochs=5)
    agent.learn(batch_size=2, epochs=5, entropy_coef=.01)
    agent.learn(batch_size=2, epochs=2, repeat=2)

    path = agent.save('')
    note = agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)
