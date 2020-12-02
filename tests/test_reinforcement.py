"""
Author: Travis Hammond
Version: 12_28_2020
"""

import pytest

from paiutils.reinforcement import *


def test_environment():
    env = Environment((5, 10), (3,))
    assert env.state_shape == (5, 10)
    assert env.action_shape == (3,)
    assert env.discrete_state_space == None
    env.state = 5
    env.reset()
    assert env.state == None
    assert len(env.step(3)) == 3

    agent = Agent((3,), GreedyPolicy())
    agent.set_playing_data()
    env.play_episode(agent, 1, random=True)
    env.play_episode(agent, 1, render=True)
    env.play_episode(agent, 10)

    env.play_episodes(agent, 10, 10)

    with pytest.raises(TypeError):
        env.play_episode([1], 1)
    agent = Agent((3,), GreedyPolicy())
    with pytest.raises(ValueError):
        env.play_episode(agent, 1)

    env.render()
    env.close()

def test_gym_wrapper():
    pass

def test_multiseq_agent_environment():
    pass
 
def test_policy():
    policy = Policy()
    assert policy.select_action(lambda: 5, False) == 5
    policy.reset()
    policy.end_episode()

def test_greedy_policy():
    policy = GreedyPolicy()
    assert policy.select_action(lambda: [5, 1, 2], False) == 0
    policy.reset()
    policy.end_episode()

def test_ascetic_policy():
    policy = AsceticPolicy()
    assert policy.select_action(lambda: [5, 1, 2], False) == 1
    policy.reset()
    policy.end_episode()

def test_stochastic_policy():
    policy = StochasticPolicy(GreedyPolicy(), Decay(.5, .1), 0, (3,))
    assert policy.select_action(lambda: [5, 1, 2], False) == 0
    assert 2 >= policy.select_action(lambda: [5, 1, 2], True) >= 0
    for _ in range(5):
        policy.end_episode()
    assert policy.select_action(lambda: [5, 1, 2], True) == 0
    policy.reset()
    policy.end_episode()

def test_noise_policy():
    pass

def test_uniform_noise_policy():
    pass

def test_temporal_noise_policy():
    pass

def test_decay():
    decay = Decay(1, .1)
    for step in range(10):
        assert decay() == 1 - .1 * step 
    decay.reset()
    assert decay() == 1
    for step in range(10):
        decay.step()
    assert decay() == 0

    decay = Decay(1, .1, .1, step_every_call=False)
    for step in range(10):
        assert decay() == 1
    for step in range(15):
        decay.step()
    assert decay() == .1

    with pytest.raises(ValueError):
        Decay(.5, .1, .6)

def test_exponential_decay():
    pass

def test_linear_decay():
    pass

def test_memory():
    memory = Memory()
    assert len(memory) == 0
    for _ in range(10000):
        memory.add(np.random.random(512))
    assert len(memory) == 10000
    assert memory[0].shape == (512,)
    assert memory[-1].shape == (512,)
    assert memory[9999].shape == (512,)
    memory[100] = np.ones(512)
    assert (memory[100] == np.ones(512)).all()
    memory[-100] = memory[100]
    assert (memory[-100] == np.ones(512)).all()
    assert isinstance(memory.array(), np.ndarray)
    assert memory.array().shape == (10000, 512)
    memory.reset()
    assert len(memory) == 0
    memory.end_episode()

    memory = Memory(max_len=100)
    assert len(memory) == 0
    for _ in range(1000):
        memory.add(np.random.random(512))
    assert len(memory) == 100

    arrays, indexes = Memory.create_shuffled_subset([memory], 50)
    assert arrays[0].shape == (50, 512)
    assert len(indexes) == 50

    arrays, indexes = Memory.create_shuffled_subset(
        [memory], 50, weights=np.ones(len(memory)) / len(memory)
    )
    assert arrays[0].shape == (50, 512)
    assert len(indexes) == 50

    with pytest.raises(ValueError):
        Memory.create_shuffled_subset([memory], 101)

    memory2 = Memory(max_len=50)
    for _ in range(50):
        memory2.add(np.random.random(10))
    with pytest.raises(ValueError):
        Memory.create_shuffled_subset([memory, memory2], 40)

    with pytest.raises(TypeError):
        Memory.create_shuffled_subset([memory, None], 40)


def test_etd_memory():
    memory = ETDMemory(10, np.zeros(512))
    assert len(memory) == 0
    for _ in range(10000):
        memory.add(np.random.random(512))
    assert len(memory) == 10000
    assert memory[0].shape == (512,)
    assert memory[-1].shape == (512,)
    assert memory[9999].shape == (512,)
    memory[100] = np.ones(512)
    assert (memory[100] == np.ones(512)).all()
    memory[-100] = memory[100]
    assert (memory[-100] == np.ones(512)).all()
    assert isinstance(memory.array(), np.ndarray)
    assert memory.array().shape == (10000, 10, 512)
    memory.reset()
    assert len(memory) == 0
    memory.end_episode()

    for _ in range(100):
        for _ in range(100):
            memory.add(np.random.random(512))
        memory.end_episode()
    assert memory.array().shape == (10000, 10, 512)

    arrays, indexes = ETDMemory.create_shuffled_subset([memory], 50)
    assert arrays[0].shape == (50, 10, 512)
    assert len(indexes) == 50

    memory.reset()
    for _ in range(50):
        memory.add(np.random.random(512))

    arrays, indexes = ETDMemory.create_shuffled_subset(
        [memory], 50, weights=np.ones(len(memory)) / len(memory)
    )
    assert arrays[0].shape == (50, 10, 512)
    assert len(indexes) == 50

    arrays, indexes = Memory.create_shuffled_subset([memory], 40)
    assert arrays[0].shape == (40, 512)
    assert len(indexes) == 40

    with pytest.raises(ValueError):
        ETDMemory.create_shuffled_subset([memory], 101)

    memory2 = ETDMemory(10, np.zeros(512))
    for _ in range(50):
        memory2.add(np.random.random(10))
    with pytest.raises(ValueError):
        ETDMemory.create_shuffled_subset([memory, memory2], 40)

    with pytest.raises(TypeError):
        ETDMemory.create_shuffled_subset([memory, None], 40)

    with pytest.raises(NotImplementedError):
        memory = ETDMemory(10, np.zeros(512), max_len=100)
        assert len(memory) == 0
        for _ in range(1000):
            memory.add(np.random.random(512))
        assert len(memory) == 100

def test_ring_memory():
    memory = RingMemory(max_len=100)
    assert len(memory) == 0
    for _ in range(1000):
        memory.add(np.random.random(512))
    assert len(memory) == 100

def test_playingdata():
    PlayingData(True, True, 10, True, {'test': True})
    PlayingData(False, False, 0, False, {})

    with pytest.raises(ValueError):
        PlayingData(1, True, 10, True, {'test': True})

    with pytest.raises(ValueError):
        PlayingData(True, 1, 10, True, {'test': True})

    with pytest.raises(ValueError):
        PlayingData(False, True, -1, True, {'test': True})

    with pytest.raises(ValueError):
        PlayingData(True, False, 10, 1, {'test': True})

    with pytest.raises(TypeError):
        PlayingData(True, True, 10, False, [True])

def test_agent():
    with pytest.raises(TypeError):
        Agent(5, Policy())

    with pytest.raises(TypeError):
        Agent((5,), lambda x: 5)
    
    agent = Agent((5,), GreedyPolicy())
    for _ in range(10):
        assert 0 <= agent.select_action(5, training=True) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    agent.add_memory(None, None, None, None, None)
    agent.forget()
    agent.end_episode()
    agent.learn()
    path = agent.save('', 'hello')
    agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

def test_q_agent():
    pass

def test_pq_agent():
    pass

def test_memory_agent():
    with pytest.raises(TypeError):
        MemoryAgent(5, Policy())

    with pytest.raises(TypeError):
        MemoryAgent((5,), lambda x: 5)
    
    agent = MemoryAgent((5,), GreedyPolicy())
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=True) <= 4
    for _ in range(10):
        assert 0 <= agent.select_action(np.random.random(512),
                                        training=False) <= 4

    agent.set_playing_data()
    assert isinstance(agent.playing_data, PlayingData)

    agent.add_memory(None, None, None, None, None)

    agent.memory['test'] = Memory()
    for _ in range(10):
        agent.memory['test'].add(np.random.random(512))
    assert len(agent.memory['test']) == 10
    agent.end_episode()
    agent.forget()
    assert len(agent.memory['test']) == 0

    agent.learn()
    path = agent.save('', 'hello')
    agent.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)
    assert agent.time_distributed_states is None

    agent.memory['states'] = ETDMemory(10, np.zeros(512))
    for _ in range(10):
        for _ in range(10):
            agent.memory['states'].add(np.random.random(512))
        agent.end_episode()
        assert agent.time_distributed_states.shape == (10, 512)
    assert len(agent.memory['states']) == 100

    agent.learn()
    path = agent.save('', 'hello')
    agent.forget()
    agent.load(path)
    assert agent.memory['states'].array().shape == (100, 10, 512)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)
 
def test_dqn_agent():
    inputs = keras.layers.Input((512,))
    x = keras.layers.Dense(1024)(inputs)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    qmodel = keras.models.Model(inputs=[inputs], outputs=[outputs])
    qmodel.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    agent = DQNAgent(GreedyPolicy(), qmodel, .99)
    agent = DQNAgent(GreedyPolicy(), qmodel, .99, enable_target=True)
    agent = DQNAgent(GreedyPolicy(), qmodel, .99, enable_per=True)
    agent = DQNAgent(GreedyPolicy(), qmodel, .99, enable_double=True)

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

    agent = DQNAgent(GreedyPolicy(), qmodel, .99, enable_per=True)
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

 
def test_pg_agent():
    pass
 
def test_ddpg_agent():
    pass
