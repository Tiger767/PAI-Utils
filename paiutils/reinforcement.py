"""
Author: Travis Hammond
Version: 11_26_2020
"""


import os
from datetime import datetime
from time import sleep
from collections import deque

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import model_from_json


class Environment:
    """This class handles the environment in which the Agent
       performs actions in and can get rewards from.
    """

    def __init__(self, state_shape, action_shape):
        """Initalizes state and action shapes and sets the state.
        params:
            state_shape: A tuple of integers, which is the
                         expected state shape for the agent,
                         or an integer of the discrete state
                         space
            action_shape: A tuple of integers, which is the
                          expected action shape
        """
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.action_shape = action_shape

    def reset(self):
        """Resets the environment to its initialized state.
        return: A numpy ndarray, which is the state
        """
        self.state = None
        return self.state

    def step(self, action):
        """Moves the current state one step forward
           with regard to the action.
        params:
            action: An integer or value that determines an action
        return: A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        self.state = None
        return self.state, 0, False

    def play_episode(self, agent, max_steps,
                     random=False, random_bounds=None,
                     render=False, verbose=True):
        """Plays a single complete episode with the agent.
        params:
            agent: An instance of Agent, which will be used to
                   interact in the environment
            max_steps: An integer, which is the max steps an episode
                       can take before terminating the episode
            random: A booelan, which determines if the agent should not
                    be used, but instead pick random actions
            random_bounds: A tuple of two bounds (lower and upper), which
                           are used for random actions that are not onehots
            render: A boolean, which determines if the environment should
                    be rendered each step
            verbose: A boolean, which determines if information should be
                     printed to the screen
        return: A tuple of an integer (last step) and a float (total reward)
        """
        if not isinstance(agent, Agent):
            raise TypeError('The instance agent is not a child of Agent.')
        if not isinstance(agent.playing_data, PlayingData):
            raise ValueError('Invalid playing_data value. '
                             '(Forgot to set playing_data?)')
        total_reward = 0
        state = self.reset()
        if render:
            self.render()
        for step in range(1, max_steps + 1):
            if random:
                if random_bounds is None:
                    action = np.random.randint(0, self.action_shape[0])
                else:
                    action = np.random.uniform(*random_bounds,
                                               size=self.action_shape)
            else:
                action = agent.select_action(
                    state, training=agent.playing_data.training
                )
            new_state, reward, terminal = self.step(action)

            total_reward += reward

            if agent.playing_data.memorizing:
                agent.add_memory(state, action, new_state, reward, terminal)

            state = new_state

            if verbose:
                print(f'Step: {step} - Reward: {reward} '
                      f'- Action: {action}')
            if render:
                self.render()
            if (agent.playing_data.training
                    and agent.playing_data.learns_in_episode
                    and agent.playing_data.epochs > 0):
                agent.learn(**agent.playing_data.learning_params)
            if terminal:
                break
        agent.end_episode()
        if (agent.playing_data.training
                and not agent.playing_data.learns_in_episode
                and agent.playing_data.epochs > 0):
            agent.learn(**agent.playing_data.learning_params)
        return step, total_reward

    def play_episodes(self, agent, num_episodes, max_steps,
                      random=False, random_bounds=None,
                      render=False, verbose=True,
                      episode_verbose=None):
        """Plays atleast 1 complete episode with the agent.
        params:
            agent: An instance of Agent, which will be used to
                   interact in the environment
            num_episodes: An integer, which is the number of episodes to play
            max_steps: An integer, which is the max steps an episode
                       can take before terminating the episode
            random: A booelan, which determines if the agent should not
                    be used, but instead pick random actions
            random_bounds: A tuple of two bounds (lower and upper), which
                           are used for random actions that are not onehots
            render: A boolean, which determines if the environment should
                    be rendered each step
            verbose: A boolean, which determines if information should be
                     printed to the screen
            episode_verbose: A boolean, which determines if single episode
                             information should be printed to the screen
        return: A float, which is the average total reward of all episodes
        """
        if episode_verbose is None:
            episode_verbose = verbose
        total_rewards = 0
        best_reward = 'Unknown'
        for episode in range(1, num_episodes + 1):
            step, total_reward = self.play_episode(
                agent, max_steps, random=random, random_bounds=random_bounds,
                render=render, verbose=episode_verbose,
            )
            total_rewards += total_reward
            if best_reward == 'Unknown' or total_reward > best_reward:
                best_reward = total_reward
            if verbose:
                str_time = datetime.now().strftime(r'%H:%M:%S')
                print(f'Time: {str_time} - Episode: {episode} - '
                      f'Steps: {step} - '
                      f'Total Reward: {total_reward} - '
                      f'Best Total Reward: {best_reward} - '
                      f'Average Total Reward: {total_rewards / episode}')
        return total_rewards / episode

    def close(self):
        """Closes any threads or loose ends of the environment.
        """
        pass

    def render(self):
        """Renders the environment.
        """
        pass


class GymWrapper(Environment):
    """This class is a environment wrapper for OpenAI Gyms."""

    def __init__(self, gym, state_shape, action_shape):
        """Initalizes state and action shapes and sets the state.
        params:
            gym: A OpenAI Gym
            state_shape: A tuple of integers, which is the
                         expected state shape for the agent,
                         or an integer of the discrete state
                         space
            action_shape: A tuple of integers, which is the
                          expected action shape
        """
        self.gym = gym
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.action_shape = action_shape

    def reset(self):
        """Resets the environment to its initialized state.
        return: A numpy ndarray, which is the state
        """
        state = self.gym.reset()
        if self.discrete_state_space is None:
            return state
        else:
            return [state]

    def step(self, action):
        """Moves the current state one step forward
           with regard to the action.
        params:
            action: An integer or value that determines an action
        return: A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        state, reward, terminal, _ = self.gym.step(action)
        if self.discrete_state_space is None:
            return state, reward, terminal
        else:
            return [state], reward, terminal

    def close(self):
        """Closes any threads or loose ends of the environment.
        """
        self.gym.close()

    def render(self):
        """Renders the environment.
        """
        self.gym.render()


class MultiSeqAgentEnvironment(Environment):
    """This class handles the environment in which multiple agents
       can perform actions against eachother in a sequential manner.
    """

    def __init__(self, state_shape, action_shape):
        """Initalizes state and action shapes and sets the state.
        params:
            state_shape: A tuple of integers, which is the
                         expected state shape for the agent
            action_shape: A tuple of integers, which is the
                          expected action shape
        """
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.action_shape = action_shape

    def reset(self, num_agents):
        """Resets the environment to its initialized state.
        params:
            num_agents: An integer, which is the number of states needed
        return: A numpy ndarray, which is the state
        """
        state = None
        return [state] * num_agents

    def step(self, agent_ndx, action):
        """Moves the current state one step forward
           with regard to the agent's action.
        params:
            agent_ndx: An integer, which is the index of the
                       agent taking a step
            action: An integer or value that determines an action
        return: A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        state = None
        return state, 0, self.terminal

    def play_episode(self, agents, max_steps, shuffle=True, 
                     random=False, random_bounds=None,
                     render=False, verbose=True):
        """Plays a single complete episode with the agents.
        params:
            agents: A list of Agent instances, which will be used to
                    interact in the environment
            max_steps: An integer, which is the max steps an episode
                       can take before terminating the episode
            shuffle: A boolean, which determines if the agents' positions
                     should be shuffled
            random: A booelan, which determines if the agent should not
                    be used, but instead pick random actions
            random_bounds: A tuple of two bounds (lower and upper), which
                           are used for random actions that are not onehots
            render: A boolean, which determines if the environment should
                    be rendered each step
            verbose: A boolean, which determines if information should be
                     printed to the screen
        return: A tuple of a list of integers (last steps)
                and a list of floats (total rewards)
        """
        num_agents = len(agents)
        ndxs = np.arange(num_agents)
        if shuffle:
            np.random.shuffle(ndxs)
        for ndx in ndxs:
            assert isinstance(agents[ndx], Agent), (
                'The instance agent is not a child of Agent.'
            )
            assert isinstance(agents[ndx].playing_data, PlayingData), (
                'Invalid playing_data value.'
            )
        total_rewards = [0] * num_agents
        states = self.reset(num_agents)
        if render:
            self.render()
        break_loop = [False] * num_agents
        for step in range(1, max_steps + 1):
            for ndx in ndxs:
                if random:
                    if random_bounds is None:
                        action = np.random.randint(0, self.action_shape[0])
                    else:
                        action = np.random.uniform(*random_bounds,
                                                   size=self.action_shape)
                else:
                    action = agents[ndx].select_action(
                        states[ndx], training=agents[ndx].playing_data.training
                    )
                new_state, reward, terminal = self.step(ndx, action)
                total_rewards[ndx] += reward

                if agents[ndx].playing_data.memorizing:
                    agents[ndx].add_memory(states[ndx], action, new_state,
                                           reward, terminal)
                states[ndx] = new_state

                if verbose:
                    print(f'Step: {step} - Agent: {ndx} - '
                          f'Reward: {reward} - Action: {action}')
                if render:
                    self.render()
                if (agents[ndx].playing_data.training
                        and agents[ndx].playing_data.learns_in_episode
                        and agents[ndx].playing_data.epochs > 0):
                    agents[ndx].learn(
                        **agents[ndx].playing_data.learning_params
                    )
                if terminal:
                    break_loop[ndx] = True
                    if False not in break_loop:
                        break
            if False not in break_loop:
                break
        for ndx in ndxs:
            agents[ndx].end_episode()
            if (agents[ndx].playing_data.training
                    and not agents[ndx].playing_data.learns_in_episode
                    and agents[ndx].playing_data.epochs > 0):
                agents[ndx].learn(**agents[ndx].playing_data.learning_params)
        return step, total_rewards

    def play_episodes(self, agents, num_episodes, max_steps, shuffle=True,
                      random=False, random_bounds=None, render=False,
                      verbose=True, episode_verbose=None):
        """Plays at least 1 complete episode with the agents.
        params:
            agents: A list of Agent instances, which will be used to
                    interact in the environment
            num_episodes: An integer, which is the number of episodes to play
            max_steps: An integer, which is the max steps an episode
                       can take before terminating the episode
            shuffle: A boolean, which determines if the agents' positions
                     should be shuffled
            random: A booelan, which determines if the agent should not
                    be used, but instead pick random actions
            random_bounds: A tuple of two bounds (lower and upper), which
                           are used for random actions that are not onehots
            render: A boolean, which determines if the environment should
                    be rendered each step
            verbose: A boolean, which determines if information should be
                     printed to the screen
            episode_verbose: A boolean, which determines if single episode
                             information should be printed to the screen
        return: A list of floats, which are the average total reward of all
                episodes for each agent
        """
        if episode_verbose is None:
            episode_verbose = verbose
        num_agents = len(agents)
        total_rewards = [0] * num_agents
        best_rewards = ['Unknown'] * num_agents
        for episode in range(1, num_episodes + 1):
            step, total_reward = self.play_episode(
                agents, max_steps, shuffle=shuffle, random=random,
                random_bounds=random_bounds, render=render,
                verbose=episode_verbose,
            )
            for ndx in range(num_agents):
                total_rewards[ndx] += total_reward[ndx]
                if (best_rewards[ndx] == 'Unknown'
                        or total_reward[ndx] > best_rewards[ndx]):
                    best_rewards[ndx] = total_reward[ndx]
            if verbose:
                str_time = datetime.now().strftime(r'%H:%M:%S')
                print(f'Time: {str_time} - Episode: {episode} - '
                      f'Steps: {step}')
                for ndx in range(num_agents):
                    avg_total_reward = total_rewards[ndx] / episode
                    print(f'Agent {ndx}. '
                          f'Total Reward: {total_reward[ndx]} - '
                          f'Best Total Reward: {best_rewards[ndx]} - '
                          f'Average Total Reward: {avg_total_reward}')
        return [tr / episode for tr in total_rewards]


class Policy:
    """This class is used for calling an Agent's action function."""

    def __init__(self):
        """Initalizes the Policy."""
        pass

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        return action_func()

    def reset(self):
        """Resets any states."""
        pass


class GreedyPolicy(Policy):
    """This class is used for calling an Agent's action function and
       selecting the greediest action.
    """

    def __init__(self):
        super().__init__()

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a list of values
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        action = np.argmax(action_func())
        return action


class AsceticPolicy(Policy):
    """This class is used for calling an Agent's action function and
       selecting the most ascetic action.
    """

    def __init__(self):
        """Initalizes the Policy."""
        super().__init__()

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a list of values
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        action = np.argmin(action_func())
        return action


class StochasticPolicy(Policy):
    def __init__(self, policy, stochasticity_decay_training,
                 stochasticity_testing, action_size):
        """Initalizes the Policy's states.
        params:
            policy: A policy instance
            stochasticity_decay_training: A decay instance which decays
                                          the stochasticity of the policy
            stochasticity_testing: A float, which is the stochasticity
                                   of the policy when the agent is not
                                   training
            action_size: An integer, which is the size of the action ndarray
        """
        super().__init__()
        self.policy = policy
        self.stochasticity_decay_training = stochasticity_decay_training
        self.stochasticity_testing = stochasticity_testing
        self.action_size = action_size

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a list of values
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        if training:
            stochasticity = self.stochasticity_decay_training()
        else:
            stochasticity = self.stochasticity_testing
        if np.random.uniform() < stochasticity:
            return np.random.randint(0, self.action_size)
        else:
            return self.policy.select_action(action_func, training)

    def reset(self):
        """Resets state of the stochasticity decay instance."""
        self.stochasticity_decay_training.reset()


class NoisePolicy(Policy):
    """This class is used for adding normal noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds):
        """Initalizes the Noise Policy.
        params:
            noise_scale_decay_training: A decay instance, which decays
                                        the noise scale (a fraction of
                                        action range) for the policy
            noise_scale_testing: A float, which is the noise scale
                                 of the policy when the agent is not
                                 training
            action_bounds: A tuple of two floats/integers, which are
                           the lower and upper bounds of the action
                           range
        """
        self.noise_scale_decay_training = noise_scale_decay_training
        self.noise_scale_testing = noise_scale_testing
        self.action_bounds = action_bounds

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = action_func()
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        noise = np.random.normal(scale=noise_scale, size=actions.shape)
        return np.clip(actions + noise, *self.action_bounds)

    def reset(self):
        """Resets decay state."""
        self.noise_scale_decay_training.reset()


class UniformNoisePolicy(NoisePolicy):
    """This class is used for adding noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds, additive=False):
        """Initalizes the Uniform Noise Policy.
        params:
            noise_scale_decay_training: A decay instance, which decays
                                        the noise scale (a fraction of
                                        action range) for the policy
            noise_scale_testing: A float, which is the noise scale
                                 of the policy when the agent is not
                                 training
            action_bounds: A tuple of two floats/integers, which are
                           the lower and upper bounds of the action
                           range
            additive: A boolean, which determines if the noise should be
                      added or replace the action value completely
        """
        super().__init__(noise_scale_decay_training,
                         noise_scale_testing, action_bounds)
        self.additive = additive

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = action_func()
        noise = np.random.uniform(*self.action_bounds,
                                  size=actions.shape)
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        if self.additive:
            return np.clip(actions + noise * noise_scale, *self.action_bounds)
        else:
            if np.random.uniform() < noise_scale:
                return noise
            else:
                return actions


class TemporalNoisePolicy(NoisePolicy):
    """This class is used for adding temporal noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds,
                 sigma=.3, theta=.15, dt=.01, init_noise=None):
        """Initalizes the Temporal Noise Policy.
        params:
            noise_scale_decay_training: A decay instance, which decays
                                        the noise scale (a fraction of
                                        action range) for the policy
            noise_scale_testing: A float, which is the noise scale
                                 of the policy when the agent is not
                                 training
            action_bounds: A tuple of two floats/integers, which are
                           the lower and upper bounds of the action
                           range
        """
        super().__init__(noise_scale_decay_training,
                         noise_scale_testing, action_bounds)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.sqrt_dt = np.sqrt(self.dt)
        if init_noise is None:
            self.init_noise = None
        else:
            self.init_noise = init_noise
        self.last_noise = None

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.
        params:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = action_func()
        if self.init_noise is None:
            self.init_noise = np.full(actions.shape,
                                      np.mean(self.action_bounds))
            self.last_noise = self.init_noise
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        noise = np.random.normal(scale=noise_scale, size=actions.shape)
        noise = (self.last_noise +
                 self.theta * -self.last_noise * self.dt +
                 self.sigma * self.sqrt_dt * noise)
        self.last_noise = noise
        return np.clip(actions + noise, *self.action_bounds)

    def reset(self):
        """Resets decay state and initial actions."""
        super().reset()
        self.last_noise = self.init_noise


class Decay:
    """This class decays a initial value to a minimum
       value through a given number of steps.
       (formula: max(initial_value - constant * steps, 0))
    """

    def __init__(self, initial_value, constant, min_value=0):
        """Initalizes the state of the decay object.
        params:
            initial_value: A float, which is the starting value to decay
            constant: A float, which is the slope/rate that the decay occurs
            min_value: A float, which is the minimum value the decay can reach
        """
        assert initial_value >= min_value, (
            'initial_value must be greater or equal to min_value'
        )
        self.initial_value = initial_value
        self.steps = 0
        self.constant = constant
        self.min_value = min_value

    def reset(self):
        """Resets the steps"""
        self.steps = 0

    def __call__(self):
        """Returns the current value with regard to the state of decay.
        return: A float
        """
        self.steps += 1
        value = self.initial_value - self.constant * self.steps
        return np.max([value, self.min_value])


class ExponentialDecay(Decay):
    """This class decays a initial value to a minimum
       value exponentially through a given number of steps.
       (formula: inital_value * (1 - rate)^steps + min_value)
    """

    def __init__(self, initial_value, rate, min_value=0):
        """Initalizes the state of the decay object.
        params:
            initial_value: A float, which is the starting value to decay
            rate: A float, which is the slope/rate that the decay occurs
            min_value: A float, which is the minimum value the decay can reach
        """
        assert initial_value >= min_value, (
            'initial_value must be greater or equal to min_value'
        )
        self.initial_value = initial_value
        self.rate = rate
        self.min_value = min_value
        self.steps = 0

    def __call__(self):
        """Returns the current value with regard to the state of decay.
        return: A float
        """
        self.steps += 1
        return np.maximum(self.initial_value * (1 - self.rate)**self.steps,
                          self.min_value)


class LinearDecay(Decay):
    """This class decays a initial value to a minimum
       value linearly through a given number of steps.
       (formula: max(initial_value - (inital_value - min_value)
                     / total_steps * steps, min_value))
    """

    def __init__(self, initial_value, total_steps, min_value=0):
        """Initalizes the state of the decay object.
        params:
            initial_value: A float, which is the starting value to decay
            total_steps: An integer, which is the number of steps until
                         min_value would be reach
            min_value: A float, which is the minimum value the decay
                       can reach
        """
        assert initial_value >= min_value, (
            'initial_value must be greater or equal to min_value'
        )
        self.initial_value = initial_value
        self.total_steps = total_steps
        self.min_value = min_value
        self.a = (-1 * (self.initial_value - self.min_value) /
                  self.total_steps)
        self.steps = 0

    def __call__(self):
        """Returns the current value with regard to the state of decay.
        return: A float
        """
        self.steps += 1
        value = self.a * self.steps + self.initial_value
        return np.max([value, self.min_value])


class Memory:
    """This class is used by agents to store episode information.
       (uses a normal python list)
    """

    def __init__(self, max_len=None):
        """Initalizes the memory.
        params:
            max_len: An integer, which is the max length of memory
                     (if reached, the oldest memory will be removed)
        """
        self.max_len = max_len
        self.buffer = []

    def __len__(self):
        """Returns the number of entries in the memory.
        return: An integer
        """
        return len(self.buffer)

    def add(self, x):
        """Adds a entry to memory.
        params:
            x: A entry similar to other entries
        """
        self.buffer.append(x)
        if (self.max_len is not None
                and len(self.buffer) > self.max_len):
            del self.buffer[0]

    def __getitem__(self, key):
        """Returns an item given a key.
        params:
            key: A valid key or index for a memory entry
        """
        return self.buffer[key]

    def __setitem__(self, key, value):
        """Sets a entry to a given key.
        params:
            key: A valid key or index for a memory entry
            value: A entry similar to other entries
        """
        self.buffer[key] = value

    def array(self):
        """Returns a copy of the memory.
        return: A numpy ndarray
        """
        return np.array(self.buffer)

    def reset(self):
        """Resets or clears the memory.
        """
        self.buffer.clear()

    @staticmethod
    def create_shuffled_subset(memories, subset_size, weights=None):
        """Creates a list of numpy arrays of a shuffled subset of memories.
        params:
            memories: A list of Memeory Objects (not asserted but assumed)
            subset_size: A integer, which is the size of the
                         outer dimension of each ndarray
            weights: A list of probabilities that add up to 1
        return: arrays and shuffled indexes
        """
        indexes = np.random.choice(np.arange(len(memories[0])),
                                   size=subset_size, replace=False,
                                   p=weights)
        if len(memories[0]) // 10 > subset_size:
            arrays = [memory.array()[indexes] for memory in memories]
        else:
            arrays = [np.empty(memory[0].shape) for memory in memories]
            for ndx, rndx in enumerate(indexes):
                for array, memory in zip(arrays, memories):
                    array[ndx] = memory[rndx]
        return arrays, indexes


class BufferedMemory(Memory):
    """This class is used by agents to store episode information.
       (uses a numpy ndarray)
    """

    def __init__(self, length, shape, dtype=np.float):
        """Initalizes the memory.
        params:
            length: An integer, which is the max length of memory
                    and the amount buffered
                    (oldest elements will be overwritten at max)
            shape: A tuple, which is the shape of each element
            dtype: A numpy data type, which is the data type of each element
        """
        self.length = length
        self.buffer = np.empty((length, *shape), dtype=dtype)
        self.start = 0
        self.end = 0
        self.overflow = False
        self.filled = False

    def __len__(self):
        """Returns the number of entries in the memory.
        return: An integer
        """
        if self.filled:
            return self.length
        return self.end - self.start

    def add(self, x):
        """Adds a entry to memory.
        params:
            x: A entry similar to other entries
        """
        self.buffer[self.end] = x
        self.end += 1
        if self.end == self.length:
            self.filled = True
            self.overflow = True
            self.end = 0
            self.start = -1
        if self.overflow:
            self.start += 1
            if self.start == self.length:
                self.start = 0
                self.overflow = False

    def __getitem__(self, key):
        """Returns an item given a key.
        params:
            key: A valid key or index for a memory entry
        """
        if key >= 0:
            if key >= self.__len__():
                raise IndexError('out of range')
            key = (self.start + key) % self.length
        else:
            if -key > self.__len__():
                raise IndexError('out of range')
            key = (self.end + key) % self.length
        return self.buffer[key]

    def __setitem__(self, key, value):
        """Sets a entry to a given key.
        params:
            key: A valid key or index for a memory entry
            value: A entry similar to other entries
        """
        if key >= 0:
            if key >= self.__len__():
                raise IndexError('out of range')
            key = (self.start + key) % self.length
        else:
            if -key > self.__len__():
                raise IndexError('out of range')
            key = (self.end + key) % self.length
        self.buffer[key] = value

    def array(self):
        """Returns a copy of the memory.
        return: A numpy ndarray
        """
        if self.filled:
            return self.buffer
        return self.buffer[self.start:self.end]

    def reset(self):
        """Resets or clears the memory.
        """
        self.start = 0
        self.end = 0
        self.overflow = False
        self.filled = False

    @staticmethod
    def create_shuffled_subset(memories, subset_size, weights=None):
        """Creates a list of numpy arrays of a shuffled subset of memories.
        params:
            memories: A list of BufferedMemeory Objects
                      (not asserted but assumed)
            subset_size: A integer, which is the size of the
                         outer dimension of each ndarray
            weights: A list of probabilities that add up to 1
        return: arrays and shuffled indexes
        """
        if weights is not None:
            weights = np.roll(weights, memories[0].start)
        indexes = np.random.choice(np.arange(len(memories[0])),
                                   size=subset_size, replace=False,
                                   p=weights)
        return [memory.buffer[indexes] for memory in memories], indexes


class RingMemory(Memory):
    """This class is used by agents to store episode information.
       (uses a deque)
    """

    def __init__(self, max_len):
        """Initalizes the memory.
        params:
            max_len: An integer, which is the max length of memory
                     (if reached, the oldest memory will be removed)
        """
        self.buffer = deque(maxlen=max_len)

    def add(self, x):
        """Adds a entry to memory.
        params:
            x: A entry similar to other entries
        """
        self.buffer.append(x)


class PlayingData:
    """This class is used for containing data
       that the environment needs to know, but the agent has.
    """

    def __init__(self, training, memorizing, epochs,
                 learns_in_episode, learning_params):
        """Initalizes the data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            memorizing: A boolean, which determines if the agent
                        should be adding the information obtained
                        through playing an episode to memory
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            learns_in_episode: A boolean, which determines if the agent
                               learns during a episode or at the end
            learning_params: A dictionary of parameters for the agent's
                             learn method
        """
        assert training is True or training is False, (
            'Invalid training value. Must be True or False.'
        )
        assert memorizing is True or memorizing is False, (
            'Invalid memorizing value. Must be True or False.'
        )
        assert epochs >= 0, (
            'Invalid epoch value. Must be greater or equal to zero.'
        )
        assert learns_in_episode is True or learns_in_episode is False, (
            'Invalid learns_in_episode value. Must be True or False.'
        )
        assert isinstance(learning_params, dict), (
            'Invalid learning_params value. Must be a dictionary.'
        )
        self.training = training
        self.memorizing = memorizing
        self.epochs = epochs
        self.learns_in_episode = learns_in_episode
        self.learning_params = learning_params


class Agent:
    """This class is the base class for all agent classes,
       and essentially is a random agent.
    """

    def __init__(self, action_shape, policy):
        """Initalizes the agent.
        params:
            action_shape: A tuple of integers, which is the
                          action shape of the environment
            policy: A policy instance
        """
        self.action_shape = action_shape
        self.policy = policy
        self.playing_data = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value or list of values, which is the
                   state to get the action for
            training: A boolean, which determines if the
                      agent is training
        return: A value, which is the selected action
        """
        def _select_action():
            return np.random.random(self.action_shape)
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self):
        """Sets the episode data."""
        self.playing_data = PlayingData(False, False, 0, False, {})

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
           (For this agent all memory is discarded)
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        pass

    def forget(self):
        """Forgets or clears all memory."""
        pass

    def end_episode(self):
        """Ends the episode for the agent."""
        pass

    def learn(self, verbose=True):
        """Trains the agent on a batch of its experiences.
           (For this agent no learning is needed)
        params:
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        pass

    def load(self, path):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
        """
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            print(file.read(), end='')

    def save(self, path, note):
        """Saves a note to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder
        return: A string, which is the complete path of the save
        """
        time = datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            file.write(note)
        return path


def encrypt(string):
    """Encrpts a string with a XOR cipher.
       (assumes security is not necessarily required)
    params:
        string: A string to encrypt
    return: A bytes object
    """
    key = 175
    string = map(ord, string)
    result = []
    for plain in string:
        key ^= plain
        result.append(key)
    return bytes(result)


class QAgent(Agent):
    """This class is a Q-learning Agent. It does not uses a neural network,
       but instead uses a table.
    """

    def __init__(self, discrete_state_space, action_size,
                 policy, discounted_rate):
        """Initalizes the Q-learning agent.
        params:
            discrete_state_space: An integer, which
            action_size: An integers, which is the
                         action size of the environment
            policy: A policy instance
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
        """
        Agent.__init__(self, action_size, policy)
        self.discrete_state_space = discrete_state_space
        self.discounted_rate = discounted_rate
        self.qtable = np.zeros((self.discrete_state_space,
                                self.action_shape))
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.terminal = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value or list of values, which is the
                   state to look up the action for in the table
            training: A boolean, which determines if the
                      agent is training
        return: A value, which is the selected action
        """
        def _select_action():
            nonlocal state
            if isinstance(state, list) and len(state) == 1:
                state = state[0]
            return self.qtable[state]
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self, training=False, learning_rate=None,
                         verbose=False):
        """Sets the playing data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            learning_rate: A float, which is the rate that the table
                           is updated with the currect Q reward
                           (Must be provided if training is True)
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        self.playing_data = PlayingData(training, training, 1, True,
                                        {'learning_rate': learning_rate,
                                         'verbose': verbose})

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        if isinstance(state, list) and len(state) == 1:
            state = state[0]
        if isinstance(new_state, list) and len(new_state) == 1:
            new_state = new_state[0]
        self.state = state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.terminal = terminal

    def forget(self):
        """Forgets or clears all memory."""
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.terminal = None

    def learn(self, learning_rate, verbose=True):
        """Trains the agent on its last experience.
        params:
            learning_rate: A float, which is the rate that the table
                           is updated with the currect Q reward
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        assert self.state is not None, (
            'Memory is empty.'
        )
        discounted_reward = 0
        if not self.terminal:
            discounted_reward = (self.discounted_rate *
                                 np.amax(self.qtable[self.new_state]))
        self.qtable[self.state, self.action] = (
            (1 - learning_rate) * self.qtable[self.state, self.action] +
            learning_rate * (self.reward + discounted_reward)
        )
        if verbose:
            pass

    def load(self, path):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
        """
        Agent.load(self, path)
        self.qtable = np.load(os.path.join(path, 'qtable.npy'))

    def save(self, path, note='QAgent Save'):
        """Saves a note and qtable to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder
        return: A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        np.save(os.path.join(path, 'qtable.npy'), self.qtable)
        return path


class PQAgent(QAgent):
    """This class is like QAgent, but it uses multiple variables at once,
       hince Parallel Q Agent.
    """

    def __init__(self, discrete_state_space, action_size,
                 policy, discounted_rates, learning_rates):
        """Initalizes the Q-learning agent.
        params:
            discrete_state_space: An integer, which
            action_size: An integers, which is the
                         action size of the environment
            policy: A policy instance
            discounted_rates: A list of floats within 0.0-1.0, which
                              are the rates that future rewards should
                              be counted for the current reward
            learning_rate: A list of floats, which are the rates
                           that the table is updated with the currect
                           Q reward
        """
        Agent.__init__(self, action_size, policy)
        self.discrete_state_space = discrete_state_space
        self.discounted_rates = np.array(discounted_rates)
        self.learning_rates = np.array(learning_rates)
        self.inv_learning_rates = 1 - self.learning_rates
        self.qtables = np.zeros((len(self.learning_rates),
                                 len(self.discounted_rates),
                                 self.discrete_state_space,
                                 self.action_shape))
        self.selected_qtable = None
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.terminal = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value or list of values, which is the
                   state to look up the action for in the table
            training: A boolean, which determines if the
                      agent is training
        return: A value, which is the selected action
        """
        def _select_action():
            nonlocal state
            if isinstance(state, list) and len(state) == 1:
                state = state[0]
            return self.selected_qtable()[state]
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self, training=False, learning_rate_ndx=None,
                         discounted_rate_ndx=None, verbose=False):
        """Sets the playing data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            learning_rate_ndx: An integer, which is a ndx for
                               the learning rates
            discounted_rate_ndx: An integer, which is a ndx for
                                 the discounted rate
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        lrn = 0
        drn = 0
        if learning_rate_ndx is not None:
            self.learning_rates[learning_rate_ndx]
            lrn = learning_rate_ndx
        if discounted_rate_ndx is not None:
            self.discounted_rates[discounted_rate_ndx]
            drn = discounted_rate_ndx
        self.selected_qtable = lambda: self.qtables[lrn, drn]
        self.playing_data = PlayingData(
            training, training, 1, True,
            {'learning_rate_ndx': learning_rate_ndx,
             'discounted_rate_ndx': discounted_rate_ndx,
             'verbose': verbose}
        )

    def learn(self, learning_rate_ndx=None,
              discounted_rate_ndx=None, verbose=True):
        """Trains the agent on its last experience.
        params:
            learning_rate_ndx: An integer, which is a ndx for
                               the learning rates
            discounted_rate_ndx: An integer, which is a ndx for
                                 the discounted rate
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        assert self.state is not None, (
            'Memory is empty.'
        )
        lrn = learning_rate_ndx
        drn = discounted_rate_ndx
        if lrn is None and drn is None:
            discounted_reward = 0
            if not self.terminal:
                discounted_reward = (
                    self.discounted_rates *
                    np.amax(self.qtables[:, :, self.new_state], axis=-1)
                )
            self.qtables[:, :, self.state, self.action] = (
                self.inv_learning_rates[:, np.newaxis] *
                self.qtables[:, :, self.state, self.action] +
                self.learning_rates[:, np.newaxis] *
                (self.reward + discounted_reward)
            )
        elif drn is None:
            discounted_reward = 0
            if not self.terminal:
                discounted_reward = (
                    self.discounted_rates *
                    np.amax(self.qtables[lrn, :, self.new_state], axis=-1)
                )
            self.qtables[lrn, :, self.state, self.action] = (
                self.inv_learning_rates[lrn, np.newaxis] *
                self.qtables[lrn, :, self.state, self.action] +
                self.learning_rates[lrn, np.newaxis] *
                (self.reward + discounted_reward)
            )
        elif lrn is None:
            discounted_reward = 0
            if not self.terminal:
                discounted_reward = (
                    self.discounted_rates[drn] *
                    np.amax(self.qtables[:, drn, self.new_state], axis=-1)
                )
            self.qtables[:, drn, self.state, self.action] = (
                self.inv_learning_rates[:, np.newaxis] *
                self.qtables[:, drn, self.state, self.action] +
                self.learning_rates[:, np.newaxis] *
                (self.reward + discounted_reward)
            )
        else:
            discounted_reward = 0
            if not self.terminal:
                discounted_reward = (
                    self.discounted_rates[drn] *
                    np.amax(self.qtables[lrn, drn, self.new_state], axis=-1)
                )
            self.qtables[lrn, drn, self.state, self.action] = (
                self.inv_learning_rates[lrn] *
                self.qtables[lrn, drn, self.state, self.action] +
                self.learning_rates[lrn] *
                (self.reward + discounted_reward)
            )

        if verbose:
            pass

    def load(self, path):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
                  from
        """
        Agent.load(self, path)
        self.qtables = np.load(os.path.join(path, 'qtables.npy'))

    def save(self, path, note='PQAgent Save'):
        """Saves a note and qtables to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder
        return: A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        np.save(os.path.join(path, 'qtables.npy'), self.qtables)
        return path


class DQNAgent(Agent):
    """This class is an Agent that uses a Deep Q Network instead of
       a table like the QAgent. This allows for generalizations and
       large environment states.
    """

    @staticmethod
    def get_dueling_output_layer(action_shape, dueling_type='avg'):
        assert dueling_type in ['avg', 'max', 'naive'], (
            "Dueling type must be 'avg', 'max', or 'naive'"
        )

        def layer(x1, x2):
            x1 = keras.layers.Dense(1)(x1)
            x2 = keras.layers.Dense(action_shape[0])(x2)
            x = keras.layers.Concatenate()([x1, x2])
            if dueling_type == 'avg':
                def dueling(a):
                    return (K.expand_dims(a[:, 0], -1) + a[:, 1:] -
                            K.mean(a[:, 1:], axis=1, keepdims=True))
            elif dueling_type == 'max':
                def dueling(a):
                    return (K.expand_dims(a[:, 0], -1) + a[:, 1:] -
                            K.max(a[:, 1:], axis=1, keepdims=True))
            else:
                def dueling(a):
                    return K.expand_dims(a[:, 0], -1) + a[:, 1:]
            return keras.layers.Lambda(dueling, output_shape=action_shape,
                                       name='q_output')(x)
        return layer

    def __init__(self, policy, qmodel, discounted_rate,
                 create_memory=lambda: Memory(),
                 enable_target=True, enable_double=False,
                 enable_PER=False):
        """Initalizes the Deep Q Network Agent.
        params:
            policy: A policy instance
            qmodel: A keras model, which takes the state as input and outputs
                    Q Values
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
            enable_target: A boolean, which determines if a target model
                           should be used
            enable_double: A boolean, which determiens if the Double Deep Q
                           Network should be used
            enable_PER: A boolean, which determines if prioritized experience
                        replay should be used (The implementation for this is
                        not the normal tree implementation, and only weights
                        the probabilily of being choosen and not also the
                        gradient)
        """
        Agent.__init__(self, qmodel.output_shape[1:], policy)
        self.qmodel = qmodel
        self.target_qmodel = None
        self.enable_target = enable_target or enable_double
        self.enable_double = enable_double
        if self.enable_target:
            self.target_qmodel = keras.models.clone_model(qmodel)
            self.target_qmodel.compile(optimizer='sgd', loss='mse')
        else:
            self.target_qmodel = self.qmodel
        self.discounted_rate = discounted_rate
        self.states = create_memory()
        self.next_states = create_memory()
        self.actions = create_memory()
        self.rewards = create_memory()
        self.terminals = create_memory()
        if enable_PER:
            self.per_losses = create_memory()

            # assuming the true max loss will be less than 100
            # at least at the begining
            self.max_loss = 100.0
        else:
            self.per_losses = None
        self.action_identity = np.identity(self.action_shape[0])
        self.total_steps = 0
        self.metric = tf.keras.metrics.Mean(name='loss')

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.qmodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=self.qmodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value, which is the state to predict
                   the Q values for
            training: A boolean, which determines if the
                      agent is training
        return: A value, which is the selected action
        """
        def _select_action():
            qvalues = self.qmodel(np.expand_dims(state, axis=0),
                                  training=False)[0].numpy()
            return qvalues
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self, training=False, memorizing=False,
                         learns_in_episode=False, batch_size=None,
                         mini_batch=0, epochs=1, repeat=1,
                         target_update_interval=1, tau=1.0, verbose=True):
        """Sets the playing data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            memorizing: A boolean, which determines if the agent
                        should be adding the information obtained
                        through playing an episode to memory
            learns_in_episode: A boolean, which determines if the agent
                               learns during a episode or at the end
            batch_size: An integer, which is the size of each batch
                        within the mini-batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled and qvalues repredicted)
            target_update_interval: An integer, which is the number of
                                    complete training instances
                                    (repeats do not count) until the
                                    target model weights are updated
            tau: A float, which is the strength of the copy from the
                 qmodel to the target qmodel (1.0 is a hard copy and
                 less is softer)
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'target_update_interval': target_update_interval,
                           'tau': tau,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        learns_in_episode, learning_params)

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        self.states.add(np.array(state))
        self.next_states.add(np.array(new_state))
        self.actions.add(self.action_identity[action])
        self.rewards.add(reward)
        self.terminals.add(0 if terminal else 1)
        if self.per_losses is not None:
            self.per_losses.add(self.max_loss)

    def forget(self):
        """Forgets or clears all memory."""
        self.states.reset()
        self.next_states.reset()
        self.actions.reset()
        self.rewards.reset()
        self.terminals.reset()
        if self.per_losses is not None:
            self.per_losses.reset()

    def update_target(self, tau):
        """Updates the target Q Model weights.
        params:
            tau: A float, which is the strength of the copy from the
                 qmodel to the target qmodel (1.0 is a hard copy and
                 less is softer)
        """
        if tau == 1.0:
            self.target_qmodel.set_weights(self.qmodel.get_weights())
        else:
            tws = self.target_qmodel.trainable_variables
            ws = self.qmodel.trainable_variables
            for ndx in range(len(tws)):
                tws[ndx] = ws[ndx] * tau + tws[ndx] * (1 - tau)

    def _train_step(self, states, next_states,
                    actions, terminals, rewards):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            next_states: A tensor that contains the states of
                         the environment after an action was performed
            actions: A tensor that contains onehot encodings of
                     the action performed
            terminals: A tensor that contains ones for nonterminal
                       states and zeros for terminal states
            rewards: A tensor that contains the reward for the action
                     performed in the environment
        return: A loss for this batch of data
        """
        if self.enable_double:
            qvalues = self.qmodel(next_states, training=False)
            actions = tf.argmax(qvalues, axis=-1)
            qvalues = self.target_qmodel(next_states, training=False)
            qvalues = tf.squeeze(tf.gather(qvalues, actions[:, tf.newaxis],
                                           axis=-1, batch_dims=1))
        else:
            qvalues = self.target_qmodel(next_states, training=False)
            qvalues = tf.reduce_max(qvalues, axis=-1)
        qvalues = (rewards +
                   self.discounted_rate * qvalues * terminals)
        with tf.GradientTape() as tape:
            y_pred = self.qmodel(states, training=True)
            if len(self.qmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.qmodel.losses)
            else:
                reg_loss = 0
            y_true = (y_pred * (1 - actions) +
                      qvalues[:, tf.newaxis] * actions)
            loss = self.qmodel.compiled_loss._losses.fn(
                y_true, y_pred
            ) + reg_loss
        grads = tape.gradient(loss, self.qmodel.trainable_variables)
        self.qmodel.optimizer.apply_gradients(
            zip(grads, self.qmodel.trainable_variables)
        )
        self.metric(loss)

        return tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

    def _train(self, states, next_states, actions, terminals,
               rewards, epochs, batch_size, verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            next_states: A numpy array that contains the states of
                         the environment after an action was performed
            actions: A numpy array that contains onehot encodings of
                     the action performed
            terminals: A numpy array that contains ones for nonterminal
                       states and zeros for terminal states
            rewards: A numpy array that contains the reward for the action
                     performed in the environment
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            verbose: A boolean, which determines if information should
                     be printed to the screen
        return: A list of floats, which are the absolute losses for all
                the data
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             next_states.astype(float_type),
             actions.astype(float_type),
             terminals.astype(float_type),
             rewards.astype(float_type))
        ).batch(batch_size)
        losses = []
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')

            count = 0
            for batch in batches:
                if epoch == epochs:
                    losses.append(self._tf_train_step(*batch).numpy())
                else:
                    self._tf_train_step(*batch)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            loss_results = self.metric.result()
            self.metric.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {loss_results}')
        return np.hstack(losses)

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1,
              target_update_interval=1, tau=1.0, verbose=True):
        """Trains the agent on a sample of its experiences.
        params:
            batch_size: An integer, which is the size of each batch
                        within the mini-batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled and qvalues repredicted)
            target_update_interval: An integer, which is the number of
                                    complete training instances
                                    (repeats do not count) until the
                                    target model weights are updated
            tau: A float, which is the strength of the copy from the
                 qmodel to the target qmodel (1.0 is a hard copy and
                 less is softer)
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        self.total_steps += 1
        if mini_batch > 0 and len(self.states) > mini_batch:
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat+1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            if self.per_losses is None:
                arrays, indexes = self.states.create_shuffled_subset(
                    [self.states, self.next_states, self.actions,
                     self.terminals, self.rewards],
                    length
                )
            else:
                per_losses = self.per_losses.array()
                self.max_loss = per_losses.max()
                per_losses = per_losses / per_losses.sum()
                arrays, indexes = self.states.create_shuffled_subset(
                    [self.states, self.next_states, self.actions,
                     self.terminals, self.rewards],
                    length, weights=per_losses
                )
            losses = self._train(*arrays, epochs, batch_size, verbose=verbose)
            if self.per_losses is not None:
                for ndx, loss in zip(indexes, losses):
                    self.per_losses[ndx] = loss

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

    def load(self, path, load_model=True, load_data=True):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architecture and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded
        """
        Agent.load(self, path)
        if load_model:
            with open(os.path.join(path, 'qmodel.json'), 'r') as file:
                self.qmodel = model_from_json(file.read())
            self.qmodel.load_weights(os.path.join(path, 'qweights.h5'))
            if self.enable_target:
                self.target_qmodel = keras.models.clone_model(self.qmodel)
                self.target_qmodel.compile(optimizer='sgd', loss='mse')
            else:
                self.target_qmodel = self.qmodel

        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for state in file['states']:
                    self.states.add(state)
                for new_state in file['next_states']:
                    self.next_states.add(new_state)
                for action in file['actions']:
                    self.actions.add(action)
                for reward in file['rewards']:
                    self.rewards.add(reward)
                for terminal in file['terminals']:
                    self.terminals.add(terminal)
                if self.per_losses is not None:
                    if 'per_losses' in file:
                        for loss in file['per_losses']:
                            self.per_losses.add(loss)
                    else:
                        for _ in range(len(self.states)):
                            self.per_losses.add(self.max_loss)

    def save(self, path, save_model=True,
             save_data=True, note='DQNAgent Save'):
        """Saves a note, model weights, and memory to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architecture and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder
        return: A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        if save_model:
            with open(os.path.join(path, 'qmodel.json'), 'w') as file:
                file.write(self.qmodel.to_json())
            self.qmodel.save_weights(os.path.join(path, 'qweights.h5'))
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset(
                    'next_states', data=self.next_states.array()
                )
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('rewards', data=self.rewards.array())
                file.create_dataset('terminals', data=self.terminals.array())
                if self.per_losses is not None:
                    file.create_dataset(
                        'per_losses', data=self.per_losses.array()
                    )
        return path


class PGAgent(Agent):
    """This class is an Agent that uses a Neural Network like the DQN Agent,
       but instead of learning to predict Q values, it predicts actions. It
       learns to predict these actions through Policy Gradients (PG).
    """

    def __init__(self, amodel, discounted_rate, create_memory=lambda: Memory(),
                 policy=None):
        """Initalizes the Policy Gradient Agent.
        params:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
            policy: A policy instance
        """
        Agent.__init__(self, amodel.output_shape[1:], policy)
        self.amodel = amodel
        self.discounted_rate = discounted_rate
        self.states = create_memory()
        self.actions = create_memory()
        self.drewards = create_memory()
        self.episode_rewards = []
        self.action_identity = np.identity(self.action_shape[0])
        self.metric = tf.keras.metrics.Mean(name='loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=tf.keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value, which is the state to predict
                   the action for
            training: A boolean, which determines if the
                      agent is training
        return: A value, which is the selected action
        """
        def _select_action():
            actions = self.amodel(np.expand_dims(state, axis=0),
                                  training=False)[0].numpy()
            return np.random.choice(np.arange(self.action_shape[0]),
                                    p=actions)
        if self.policy is None:
            return _select_action()
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self, training=False, memorizing=False,
                         batch_size=None, mini_batch=0, epochs=1,
                         repeat=1, entropy_coef=0, verbose=True):
        """Sets the playing data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            memorizing: A boolean, which determines if the agent
                        should be adding the information obtained
                        through playing an episode to memory
            batch_size: An integer, which is the size of each batch
                        within the mini-batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled and qvalues repredicted)
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'entropy_coef': entropy_coef,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        False, learning_params)

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action (discarded)
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
                      (discarded)
        """
        self.states.add(np.array(state))
        self.actions.add(self.action_identity[action])
        self.episode_rewards.append(reward)

    def forget(self):
        """Forgets or clears all memory."""
        self.states.reset()
        self.actions.reset()
        self.drewards.reset()
        self.episode_rewards.clear()

    def end_episode(self):
        """Ends the episode, and creates drewards based
           on the episodes rewards.
        """
        if len(self.episode_rewards) > 0:
            dreward = 0
            dreward_list = []
            for reward in reversed(self.episode_rewards):
                dreward *= self.discounted_rate
                dreward += reward
                dreward_list.append(dreward)
            self.episode_rewards.clear()
            for dreward in reversed(dreward_list):
                self.drewards.add(dreward)

    def _train_step(self, states, drewards, actions, entropy_coef):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            actions: A tensor that contains onehot encodings of
                     the action performed
            entropy_coef: A tensor constant float, which is the
                          coefficent of entropy to add to the
                          actor loss
        """
        with tf.GradientTape() as tape:
            y_pred = self.amodel(states, training=True)
            # log_softmax may be mathematically correct, but in practice
            # seems to give worse results
            log_probs = tf.reduce_sum(
                actions *
                tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=1
            )
            loss = -tf.reduce_mean(drewards * log_probs)
            entropy = tf.reduce_sum(
                y_pred * tf.math.log(y_pred + tf.keras.backend.epsilon()),
                axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric(loss)

    def _train(self, states, drewards, actions,
               epochs, batch_size, entropy_coef, verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted reward
                      for the action performed in the environment
            actions: A numpy array that contains onehot encodings of
                     the action performed
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if information should
                     be printed to the screen
        return: A float, which is the mean loss of batches (not exactly a loss)
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             drewards.astype(float_type),
             actions.astype(float_type))
        ).batch(batch_size)
        entropy_coef = tf.constant(entropy_coef,
                                   dtype=float_type)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._tf_train_step(*batch, entropy_coef)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            loss_results = self.metric.result()
            self.metric.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {loss_results}')
        return loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1, entropy_coef=0, verbose=True):
        """Trains the agent on a sample of its experiences.
        params:
            batch_size: An integer, which is the size of each batch
                        within the mini_batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled)
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        if mini_batch > 0 and len(self.states) > mini_batch:
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat+1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            arrays, _ = self.states.create_shuffled_subset(
                [self.states, self.drewards, self.actions], length
            )
            std = arrays[1].std()
            if std == 0:
                return False
            arrays[1] = (arrays[1] - arrays[1].mean()) / std
            self._train(*arrays, epochs, batch_size,
                        entropy_coef, verbose=verbose)

    def load(self, path, load_model=True, load_data=True):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architecture and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded
        """
        Agent.load(self, path)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(file.read())
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))

        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for state in file['states']:
                    self.states.add(state)
                for action in file['actions']:
                    self.actions.add(action)
                for dreward in file['drewards']:
                    self.drewards.add(dreward)

    def save(self, path, save_model=True, save_data=True, note='PGAgent Save'):
        """Saves a note, model weights, and memory to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architecture and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder
        return: A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('drewards', data=self.drewards.array())
        return path


class DDPGAgent(Agent):
    """This class (Deep Deterministic Policy Gradient Agent) is an Agent
       that uses two Neural Networks. An Actor network, which is like
       a PGAgent Network and a Critic Network like the DQNAgent
       Network. The critic rates the actions of the actor.
    """

    def __init__(self, policy, amodel, cmodel, discounted_rate,
                 create_memory=lambda: Memory(),
                 enable_target=False):
        """Initalizes the DDPG Agent.
        params:
            policy: A policy instance
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            cmodel: A keras model, which takes the state and a action as input
                     and outputs Q Values (a judgement)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
            enable_target: A boolean, which determines if a target model
                           should be used for the critic
        """
        print('WARNING: This implementation may be incorrect.')
        Agent.__init__(self, amodel.output_shape[1:], policy)
        self.amodel = amodel
        self.cmodel = cmodel
        self.target_cmodel = None
        self.enable_target = enable_target
        self.discounted_rate = discounted_rate
        if self.enable_target:
            self.target_amodel = keras.models.clone_model(amodel)
            self.target_amodel.compile(optimizer='sgd', loss='mse')
            self.target_cmodel = keras.models.clone_model(cmodel)
            self.target_cmodel.compile(optimizer='sgd', loss='mse')
        else:
            self.target_amodel = self.amodel
            self.target_cmodel = self.cmodel

        self.states = create_memory()
        self.next_states = create_memory()
        self.actions = create_memory()
        self.rewards = create_memory()
        self.terminals = create_memory()
        self.total_steps = 0
        self.metric_c = tf.keras.metrics.Mean(name='critic_loss')
        self.metric_a = tf.keras.metrics.Mean(name='actor_loss')

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=self.amodel.output_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.
        params:
            state: A value or list of values, which is the
                   state to predict the actions for
            training: A boolean, which determines if the
                      agent is training
        return: A value or list of values, which is the
                selected action
        """
        def _select_action():
            actions = self.amodel(np.expand_dims(state, axis=0),
                                  training=False)[0].numpy()
            return actions
        actions = self.policy.select_action(_select_action,
                                            training=training)
        return actions

    def set_playing_data(self, training=False, memorizing=False,
                         learns_in_episode=False, batch_size=None,
                         mini_batch=0, epochs=1, repeat=1,
                         target_update_interval=1, tau=1.0, verbose=True):
        """Sets the playing data.
        params:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            memorizing: A boolean, which determines if the agent
                        should be adding the information obtained
                        through playing an episode to memory
            learns_in_episode: A boolean, which determines if the agent
                               learns during a episode or at the end
            batch_size: An integer, which is the size of each batch
                        within the mini-batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled and qvalues repredicted)
            target_update_interval: An integer, which is the number of
                                    complete training instances
                                    (repeats do not count) until the
                                    target critic model weights are updated
            tau: A float, which is the strength of the copy from the
                 Actor or Critic model to the target models
                 (1.0 is a hard copy and less is softer)
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'target_update_interval': target_update_interval,
                           'tau': tau,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        learns_in_episode, learning_params)

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value or list of values, which is the
                    action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        self.states.add(np.array(state))
        self.next_states.add(np.array(new_state))
        self.actions.add(action)
        self.rewards.add(reward)
        self.terminals.add(0 if terminal else 1)

    def forget(self):
        """Forgets or clears all memory."""
        self.states.reset()
        self.next_states.reset()
        self.actions.reset()
        self.rewards.reset()
        self.terminals.reset()

    def update_target(self, tau):
        """Updates the target Actor and Critic Model weights.
        params:
            tau: A float, which is the strength of the copy from the
                 Actor or Critic model to the target models
                 (1.0 is a hard copy and less is softer)
        """
        if tau == 1.0:
            self.target_amodel.set_weights(self.amodel.get_weights())
            self.target_cmodel.set_weights(self.cmodel.get_weights())
        else:
            tws = self.target_amodel.trainable_variables
            ws = self.amodel.trainable_variables
            for ndx in range(len(tws)):
                tws[ndx] = ws[ndx] * tau + tws[ndx] * (1 - tau)
            tws = self.target_cmodel.trainable_variables
            ws = self.cmodel.trainable_variables
            for ndx in range(len(tws)):
                tws[ndx] = ws[ndx] * tau + tws[ndx] * (1 - tau)

    def _train_step(self, states, next_states, actions, terminals, rewards):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            next_states: A tensor that contains the states of
                         the environment after an action was performed
            actions: A tensor that contains the actions performed
            terminals: A tensor that contains ones for nonterminal
                       states and zeros for terminal states
            rewards: A tensor that contains the reward for the action
                     performed in the environment
        """
        next_actions = self.target_amodel(next_states, training=False)
        next_qvalues = self.target_cmodel([next_states, next_actions],
                                          training=False)
        qvalues_true = (rewards +
                        self.discounted_rate * next_qvalues * terminals)
        # Critic
        with tf.GradientTape() as tape:
            qvalues_pred = self.cmodel([states, actions], training=True)
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.compiled_loss._losses.fn(
                qvalues_true, qvalues_pred
            )
            loss = tf.reduce_mean(loss) + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        # Actor
        with tf.GradientTape() as tape:
            action_preds = self.amodel(states, training=True)
            loss = -tf.reduce_mean(tf.reduce_sum(
                self.cmodel([states, action_preds], training=False), axis=1
            ))
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric_a(loss)

    def _train(self, states, next_states, actions, terminals, rewards,
               epochs, batch_size, verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            next_states: A numpy array that contains the states of
                         the environment after an action was performed
            actions: A numpy array that contains the actions performed
            terminals: A numpy array that contains ones for nonterminal
                       states and zeros for terminal states
            rewards: A numpy array that contains the reward for the action
                     performed in the environment
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            verbose: A boolean, which determines if information should
                     be printed to the screen
        return: A float, which is the mean critic loss of the batches
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             next_states.astype(float_type),
             actions.astype(float_type),
             terminals.astype(float_type),
             rewards.astype(float_type))
        ).batch(batch_size)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._tf_train_step(*batch)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            critic_loss_results = self.metric_c.result()
            actor_loss_results = self.metric_a.result()
            self.metric_c.reset_states()
            self.metric_a.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'critic_loss: {critic_loss_results} - '
                      f'actor_loss: {actor_loss_results}')
        return critic_loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1,
              target_update_interval=1, tau=1.0, verbose=True):
        """Trains the agent on a sample of its experiences.
        params:
            batch_size: An integer, which is the size of each batch
                        within the mini_batch during one training instance
            mini_batch: An integer, which is the entire batch size for
                        one training instance
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            repeat: An integer, which is the times to repeat a training
                    instance in one training instance (similar to epochs,
                    but mini_batch is resampled and predictions are
                    repredicted)
            target_update_interval: An integer, which is the number of
                                    complete training instances
                                    (repeats do not count) until the
                                    target critic model weights are updated
            tau: A float, which is the strength of the copy from the
                 Actor or Critic model to the target models
                 (1.0 is a hard copy and less is softer)
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        self.total_steps += 1
        if mini_batch > 0 and len(self.states) > mini_batch:
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat+1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            arrays, _ = self.states.create_shuffled_subset(
                [self.states, self.next_states, self.actions,
                 self.terminals, self.rewards],
                length
            )
            self._train(*arrays, epochs, batch_size, verbose=verbose)

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

    def load(self, path, load_model=True, load_data=True):
        """Loads a save from a folder.
        params:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architectures and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded
        """
        Agent.load(self, path)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(file.read())
            with open(os.path.join(path, 'cmodel.json'), 'r') as file:
                self.cmodel = model_from_json(file.read())
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))
            self.cmodel.load_weights(os.path.join(path, 'cweights.h5'))
        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for state in file['states']:
                    self.states.add(state)
                for new_state in file['next_states']:
                    self.next_states.add(new_state)
                for action in file['actions']:
                    self.actions.add(action)
                for reward in file['rewards']:
                    self.rewards.add(reward)
                for terminal in file['terminals']:
                    self.terminals.add(terminal)

    def save(self, path, save_model=True, save_data=True,
             note='DDPGAgent Save'):
        """Saves a note, weights of the models, and memory to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architectures and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder
        return: A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            with open(os.path.join(path, 'cmodel.json'), 'w') as file:
                file.write(self.cmodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
            self.cmodel.save_weights(os.path.join(path, 'cweights.h5'))
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset('next_states',
                                    data=self.next_states.array())
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('rewards', data=self.rewards.array())
                file.create_dataset('terminals', data=self.terminals.array())
        return path
