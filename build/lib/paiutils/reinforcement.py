"""
Author: Travis Hammond
Version: 12_21_2020
"""


import os
from datetime import datetime

import h5py
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import model_from_json


class Environment:
    """This class handles the environment in which the Agent
       performs actions in and can get rewards from.
    """

    def __init__(self, state_shape, action_size):
        """Initalizes state and action shapes and sets the state.

        Args:
            state_shape: A tuple of integers, which is the
                         expected state shape for the agent,
                         or an integer of the discrete state
                         space
            action_size: An integer which is the discrete size
                         of the action space
        """
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.action_size = action_size

    def reset(self):
        """Resets the environment to its initialized state.

        Returns:
            A numpy ndarray, which is the state
        """
        self.state = None
        return self.state

    def step(self, action):
        """Moves the current state one step forward
           with regard to the action.

        Args:
            action: An integer or value that determines an action

        Returns:
            A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        self.state = None
        return self.state, 0, False

    def play_episode(self, agent, max_steps,
                     random=False, random_bounds=None,
                     render=False, verbose=True):
        """Plays a single complete episode with the agent.

        Args:
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

        Returns:
            A tuple of an integer (last step) and a float (total reward)
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
                    action = np.random.randint(0, self.action_size)
                else:
                    action = np.random.uniform(*random_bounds,
                                               size=self.action_size)
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
                      episode_verbose=None,
                      end_episode_callback=None):
        """Plays atleast 1 complete episode with the agent.

        Args:
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
            end_episode_callback: A function called at the end of each episode
                                  with episode count, steps, and total reward
                                  from the most recent episode. If True
                                  is returned, play_episodes will stop
                                  early.

        Returns:
            A float, which is the average total reward of all episodes
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
                if isinstance(agent, MemoryAgent):
                    mem_len = len(next(iter(agent.memory.values())))
                    mem_str = f' - Memory Size: {mem_len}'
                else:
                    mem_str = ''
                print(f'Time: {str_time} - Episode: {episode} - '
                      f'Steps: {step} - '
                      f'Total Reward: {total_reward} - '
                      f'Best Total Reward: {best_reward} - '
                      f'Average Total Reward: {total_rewards / episode}'
                      f'{mem_str}')
            if end_episode_callback is not None:
                end = end_episode_callback(
                    episode, step, total_reward
                )
                if end:
                    break
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

    def __init__(self, genv):
        """Initalizes state and action shapes and sets the state.

        Args:
            genv: An OpenAI Gym
        """
        self.genv = genv
        if isinstance(self.genv.observation_space, gym.spaces.Discrete):
            self.discrete_state_space = self.genv.observation_space.n
            self.state_shape = 1
        elif isinstance(self.genv.observation_space, gym.spaces.Box):
            self.discrete_state_space = None
            self.state_shape = self.genv.observation_space.shape
        else:
            raise NotImplementedError('Only Discrete and Box '
                                      'observation spaces '
                                      'are supported')

        if isinstance(self.genv.action_space, gym.spaces.Discrete):
            self.action_size = self.genv.action_space.n
        elif isinstance(self.genv.observation_space, gym.spaces.Box):
            if len(self.genv.action_space.shape) > 1:
                raise NotImplementedError('Box action spaces with more '
                                          'than one dimension are not '
                                          'supported')
            self.action_size = self.genv.action_space.shape[0]
        else:
            raise NotImplementedError('Only Discrete action '
                                      'spaces are supported')

    def reset(self):
        """Resets the environment to its initialized state.

        Returns:
            A numpy ndarray, which is the state
        """
        self.state = self.genv.reset()
        if self.discrete_state_space is None:
            return self.state
        else:
            return [self.state]

    def step(self, action):
        """Moves the current state one step forward
           with regard to the action.

        Args:
            action: An integer or value that determines an action

        Returns:
            A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        self.state, reward, terminal, _ = self.genv.step(action)
        if self.discrete_state_space is None:
            return self.state, reward, terminal
        else:
            return [self.state], reward, terminal

    def close(self):
        """Closes any threads or loose ends of the environment.
        """
        self.genv.close()

    def render(self):
        """Renders the environment.
        """
        self.genv.render()


class MultiSeqAgentEnvironment(Environment):
    """This class handles the environment in which multiple agents
       can perform actions against eachother in a sequential manner.
    """

    def __init__(self, state_shape, action_size):
        """Initalizes state and action shapes and sets the state.

        Args:
            state_shape: A tuple of integers, which is the
                         expected state shape for the agent
            action_size: An integer which is the discrete size
                         of the action space
        """
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.action_size = action_size

    def reset(self, num_agents):
        """Resets the environment to its initialized state.

        Args:
            num_agents: An integer, which is the number of states needed

        Returns:
            A numpy ndarray, which is the state
        """
        self.state = None
        return [self.state] * num_agents

    def step(self, agent_ndx, action):
        """Moves the current state one step forward
           with regard to the agent's action.

        Args:
            agent_ndx: An integer, which is the index of the
                       agent taking a step
            action: An integer or value that determines an action

        Returns:
            A tuple of a ndarray (state), a float/integer (reward),
                and a boolean (terminal state)
        """
        self.state = None
        return self.state, 0, False

    def play_episode(self, agents, max_steps, shuffle=True,
                     random=False, random_bounds=None,
                     render=False, verbose=True):
        """Plays a single complete episode with the agents.

        Args:
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

        Returns:
            A tuple of a list of integers (last steps)
                and a list of floats (total rewards)
        """
        num_agents = len(agents)
        ndxs = np.arange(num_agents)
        if shuffle:
            np.random.shuffle(ndxs)
        for ndx in ndxs:
            if not isinstance(agents[ndx], Agent):
                raise TypeError(f'The instance agent ({ndx}) is '
                                f'not a child of Agent.')
            if not isinstance(agents[ndx].playing_data, PlayingData):
                raise ValueError(f'Invalid playing_data value for agent '
                                 f'{ndx}. (Forgot to set playing_data?)')
        total_rewards = [0] * num_agents
        states = self.reset(num_agents)
        if render:
            self.render()
        break_loop = [False] * num_agents
        for step in range(1, max_steps + 1):
            for ndx in ndxs:
                if random:
                    if random_bounds is None:
                        action = np.random.randint(0, self.action_size)
                    else:
                        action = np.random.uniform(*random_bounds,
                                                   size=self.action_size)
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
                      verbose=True, episode_verbose=None,
                      end_episode_callback=None):
        """Plays at least 1 complete episode with the agents.

        Args:
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
            end_episode_callback: A function called at the end of each episode
                                  with episode count, steps, and total reward
                                  from the most recent episode. If True
                                  is returned, play_episodes will stop
                                  early.

        Returns:
            A list of floats, which are the average total reward of all
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
            if end_episode_callback is not None:
                end = end_episode_callback(
                    episode, step, total_reward
                )
                if end:
                    break
        return [tr / episode for tr in total_rewards]


class Policy:
    """This class is used for calling an Agent's action function."""

    def __init__(self):
        """Initalizes the Policy."""
        pass

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.

        Args:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        return action_func()

    def reset(self):
        """Resets any states."""
        pass

    def end_episode(self):
        """Tells the policy the episode ended."""
        pass


class GreedyPolicy(Policy):
    """This class is used for calling an Agent's action function and
       selecting the greediest action.
    """

    def __init__(self):
        super().__init__()

    def select_action(self, action_func, training):
        """Returns the action the Agent should take.

        Args:
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

        Args:
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

        Args:
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

        Args:
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

    def end_episode(self):
        """Tells the policy the episode ended and steps the decay."""
        self.stochasticity_decay_training.step()

    def reset(self):
        """Resets state of the stochasticity decay instance."""
        self.stochasticity_decay_training.reset()


class NoisePolicy(Policy):
    """This class is used for adding normal noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds):
        """Initalizes the Noise Policy.

        Args:
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

        Args:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = np.asarray(action_func())
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        noise = np.random.normal(scale=noise_scale, size=actions.shape)
        return np.clip(actions + noise, *self.action_bounds)

    def end_episode(self):
        """Tells the policy the episode ended and steps the decay."""
        self.noise_scale_decay_training.step()

    def reset(self):
        """Resets decay state."""
        self.noise_scale_decay_training.reset()


class UniformNoisePolicy(NoisePolicy):
    """This class is used for adding noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds, additive=False):
        """Initalizes the Uniform Noise Policy.

        Args:
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

        Args:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = np.asarray(action_func())
        noise = np.random.uniform(*self.action_bounds,
                                  size=actions.shape)
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        if self.additive:
            return np.clip(actions + noise * noise_scale, *self.action_bounds)
        else:
            return np.where(
                np.random.uniform(size=actions.shape) < noise_scale,
                noise, actions
            )


class TemporalNoisePolicy(NoisePolicy):
    """This class is used for adding temporal noise to an Agent's action."""

    def __init__(self, noise_scale_decay_training,
                 noise_scale_testing, action_bounds,
                 sigma=.3, theta=.15, dt=.01, init_noise=None):
        """Initalizes the Temporal Noise Policy.

        Args:
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

        Args:
            action_func: A function that returns a value
            training: A boolean, which determines if the
                      Agent is in a training states
        """
        actions = np.asarray(action_func())
        if self.init_noise is None:
            self.init_noise = np.full(actions.shape,
                                      np.mean(self.action_bounds))
            self.last_noise = self.init_noise
        if training:
            noise_scale = self.noise_scale_decay_training()
        else:
            noise_scale = self.noise_scale_testing
        if noise_scale == 0:
            return actions
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

    def __init__(self, initial_value, constant,
                 min_value=0, step_every_call=True):
        """Initalizes the state of the decay object.

        Args:
            initial_value: A float, which is the starting value to decay
            constant: A float, which is the slope/rate that the decay occurs
            min_value: A float, which is the minimum value the decay can reach
            step_every_call: A boolean, which determines if each call should
                             step the decay
        """
        if initial_value < min_value:
            raise ValueError(f'initial_value {initial_value} must '
                             f'be greater or equal to min_value {min_value}')
        self.initial_value = initial_value
        self.constant = constant
        self.min_value = min_value
        self.step_ever_call = step_every_call
        self.steps = 0

    def reset(self):
        """Resets the steps."""
        self.steps = 0

    def step(self):
        """Steps the decay forward."""
        self.steps += 1

    def __call__(self):
        """Returns the current value with regard to the state of decay.

        Returns:
            A float
        """
        value = self.initial_value - self.constant * self.steps
        if self.step_ever_call:
            self.step()
        return np.maximum(value, self.min_value)


class ExponentialDecay(Decay):
    """This class decays a initial value to a minimum
       value exponentially through a given number of steps.
       (formula: inital_value * (1 - rate)^steps + min_value)
    """

    def __init__(self, initial_value, rate, min_value=0,
                 step_every_call=True):
        """Initalizes the state of the decay object.

        Args:
            initial_value: A float, which is the starting value to decay
            rate: A float, which is the slope/rate that the decay occurs
            min_value: A float, which is the minimum value the decay can reach
            step_every_call: A boolean, which determines if each call should
                             step the decay
        """
        if initial_value < min_value:
            raise ValueError(f'initial_value {initial_value} must '
                             f'be greater or equal to min_value {min_value}')
        self.initial_value = initial_value
        self.rate = rate
        self.min_value = min_value
        self.step_ever_call = step_every_call
        self.steps = 0

    def __call__(self):
        """Returns the current value with regard to the state of decay.

        Returns:
            A float
        """
        value = self.initial_value * (1 - self.rate)**self.steps
        if self.step_ever_call:
            self.step()
        return np.maximum(value, self.min_value)


class LinearDecay(Decay):
    """This class decays a initial value to a minimum
       value linearly through a given number of steps.
       (formula: max(initial_value - (inital_value - min_value)
                     / total_steps * steps, min_value))
    """

    def __init__(self, initial_value, total_steps,
                 min_value=0, step_every_call=True):
        """Initalizes the state of the decay object.

        Args:
            initial_value: A float, which is the starting value to decay
            total_steps: An integer, which is the number of steps until
                         min_value would be reach
            min_value: A float, which is the minimum value the decay
                       can reach
            step_every_call: A boolean, which determines if each call should
                             step the decay
        """
        if initial_value < min_value:
            raise ValueError(f'initial_value {initial_value} must '
                             f'be greater or equal to min_value {min_value}')
        if not isinstance(total_steps, int):
            raise TypeError('total_steps should be an integer')
        self.initial_value = initial_value
        self.total_steps = total_steps
        self.min_value = min_value
        self.step_ever_call = step_every_call
        self.a = (-1 * (self.initial_value - self.min_value) /
                  self.total_steps)
        self.steps = 0

    def __call__(self):
        """Returns the current value with regard to the state of decay.

        Returns:
            A float
        """
        value = self.a * self.steps + self.initial_value
        if self.step_ever_call:
            self.step()
        return np.maximum(value, self.min_value)


class Memory:
    """This class is used by agents to store episode information.
       (uses a normal python list)
    """

    def __init__(self, max_len=None):
        """Initalizes the memory.

        Args:
            max_len: An integer, which is the max length of memory
                     (if reached, the oldest memory will be removed)
        """
        self.max_len = max_len
        self.buffer = []

    def __len__(self):
        """Returns the number of entries in the memory.

        Returns:
            An integer
        """
        return len(self.buffer)

    def add(self, x):
        """Adds a entry to memory.

        Args:
            x: A entry similar to other entries
        """
        self.buffer.append(x)
        if (self.max_len is not None
                and len(self.buffer) > self.max_len):
            del self.buffer[0]

    def __getitem__(self, key):
        """Returns an item given a key.

        Args:
            key: A valid key or index for a memory entry
        """
        return self.buffer[key]

    def __setitem__(self, key, value):
        """Sets a entry to a given key.

        Args:
            key: A valid key or index for a memory entry
            value: A entry similar to other entries
        """
        self.buffer[key] = value

    def array(self):
        """Returns a copy of the memory.

        Returns:
            A numpy ndarray
        """
        return np.array(self.buffer)

    def reset(self):
        """Resets or clears the memory.
        """
        self.buffer.clear()

    def end_episode(self):
        """Tells memory an episode ended.
        """
        pass

    def save(self, file, name):
        """Creates a h5py dataset with the memory data.

        Args:
            file: A h5py open file for writing
            name: A string, which is the dataset name
        """
        file.create_dataset(name, data=self.array())

    def load(self, file, name):
        """Loads a h5py dataset with the saved memory data.

        Args:
            file: A h5py open file for reading
            name: A string, which is the dataset name
        """
        for element in file[name]:
            self.add(element)

    @staticmethod
    def create_shuffled_subset(memories, subset_size, weights=None):
        """Creates a list of numpy arrays of a shuffled subset of memories.

        Args:
            memories: A list of Memeory Objects
            subset_size: A integer, which is the size of the
                         outer dimension of each ndarray
            weights: A list of probabilities that add up to 1

        Returns:
            arrays and shuffled indexes
        """
        length = len(memories[0])
        if subset_size > length:
            raise ValueError(f'Subset size {subset_size} is '
                             f'greater than memory length {length}')
        for memory in memories:
            if len(memory) != length:
                raise ValueError('Memories are not all the same length.')
            if not isinstance(memory, Memory):
                raise TypeError('Memories must also be Memory '
                                'or subclass instances')
        indexes = np.random.choice(np.arange(length),
                                   size=subset_size, replace=False,
                                   p=weights)
        arrays = [np.empty((subset_size, *memory[0].shape))
                  if isinstance(memory[0], np.ndarray)
                  else np.empty(subset_size)
                  for memory in memories]
        for ndx, rndx in enumerate(indexes):
            for array, memory in zip(arrays, memories):
                array[ndx] = memory[rndx]
        return arrays, indexes


class ETDMemory(Memory):
    """This class is for the efficient storage of time distributed states.
       This type of memory should only be used for states.
    """

    def __init__(self, num_time_steps, void_state, max_len=None):
        """Initalizes the memory.

        Args:
            num_time_steps: An integer, which is the number of
                            states that make up a complete state
            void_state: A ndarray, which is used when there is not
                        enough states to create a complete state
            max_len: An integer, which is the max length of memory
                     (if reached, the oldest memory will be removed)
        """
        if max_len is not None:
            raise NotImplementedError('max_len is not yet implemented')
        self.num_time_steps = num_time_steps
        self.max_len = max_len
        self.buffer = [void_state]
        self.ndxs = []
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def __len__(self):
        """Returns the number of entries in the memory.

        Returns:
            An integer
        """
        return len(self.ndxs)

    def add(self, x):
        """Adds a entry to memory.

        Args:
            x: A entry similar to other entries
        """
        self.step_ndxs = np.roll(self.step_ndxs, -1)
        self.step_ndxs[-1] = len(self.buffer)
        self.ndxs.append(self.step_ndxs)
        self.buffer.append(x)

    def __getitem__(self, key):
        """Returns an item given a key.

        Args:
            key: A valid key or index for a memory entry
        """
        return self.buffer[key + 1 if key >= 0 else key]

    def __setitem__(self, key, value):
        """Sets a entry to a given key.

        Args:
            key: A valid key or index for a memory entry
            value: A entry similar to other entries
        """
        self.buffer[key + 1 if key >= 0 else key] = value

    def array(self):
        """Returns a copy of the memory.

        Returns:
            A numpy ndarray
        """
        return np.array(self.buffer)[np.array(self.ndxs)]

    def reset(self):
        """Resets or clears the memory.
        """
        void_state = self.buffer[0]
        self.buffer.clear()
        self.buffer.append(void_state)
        self.ndxs.clear()
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def end_episode(self):
        """Tells memory an episode ended.
        """
        self.step_ndxs = np.zeros(self.num_time_steps, dtype=np.int)

    def save(self, file, name):
        """Creates a h5py dataset with the memory data.

        Args:
            file: A h5py open file for writing
            name: A string, which is the dataset name
        """
        file.create_dataset(f'{name}_buffer', data=np.array(self.buffer))
        file.create_dataset(f'{name}_ndxs', data=np.array(self.ndxs))

    def load(self, file, name):
        """Loads a h5py dataset with the saved memory data.

        Args:
            file: A h5py open file for reading
            name: A string, which is the dataset name
        """
        for element in file[f'{name}_ndxs']:
            self.ndxs.append(element)
            if element.shape != (self.num_time_steps,):
                raise ValueError('Cannot load dataset: '
                                 'invalid number of time steps')

        for element in file[f'{name}_buffer']:
            self.buffer.append(element)

    @staticmethod
    def create_shuffled_subset(memories, subset_size, weights=None):
        """Creates a list of numpy arrays of a shuffled subset of memories.

        Args:
            memories: A list of Memeory Objects (not asserted but assumed)
            subset_size: A integer, which is the size of the
                         outer dimension of each ndarray
            weights: A list of probabilities that add up to 1

        Returns:
            arrays and shuffled indexes
        """
        length = len(memories[0])
        if subset_size > length:
            raise ValueError(f'Subset size {subset_size} is '
                             f'greater than memory length {length}')
        for memory in memories:
            if len(memory) != length:
                raise ValueError('Memories are not all the same length.')
            if not isinstance(memory, Memory):
                raise TypeError('Memories must also be Memory '
                                'or subclass instances')
        indexes = np.random.choice(np.arange(len(memories[0])),
                                   size=subset_size, replace=False,
                                   p=weights)
        arrays = []
        for memory in memories:
            if isinstance(memory, ETDMemory):
                arrays.append(np.empty((subset_size,
                                        memory.num_time_steps,
                                        *memory.buffer[0].shape)))
            elif isinstance(memory[0], np.ndarray):
                arrays.append(np.empty((subset_size, *memory[0].shape)))
            else:
                arrays.append(np.empty(subset_size))
        for ndx, rndx in enumerate(indexes):
            for array, memory in zip(arrays, memories):
                if isinstance(memory, ETDMemory):
                    for andx, sndx in enumerate(memory.ndxs[rndx]):
                        array[ndx, andx] = memory.buffer[sndx]
                else:
                    array[ndx] = memory[rndx]
        return arrays, indexes


class RingMemory(Memory):
    def __init__(self, max_len):
        """Initalizes the memory.

        Args:
            max_len: An integer, which is the max length of memory
                     (if reached, the oldest memory will be removed)
        """
        Memory.__init__(self, max_len=max_len)


class PlayingData:
    """This class is used for containing data
       that the environment needs to know, but the agent has.
    """

    def __init__(self, training, memorizing, epochs,
                 learns_in_episode, learning_params):
        """Initalizes the data.

        Args:
            training: A boolean, which determines if the agent
                      should be treated as in a training mode
            memorizing: A boolean, which determines if the agent
                        should be adding the information obtained
                        through playing an episode to memory
            epochs: An integer, which is the number of epochs to train
                    in one training instance
            learns_in_episode: A boolean, which determines if the agent
                               learns during a episode or at the end
            learning_Args: A dictionary of parameters for the agent's
                             learn method
        """
        if not (training is True or training is False):
            raise ValueError('Invalid training value. Must be True or False.')
        if not (memorizing is True or memorizing is False):
            raise ValueError(
                'Invalid memorizing value. Must be True or False.')
        if epochs < 0:
            raise ValueError('Invalid epoch value. Must '
                             'be greater or equal to zero.')
        if not (learns_in_episode is True or learns_in_episode is False):
            raise ValueError('Invalid learns_in_episode value. '
                             'Must be True or False.')
        if not isinstance(learning_params, dict):
            raise TypeError('Invalid learning_params value. '
                            'Must be a dictionary.')
        self.training = training
        self.memorizing = memorizing
        self.epochs = epochs
        self.learns_in_episode = learns_in_episode
        self.learning_params = learning_params


class Agent:
    """This class is the base class for all agent classes,
       and essentially is a random agent.
    """

    def __init__(self, action_size, policy):
        """Initalizes the agent.

        Args:
            action_size: An integer which is the discrete size
                         of the action space
            policy: A policy instance
        """
        if not isinstance(action_size, int):
            raise TypeError('action_size must be an integer')
        if not isinstance(policy, Policy):
            raise TypeError('policy must be a Policy instance')
        self.action_size = action_size
        self.policy = policy
        self.playing_data = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value or list of values, which is the
                   state to get the action for
            training: A boolean, which determines if the
                      agent is training

        Returns:
            A value, which is the selected action
        """
        def _select_action():
            return np.random.random(self.action_size)
        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self):
        """Sets the episode data."""
        self.playing_data = PlayingData(False, False, 0, False, {})

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.
           (For this agent all memory is discarded)

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float, which is the evaluation of
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
        self.policy.end_episode()

    def learn(self, verbose=True):
        """Trains the agent on a batch of its experiences.
           (For this agent no learning is needed)

        Args:
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        pass

    def load(self, path):
        """Loads a save from a folder.

        Args:
            path: A string, which is the path to a folder to load

        Returns:
            A string of note.txt
        """
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            note = file.read()
        return note

    def save(self, path, note):
        """Saves a note to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        time = datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            file.write(note)
        return path


class QAgent(Agent):
    """This class is a Q-learning Agent. It does not uses a neural network,
       but instead uses a table.
    """

    def __init__(self, discrete_state_space, action_size,
                 policy, discounted_rate):
        """Initalizes the Q-learning agent.

        Args:
            discrete_state_space: An integer, which is the size of
                                  the state space
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
                                self.action_size))
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.terminal = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value or list of values, which is the
                   state to look up the action for in the table
            training: A boolean, which determines if the
                      agent is training

        Returns:
            A value, which is the selected action
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

        Args:
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

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float, which is the evaluation of
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

        Args:
            learning_rate: A float, which is the rate that the table
                           is updated with the currect Q reward
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        if self.state is None:
            raise ValueError('Memory is empty')
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

        Args:
            path: A string, which is the path to a folder to load

        Returns:
            A string of note.txt
        """
        note = Agent.load(self, path)
        self.qtable = np.load(os.path.join(path, 'qtable.npy'))
        return note

    def save(self, path, note='QAgent Save'):
        """Saves a note and qtable to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder

        Returns:
            A string, which is the complete path of the save
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

        Args:
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
                                 self.action_size))
        self.selected_qtable = lambda: self.qtables[0, 0]
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.terminal = None

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value or list of values, which is the
                   state to look up the action for in the table
            training: A boolean, which determines if the
                      agent is training

        Returns:
            A value, which is the selected action
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

        Args:
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
            lrn = learning_rate_ndx
        if discounted_rate_ndx is not None:
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

        Args:
            learning_rate_ndx: An integer, which is a ndx for
                               the learning rates
            discounted_rate_ndx: An integer, which is a ndx for
                                 the discounted rate
            verbose: A boolean, which determines if information
                     should be printed to the screen
        """
        if self.state is None:
            raise ValueError('Memory is empty')
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

        Args:
            path: A string, which is the path to a folder to load
                  from

        Returns:
            A string of note.txt
        """
        note = Agent.load(self, path)
        self.qtables = np.load(os.path.join(path, 'qtables.npy'))
        return note

    def save(self, path, note='PQAgent Save'):
        """Saves a note and qtables to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            note: A string, which is the note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        np.save(os.path.join(path, 'qtables.npy'), self.qtables)
        return path


class MemoryAgent(Agent):
    """This class is the base class for all agent that use memory.
    """

    def __init__(self, action_size, policy):
        """Initalizes the agent.

        Args:
            action_size: An integer which is the discrete size
                         of the action space
            policy: A policy instance
        """
        Agent.__init__(self, action_size, policy)
        self.memory = {}
        self.time_distributed_states = None

    def forget(self):
        """Forgets or clears all memory."""
        Agent.end_episode(self)
        for memory in self.memory.values():
            memory.reset()

    def end_episode(self):
        """Ends the episode for the agent."""
        Agent.end_episode(self)
        for memory in self.memory.values():
            memory.end_episode()
        if ('states' in self.memory
                and isinstance(self.memory['states'], ETDMemory)):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])

    def load(self, path, load_data=True):
        """Loads a save from a folder.

        Args:
            path: A string, which is the path to a folder to load
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded

            Returns:
                A string of note.txt
        """
        note = Agent.load(self, path)
        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for name, memory in self.memory.items():
                    memory.load(file, name)
        return note

    def save(self, path, save_data=True, note='MemoryAgent'):
        """Saves a note and memory to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        path = Agent.save(self, path, note)
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                for name, memory in self.memory.items():
                    memory.save(file, name)
        return path


class DQNAgent(MemoryAgent):
    """This class is an Agent that uses a Deep Q Network instead of
       a table like the QAgent. This allows for generalizations and
       large environment states.
    """

    @staticmethod
    def get_dueling_output_layer(action_size, dueling_type='avg'):
        assert dueling_type in ['avg', 'max', 'naive'], (
            "Dueling type must be 'avg', 'max', or 'naive'"
        )

        def layer(x1, x2):
            x1 = keras.layers.Dense(1)(x1)
            x2 = keras.layers.Dense(action_size)(x2)
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
            return keras.layers.Lambda(dueling, output_shape=(action_size,),
                                       name='q_output')(x)
        return layer

    def __init__(self, policy, qmodel, discounted_rate,
                 create_memory=lambda shape, dtype: Memory(),
                 enable_target=True, enable_double=False,
                 enable_per=False):
        """Initalizes the Deep Q Network Agent.

        Args:
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
            enable_per: A boolean, which determines if prioritized experience
                        replay should be used (The implementation for this is
                        not the normal tree implementation, and only weights
                        the probabilily of being choosen and not also the
                        gradient)
        """
        MemoryAgent.__init__(self, qmodel.output_shape[1], policy)
        self.qmodel = qmodel
        self.qmodel.compiled_loss.build(
            tf.zeros(self.qmodel.output_shape[1:])
        )
        self.target_qmodel = None
        self.enable_target = enable_target or enable_double
        self.enable_double = enable_double
        if self.enable_target:
            self.target_qmodel = keras.models.clone_model(qmodel)
            self.target_qmodel.compile(optimizer='sgd', loss='mse')
        else:
            self.target_qmodel = self.qmodel
        self.discounted_rate = discounted_rate
        self.states = create_memory(self.qmodel.input_shape,
                                    keras.backend.floatx())
        self.next_states = create_memory(self.qmodel.input_shape,
                                         keras.backend.floatx())
        self.actions = create_memory(self.qmodel.output_shape,
                                     keras.backend.floatx())
        self.rewards = create_memory((None,),
                                     keras.backend.floatx())
        self.terminals = create_memory((None,),
                                       keras.backend.floatx())
        self.memory = {
            'states': self.states, 'next_states': self.next_states,
            'actions': self.actions, 'rewards': self.rewards,
            'terminals': self.terminals
        }
        if isinstance(self.memory['states'], ETDMemory):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])
        if enable_per:
            self.per_losses = create_memory((None,),
                                            keras.backend.floatx())
            self.memory['per_losses'] = self.per_losses
            # assuming the true max loss will be less than 100
            # at least at the begining
            self.max_loss = 100.0
        else:
            self.per_losses = None
        self.action_identity = np.identity(self.action_size)
        self.total_steps = 0
        self.metric = keras.metrics.Mean(name='loss')

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.qmodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=self.qmodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value, which is the state to predict
                   the Q values for
            training: A boolean, which determines if the
                      agent is training

        Returns:
            A value, which is the selected action
        """
        if (self.time_distributed_states is not None
                and state.shape == self.qmodel.input_shape[2:]):
            self.time_distributed_states = np.roll(
                self.time_distributed_states, -1
            )
            self.time_distributed_states[-1] = state
            state = self.time_distributed_states

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

        Args:
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

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float, which is the evaluation of
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

    def update_target(self, tau):
        """Updates the target Q Model weights.

        Args:
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

        Args:
            states: A tensor that contains environment states
            next_states: A tensor that contains the states of
                         the environment after an action was performed
            actions: A tensor that contains onehot encodings of
                     the action performed
            terminals: A tensor that contains ones for nonterminal
                       states and zeros for terminal states
            rewards: A tensor that contains the reward for the action
                     performed in the environment

        Returns:
            A loss for this batch of data
        """
        if self.enable_double:
            qvalues = self.qmodel(next_states, training=False)
            actions = tf.argmax(qvalues, axis=-1)
            qvalues = self.target_qmodel(next_states, training=False)
            qvalues = tf.squeeze(tf.gather(qvalues, actions[:, tf.newaxis],
                                           axis=-1, batch_dims=1))
            actions = tf.one_hot(actions, self.action_size,
                                 dtype=qvalues.dtype)
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
            loss = self.qmodel.compiled_loss._losses[0].fn(
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

        Args:
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

        Returns:
            A list of floats, which are the absolute losses for all
                the data
        """
        length = states.shape[0]
        float_type = keras.backend.floatx()
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

        Args:
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
        if batch_size is None:
            batch_size = len(self.states)
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

        Args:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architecture and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded

        Returns:
            A string of note.txt
        """
        note = MemoryAgent.load(self, path, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'qmodel.json'), 'r') as file:
                self.qmodel = model_from_json(file.read())
            self.qmodel.load_weights(os.path.join(path, 'qweights.h5'))
            if self.enable_target:
                self.target_qmodel = keras.models.clone_model(self.qmodel)
                self.target_qmodel.compile(optimizer='sgd', loss='mse')
            else:
                self.target_qmodel = self.qmodel
        return note

    def save(self, path, save_model=True,
             save_data=True, note='DQNAgent Save'):
        """Saves a note, model weights, and memory to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architecture and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        path = MemoryAgent.save(self, path, save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'qmodel.json'), 'w') as file:
                file.write(self.qmodel.to_json())
            self.qmodel.save_weights(os.path.join(path, 'qweights.h5'))
        return path


class PGAgent(MemoryAgent):
    """This class is an Agent that uses a Neural Network like the DQN Agent,
       but instead of learning to predict Q values, it predicts actions. It
       learns to predict these actions through Policy Gradients (PG).
    """

    def __init__(self, amodel, discounted_rate,
                 create_memory=lambda shape, dtype: Memory()):
        """Initalizes the Policy Gradient Agent.

        Args:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
        """
        output_shape = amodel.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[-1]
        MemoryAgent.__init__(self, output_shape[1], Policy())
        self.amodel = amodel
        self.discounted_rate = discounted_rate
        self.states = create_memory(self.amodel.input_shape,
                                    keras.backend.floatx())
        self.actions = create_memory(output_shape,
                                     keras.backend.floatx())
        self.drewards = create_memory((None,),
                                      keras.backend.floatx())
        self.memory = {
            'states': self.states, 'actions': self.actions,
            'drewards': self.drewards,
        }
        if isinstance(self.memory['states'], ETDMemory):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])
        self.episode_rewards = []
        self.action_identity = np.identity(self.action_size)
        self.metric = keras.metrics.Mean(name='loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value, which is the state to predict
                   the action for
            training: A boolean, which determines if the
                      agent is training (does nothing)

        Returns:
            A value, which is the selected action
        """
        if (self.time_distributed_states is not None
                and state.shape == self.amodel.input_shape[2:]):
            self.time_distributed_states = np.roll(
                self.time_distributed_states, -1
            )
            self.time_distributed_states[-1] = state
            state = self.time_distributed_states

        actions = self.amodel(np.expand_dims(state, axis=0),
                              training=False)
        if isinstance(actions, list):
            return actions[-1][0].numpy()
        return np.random.choice(np.arange(self.action_size),
                                p=actions[0].numpy())

    def set_playing_data(self, training=False, memorizing=False,
                         batch_size=None, mini_batch=0, epochs=1,
                         repeat=1, entropy_coef=0, verbose=True):
        """Sets the playing data.

        Args:
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

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value, which is the action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action (discarded)
            reward: A float, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
                      (discarded)
        """
        self.states.add(np.array(state))
        self.actions.add(self.action_identity[action])
        self.episode_rewards.append(reward)

    def end_episode(self):
        """Ends the episode and creates drewards based
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

        MemoryAgent.end_episode(self)

    def _train_step(self, states, drewards, actions, entropy_coef):
        """Performs one gradient step with a batch of data.

        Args:
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
            log_y_pred = tf.math.log(y_pred + keras.backend.epsilon())
            log_probs = tf.reduce_sum(
                actions * log_y_pred, axis=1
            )
            loss = -tf.reduce_mean(drewards * log_probs)
            entropy = tf.reduce_sum(
                y_pred * log_y_pred, axis=1
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

        Args:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted reward
                      for the action performed in the environment
            actions: A numpy array that contains the actions performed
                     (onehot encodings for discrete action spaces)
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if information should
                     be printed to the screen

        Returns:
            A float, which is the mean loss of batches (not exactly a loss)
        """
        length = states.shape[0]
        float_type = keras.backend.floatx()
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

        Args:
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
        if batch_size is None:
            batch_size = len(self.states)
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

        Args:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architecture and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded

        Returns:
            A string of note.txt
        """
        note = MemoryAgent.load(self, path, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(file.read())
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))
        return note

    def save(self, path, save_model=True, save_data=True, note='PGAgent Save'):
        """Saves a note, model weights, and memory to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architecture and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        path = MemoryAgent.save(self, path, save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
        return path


class DDPGAgent(MemoryAgent):
    """This class (Deep Deterministic Policy Gradient Agent) is an Agent
       that uses two Neural Networks. An Actor network, which is like
       a PGAgent Network and a Critic Network like the DQNAgent
       Network. The critic rates the actions of the actor.
    """

    def __init__(self, policy, amodel, cmodel, discounted_rate,
                 create_memory=lambda shape, dtype: Memory(),
                 enable_target=False):
        """Initalizes the DDPG Agent.

        Args:
            policy: A NoisePolicy instance
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
        if not isinstance(policy, NoisePolicy):
            raise ValueError('The policy parameter must be a '
                             'instance of NoisePolicy.')
        MemoryAgent.__init__(self, amodel.output_shape[1], policy)
        self.amodel = amodel
        self.cmodel = cmodel
        if isinstance(self.cmodel.output_shape, list):
            self.cmodel.compiled_loss.build(
                tf.zeros(self.cmodel.output_shape[0][1:])
            )
        else:
            self.cmodel.compiled_loss.build(
                tf.zeros(self.cmodel.output_shape[1:])
            )
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

        self.states = create_memory(self.amodel.input_shape,
                                    keras.backend.floatx())
        self.next_states = create_memory(self.amodel.input_shape,
                                         keras.backend.floatx())
        self.actions = create_memory(self.amodel.output_shape,
                                     keras.backend.floatx())
        self.rewards = create_memory((None,),
                                     keras.backend.floatx())
        self.terminals = create_memory((None,),
                                       keras.backend.floatx())
        self.memory = {
            'states': self.states, 'next_states': self.next_states,
            'actions': self.actions, 'rewards': self.rewards,
            'terminals': self.terminals
        }
        if isinstance(self.memory['states'], ETDMemory):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])
        self.total_steps = 0
        self.metric_c = keras.metrics.Mean(name='critic_loss')
        self.metric_a = keras.metrics.Mean(name='actor_loss')

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=self.amodel.output_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        """Returns the action the Agent "believes" to be
           suited for the given state.

        Args:
            state: A value or list of values, which is the
                   state to predict the actions for
            training: A boolean, which determines if the
                      agent is training

        Returns:
            A value or list of values, which is the
                selected action
        """
        if (self.time_distributed_states is not None
                and state.shape == self.amodel.input_shape[2:]):
            self.time_distributed_states = np.roll(
                self.time_distributed_states, -1
            )
            self.time_distributed_states[-1] = state
            state = self.time_distributed_states

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

        Args:
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

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value or list of values, which is the
                    action the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        self.states.add(np.array(state))
        self.next_states.add(np.array(new_state))
        self.actions.add(action)
        self.rewards.add(reward)
        self.terminals.add(0 if terminal else 1)

    def update_target(self, tau):
        """Updates the target Actor and Critic Model weights.

        Args:
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

        Args:
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
        next_qvalues = tf.squeeze(
            self.target_cmodel([next_states, next_actions], training=False)
        )
        qvalues_true = (rewards +
                        self.discounted_rate * next_qvalues * terminals)

        # Critic
        with tf.GradientTape() as tape:
            qvalues_pred = tf.squeeze(
                self.cmodel([states, actions], training=True)
            )
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.compiled_loss._losses[0].fn(
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
            loss = -tf.reduce_mean(tf.squeeze(
                self.cmodel([states, action_preds], training=False)
            ))
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric_a(loss)

    def _train(self, states, next_states, actions, terminals, rewards,
               epochs, batch_size, verbose=True):
        """Performs multiple gradient steps of all the data.

        Args:
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

        Returns:
            A float, which is the mean critic loss of the batches
        """
        length = states.shape[0]
        float_type = keras.backend.floatx()
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
                      f'actor_loss: {actor_loss_results} - '
                      f'critic_loss: {critic_loss_results}')
        return critic_loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1,
              target_update_interval=1, tau=1.0, verbose=True):
        """Trains the agent on a sample of its experiences.

        Args:
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
        if batch_size is None:
            batch_size = len(self.states)
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

        Args:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architectures and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded

        Returns:
            A string of note.txt
        """
        note = MemoryAgent.load(self, path, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(file.read())
            with open(os.path.join(path, 'cmodel.json'), 'r') as file:
                self.cmodel = model_from_json(file.read())
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))
            self.cmodel.load_weights(os.path.join(path, 'cweights.h5'))
        return note

    def save(self, path, save_model=True, save_data=True,
             note='DDPGAgent Save'):
        """Saves a note, weights of the models, and memory to a new folder.

        Args:
            path: A string, which is the path to a folder to save within
            save_model: A boolean, which determines if the model
                        architectures and weights
                        should be saved
            save_data: A boolean, which determines if the memory
                       should be saved
            note: A string, which is a note to save in the folder

        Returns:
            A string, which is the complete path of the save
        """
        path = MemoryAgent.save(self, path, save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            with open(os.path.join(path, 'cmodel.json'), 'w') as file:
                file.write(self.cmodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
            self.cmodel.save_weights(os.path.join(path, 'cweights.h5'))
        return path
