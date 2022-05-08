"""
Author: Travis Hammond
Version: 5_7_2022
"""


import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import model_from_json

from paiutils.reinforcement import (
    Memory, PlayingData, DQNAgent,
    MemoryAgent, PGAgent, DDPGAgent
)


class DQNPGAgent(DQNAgent, PGAgent):
    """This class is an Agent that uses a Deep Q Network by default to
       select actions, but can be easily be changed to a policy gradient
       based network to predict actions.
    """

    def __init__(self, policy, qmodel, amodel, discounted_rate,
                 create_memory=lambda shape, dtype: Memory(),
                 enable_target=True, enable_double=False,
                 enable_per=False):
        """Initalizes the Deep Q Network and Policy Gradient Agent.

        Args:
            policy: A policy instance (for DQN Agent)
            qmodel: A keras model, which takes the state as input and outputs
                    Q Values
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not be applied,
                    and compiled loss are not used)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
            enable_target: A boolean, which determines if a target model
                           should be used
            enable_double: A boolean, which determiens if the Double Deep Q
                           Network should be used
            enable_per: A boolean, which determines if prioritized experience
                        replay should be used
                        (The implementation for this is not the normal tree
                         implementation, and only weights the probabilily of
                         being choosen)
        """
        DQNAgent.__init__(self, policy, qmodel, discounted_rate,
                          create_memory=create_memory,
                          enable_target=enable_target,
                          enable_double=enable_double,
                          enable_per=enable_per)
        self.amodel = amodel
        self.uses_dqn_method = True
        self.drewards = create_memory((None,),
                                      keras.backend.floatx())
        self.memory['drewards'] = self.drewards
        self.episode_rewards = []
        self.pg_metric = keras.metrics.Mean(name='pg_loss')
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
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()))
        )

    def use_dqn(self):
        self.uses_dqn_method = True

    def use_pg(self):
        self.uses_dqn_method = False

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
        if self.uses_dqn_method:
            return DQNAgent.select_action(self, state, training=training)
        else:
            return PGAgent.select_action(self, state, training=training)

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
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        DQNAgent.add_memory(self, state, action, new_state, reward, terminal)
        self.episode_rewards.append(reward)

    def forget(self):
        """Forgets or clears all memory."""
        DQNAgent.forget(self)
        self.episode_rewards.clear()

    def end_episode(self):
        """Ends the episode, and creates drewards based
           on the episodes rewards.
        """
        PGAgent.end_episode(self)

    def _train_step(self, states, next_states, actions,
                    terminals, rewards, drewards):
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
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
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

        abs_loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

        with tf.GradientTape() as tape:
            y_pred = self.amodel(states, training=True)
            # log_softmax may be more stable, but in practice
            # seems to give worse results
            log_probs = tf.reduce_sum(
                actions *
                tf.math.log(y_pred + keras.backend.epsilon()), axis=1
            )
            loss = -tf.reduce_mean(drewards * log_probs)
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.pg_metric(loss)

        return abs_loss

    def _train(self, states, next_states, actions, terminals,
               rewards, drewards, epochs, batch_size, verbose=True):
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
            drewards: A numpy array that contains the discounted reward
                      for the action performed in the environment
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
             rewards.astype(float_type),
             drewards.astype(float_type))
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
            dqn_loss_results = self.metric.result()
            self.metric.reset_states()
            pg_loss_results = self.pg_metric.result()
            self.pg_metric.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'dqn_loss: {dqn_loss_results} - '
                      f'pg_loss: {pg_loss_results}')
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
                                    target qmodel weights are updated
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
                     self.terminals, self.rewards, self.drewards],
                    length
                )
            else:
                per_losses = self.per_losses.array()
                self.max_loss = per_losses.max()
                per_losses = per_losses / per_losses.sum()
                arrays, indexes = self.states.create_shuffled_subset(
                    [self.states, self.next_states, self.actions,
                     self.terminals, self.rewards, self.drewards],
                    length, weights=per_losses
                )

            std = arrays[-1].std()
            if std == 0:
                return False
            arrays[-1] = (arrays[-1] - arrays[-1].mean()) / std
            losses = self._train(*arrays, epochs, batch_size, verbose=verbose)
            if self.per_losses is not None:
                for ndx, loss in zip(indexes, losses):
                    self.per_losses[ndx] = loss

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

    def load(self, path, load_model=True, load_data=True, custom_objects=None):
        """Loads a save from a folder.

        Args:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architectures and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model

        Returns:
            A string of note.txt
        """
        note = DQNAgent.load(
            self, path, load_model=load_model, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))
        return note

    def save(self, path, save_model=True,
             save_data=True, note='DQNPGAgent Save'):
        """Saves a note, model weights, and memory to a new folder.

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
        path = DQNAgent.save(self, path, save_model=save_model,
                             save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
        return path


class A2CAgent(PGAgent):
    """This class (Advantage Actor-Critic) is like the PGAgent, but it also
       has a critic network which is used to estimate the value function
       in order to train the Actor network on the advantages instead of
       the discounted rewards.
    """

    def __init__(self, amodel, cmodel, discounted_rate,
                 lambda_rate=0, create_memory=lambda shape, dtype: Memory()):
        """Initalizes the Policy Gradient Agent.

        Args:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            cmodel: A keras model, which takes the state as input and outputs
                    the value of that state
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            lambda_rate: A float within 0.0-1.0, which if nonzero will enable
                         generalized advantage estimation
            create_memory: A function, which returns a Memory instance
        """
        PGAgent.__init__(self, amodel, discounted_rate,
                         create_memory=create_memory)
        self.cmodel = cmodel
        self.cmodel.compiled_loss.build(
            tf.zeros(self.cmodel.output_shape[1:])
        )
        self.lambda_rate = lambda_rate
        if lambda_rate != 0:
            self.terminals = create_memory((None,),
                                           keras.backend.floatx())
            self.rewards = create_memory((None,),
                                         keras.backend.floatx())
            self.memory['terminals'] = self.terminals
            self.memory['rewards'] = self.rewards
        self.metric_c = keras.metrics.Mean(name='critic_loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()))
        )

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
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        self.states.add(np.array(state))
        self.actions.add(self.action_identity[action])
        self.episode_rewards.append(reward)
        if self.lambda_rate > 0:
            self.terminals.add(terminal)
            self.rewards.add(reward)

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
            if self.lambda_rate > 0:
                self.terminals[-1] = True

        MemoryAgent.end_episode(self)

    def _train_step(self, states, drewards, advantages,
                    actions, entropy_coef):
        """Performs one gradient step with a batch of data.

        Args:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            advantages: A tensor, which if valid (lambda_rate > 0) contains
                        advantages for the actions performed
            actions: A tensor that contains onehot encodings of
                     the action performed
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
        """
        with tf.GradientTape() as tape:
            value_pred = tf.squeeze(self.cmodel(states, training=True))
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.compiled_loss._losses[0].fn(drewards,
                                                           value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            action_pred = self.amodel(states, training=True)
            log_action_pred = tf.math.log(
                action_pred + keras.backend.epsilon()
            )
            log_probs = tf.reduce_sum(
                actions * log_action_pred, axis=1
            )
            if self.lambda_rate == 0:
                advantages = (drewards - value_pred)
            loss = -tf.reduce_mean(advantages * log_probs)
            entropy = tf.reduce_sum(
                action_pred * log_action_pred, axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric(loss)

    def _train(self, states, drewards, advantages, actions,
               epochs, batch_size, entropy_coef, verbose=True):
        """Performs multiple gradient steps of all the data.

        Args:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted rewards
                      for the actions performed in the environment
            advantages: A numpy array, which if valid (lambda_rate > 0)
                        contains advantages for the actions performed
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

        Returns:
            A float, which is the mean critic loss of the batches
        """
        length = states.shape[0]
        float_type = keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             drewards.astype(float_type),
             advantages.astype(float_type),
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
            actor_loss_results = self.metric.result()
            critic_loss_results = self.metric_c.result()
            self.metric.reset_states()
            self.metric_c.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'actor_loss: {actor_loss_results} - '
                      f'critic_loss: {critic_loss_results}')
        return critic_loss_results

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

            if self.lambda_rate == 0:
                advantages_arr = np.empty(length)
            else:
                # cmodel predict on batches if large?
                values = tf.squeeze(
                    self.cmodel(self.states.array())
                ).numpy()
                advantages = np.empty(len(self.rewards))
                for ndx in reversed(range(len(self.rewards))):
                    delta = self.rewards[ndx] - values[ndx]
                    if not self.terminals[ndx]:
                        delta += self.discounted_rate * values[ndx + 1]
                    if self.terminals[ndx]:
                        advantage = 0
                    advantage = (delta + self.discounted_rate *
                                 self.lambda_rate * advantage)
                    advantages[ndx] = advantage

            arrays, indexes = self.states.create_shuffled_subset(
                [self.states, self.drewards, self.actions], length
            )

            if self.lambda_rate == 0:
                arrays = [arrays[0], arrays[1], advantages_arr, arrays[2]]
                std = arrays[1].std()
                if std == 0:
                    return False
                arrays[1] = (arrays[1] - arrays[1].mean()) / std
            else:
                arrays = [arrays[0], arrays[1], advantages[indexes], arrays[2]]
                std = arrays[2].std()
                if std == 0:
                    return False
                arrays[2] = (arrays[2] - arrays[2].mean()) / std

            self._train(*arrays, epochs, batch_size,
                        entropy_coef, verbose=verbose)

    def load(self, path, load_model=True, load_data=True, custom_objects=None):
        """Loads a save from a folder.

        Args:
            path: A string, which is the path to a folder to load
            load_model: A boolean, which determines if the model
                        architectures and weights
                        should be loaded
            load_data: A boolean, which determines if the memory
                       from a folder should be loaded
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model

        Returns:
            A string of note.txt
        """
        note = PGAgent.load(
            self, path, load_model=load_model, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'cmodel.json'), 'r') as file:
                self.cmodel = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.cmodel.load_weights(os.path.join(path, 'cweights.h5'))
        return note

    def save(self, path, save_model=True,
             save_data=True, note='A2CAgent Save'):
        """Saves a note, models, weights, and memory to a new folder.

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
        path = PGAgent.save(self, path, save_model=save_model,
                            save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'cmodel.json'), 'w') as file:
                file.write(self.cmodel.to_json())
            self.cmodel.save_weights(os.path.join(path, 'cweights.h5'))
        return path


class PPOAgent(A2CAgent):
    """This class (Proximal Policy Optimization) is like the A2CAgent
       but attempts to avoid taking large gradient steps that would
       collapse the performacne of the agent. (this is the clip variant)
    """

    def __init__(self, amodel, cmodel, discounted_rate,
                 lambda_rate=0, clip_ratio=.2,
                 create_memory=lambda shape, dtype: Memory()):
        """Initalizes the Policy Gradient Agent.

        Args:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            cmodel: A keras model, which takes the state as input and outputs
                    the value of that state
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            lambda_rate: A float within 0.0-1.0, which if nonzero will enable
                         generalized advantage estimation
            clip_ratio: A float, which is the ratio to clip the differences
                        between new and old action probabilities
            create_memory: A function, which returns a Memory instance
        """
        A2CAgent.__init__(self, amodel, cmodel, discounted_rate,
                          lambda_rate=lambda_rate,
                          create_memory=create_memory)
        self.clip_ratio = clip_ratio
        self.old_probs = create_memory((None,),
                                       keras.backend.floatx())
        self.memory['old_probs'] = self.old_probs
        self.prob = None
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
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
                      agent is training

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
                              training=False)[0].numpy()
        action = np.random.choice(np.arange(self.action_size),
                                  p=actions)
        self.prob = actions[action]
        return action

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
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
                      (discarded)
        """
        A2CAgent.add_memory(self, state, action, new_state, reward, terminal)
        if self.prob is None:
            # actions = self.amodel(np.expand_dims(state, axis=0),
            #                       training=False)[0].numpy()
            # prob = actions[action]
            # self.old_probs.add(prob)

            # Assuming a uniform distribution
            self.old_probs.add(1 / self.action_size)
        else:
            self.old_probs.add(self.prob)
            self.prob = None

    def _train_step(self, states, drewards, advantages, actions,
                    old_probs, entropy_coef):
        """Performs one gradient step with a batch of data.

        Args:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            advantages: A tensor, which if valid (lambda_rate > 0) contains
                        advantages for the actions performed
            actions: A tensor that contains onehot encodings of
                     the action performed
            old_probs: A tensor of the old probs
            entropy_coef: A tensor constant float, which is the
                          coefficent of entropy to add to the
                          actor loss

        Returns:
            A tensor of the new probs
        """
        with tf.GradientTape() as tape:
            value_pred = tf.squeeze(self.cmodel(states, training=True))
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.compiled_loss._losses[0].fn(drewards,
                                                           value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            action_pred = self.amodel(states, training=True)
            probs = tf.reduce_sum(actions * action_pred, axis=1)
            ratio = probs / (old_probs + keras.backend.epsilon())
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio,
                                             1.0 + self.clip_ratio)
            if self.lambda_rate == 0:
                advantages = drewards - value_pred
            loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            entropy = tf.reduce_sum(
                action_pred *
                tf.math.log(action_pred + keras.backend.epsilon()),
                axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )

        self.metric(loss)
        return probs

    def _train(self, states, drewards, advantages, actions,
               old_probs, epochs, batch_size, entropy_coef,
               verbose=True):
        """Performs multiple gradient steps of all the data.

        Args:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted reward
                      for the action performed in the environment
            advantages: A numpy array, which if valid (lambda_rate > 0)
                        contains advantages for the actions performed
            actions: A numpy array that contains onehot encodings of
                     the action performed
            old_probs: A numpy array of the old probs
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if information should
                     be printed to the screen

        Returns:
            A tuple of a float (mean critic loss of the batches) and
                a numpy ndarray of probs
        """
        length = states.shape[0]
        float_type = keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             drewards.astype(float_type),
             advantages.astype(float_type),
             actions.astype(float_type),
             old_probs.astype(float_type))
        ).batch(batch_size)
        entropy_coef = tf.constant(entropy_coef,
                                   dtype=float_type)
        new_probs = []
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                if epoch == epochs:
                    new_probs.append(self._tf_train_step(*batch,
                                                         entropy_coef))
                else:
                    self._tf_train_step(*batch, entropy_coef)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            actor_loss_results = self.metric.result()
            critic_loss_results = self.metric_c.result()
            self.metric.reset_states()
            self.metric_c.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'actor_loss: {actor_loss_results} - '
                      f'critic_loss: {critic_loss_results}')
        return critic_loss_results, np.hstack(new_probs)

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

            if self.lambda_rate == 0:
                advantages_arr = np.empty(length)
            else:
                # cmodel predict on batches if large?
                #values = tf.squeeze(
                #    self.cmodel(self.states.array())
                #).numpy()
                values = np.squeeze(
                    self.cmodel.predict(self.states.array(), batch_size=1024)
                )
                advantages = np.empty(len(self.rewards))
                for ndx in reversed(range(len(self.rewards))):
                    delta = self.rewards[ndx] - values[ndx]
                    if not self.terminals[ndx]:
                        delta += self.discounted_rate * values[ndx + 1]
                    if self.terminals[ndx]:
                        advantage = 0
                    advantage = (delta + self.discounted_rate *
                                 self.lambda_rate * advantage)
                    advantages[ndx] = advantage

            arrays, indexes = self.states.create_shuffled_subset(
                [self.states, self.drewards, self.actions, self.old_probs],
                length
            )

            if self.lambda_rate == 0:
                arrays = [arrays[0], arrays[1], advantages_arr,
                          arrays[2], arrays[3]]
                std = arrays[1].std()
                if std == 0:
                    return False
                arrays[1] = (arrays[1] - arrays[1].mean()) / std
            else:
                arrays = [arrays[0], arrays[1], advantages[indexes],
                          arrays[2], arrays[3]]
                std = arrays[2].std()
                if std == 0:
                    return False
                arrays[2] = (arrays[2] - arrays[2].mean()) / std

            loss, new_probs = self._train(*arrays, epochs, batch_size,
                                          entropy_coef, verbose=verbose)
            for ndx in range(length):
                self.old_probs[indexes[ndx]] = new_probs[ndx]

    def save(self, path, save_model=True,
             save_data=True, note='PPOAgent Save'):
        """Saves a note, models, weights, and memory to a new folder.

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
        path = A2CAgent.save(self, path, save_model=save_model,
                             save_data=save_data, note=note)
        return path


class TD3Agent(DDPGAgent):
    """This class (Twin Delayed DDPG Agent) attempts to mitigate
       the problems that a DDPGAgent faces through clipping Q targets
       between two Q models, delaying policy updates, and adding noise
       to target actions.
    """

    def __init__(self, policy, amodel, cmodel, discounted_rate,
                 create_memory=lambda shape, dtype: Memory()):
        """Initalizes the DDPG Agent.

        Args:
            policy: A noise policy instance, which used for exploring
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            cmodel: A keras model, which takes the state and a action as input
                     and outputs two seperate Q Values (a judgement)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            create_memory: A function, which returns a Memory instance
        """
        DDPGAgent.__init__(self, policy, amodel, cmodel, discounted_rate,
                           create_memory=create_memory,
                           enable_target=True)
        coutput_shape = self.cmodel.output_shape
        if not isinstance(coutput_shape, list) or len(coutput_shape) != 2:
            raise ValueError('cmodel should have two outputs')
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
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(), dtype=tf.int32))
        )
        self.gradient_step_count = 0

    def set_playing_data(self, training=False, memorizing=False,
                         learns_in_episode=False, batch_size=None,
                         mini_batch=0, epochs=1, repeat=1,
                         target_update_interval=1, tau=1.0,
                         policy_noise_std=.2, policy_noise_clip=.5,
                         actor_update_infreq=2,
                         verbose=True):
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
            policy_noise_std: A float, which is the standard deviation of the
                              noise to add to the target actions for gradient
                              steps
            policy_noise_clip: A float, which is the min and max value of
                               the normal noise added to target actions
                               for gradient steps
            actor_update_infreq: An integer, which is the infrequency that
                               the actor is updated compared to the critic
            verbose: A boolean, which determines if training
                     should be verbose (print information to the screen)
        """
        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'target_update_interval': target_update_interval,
                           'tau': tau,
                           'policy_noise_std': policy_noise_std,
                           'policy_noise_clip': policy_noise_clip,
                           'actor_update_infreq': actor_update_infreq,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        learns_in_episode, learning_params)

    def _train_step(self, states, next_states, actions, terminals,
                    rewards, policy_noise_std, policy_noise_clip,
                    actor_update):
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
            policy_noise_std: A tensor constant float, which is the
                              standard deviation of the noise to add
                              to the target actions for gradient steps
            policy_noise_clip: A tensor constant float, which is the
                               min and max value of the normal noise
                               added to target actions for gradient steps
            actor_update: A tensor constant integer, which is determines
                          if the actor should update
        """
        next_actions = self.target_amodel(next_states, training=False)
        noise = tf.random.normal(tf.shape(next_actions),
                                 stddev=policy_noise_std)
        noise = tf.clip_by_value(noise, -policy_noise_clip, policy_noise_clip)
        next_actions = tf.clip_by_value(next_actions + noise,
                                        *self.policy.action_bounds)
        next_qvalues1, next_qvalues2 = self.target_cmodel(
            [next_states, next_actions], training=False
        )
        next_qvalues = tf.squeeze(tf.minimum(next_qvalues1, next_qvalues2))
        qvalues_true = (rewards +
                        self.discounted_rate * next_qvalues * terminals)
        # Critic
        with tf.GradientTape() as tape:
            qvalues_pred1, qvalues_pred2 = self.cmodel(
                [states, actions], training=True
            )
            qvalues_pred1 = tf.squeeze(qvalues_pred1)
            qvalues_pred2 = tf.squeeze(qvalues_pred2)
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss1 = self.cmodel.compiled_loss._losses[0].fn(
                qvalues_true, qvalues_pred1
            )
            loss2 = self.cmodel.compiled_loss._losses[0].fn(
                qvalues_true, qvalues_pred2
            )
            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2) + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        # Actor
        if actor_update == 0:
            with tf.GradientTape() as tape:
                action_preds = self.amodel(states, training=True)
                loss = -tf.reduce_mean(
                    self.cmodel([states, action_preds], training=False)
                )
            grads = tape.gradient(loss, self.amodel.trainable_variables)
            self.amodel.optimizer.apply_gradients(
                zip(grads, self.amodel.trainable_variables)
            )
            self.metric_a(loss)

    def _train(self, states, next_states, actions, terminals, rewards,
               epochs, batch_size, policy_noise_std, policy_noise_clip,
               actor_update_infreq, verbose=True):
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
            policy_noise_std: A float, which is the standard deviation of the
                              noise to add to the target actions for gradient
                              steps
            policy_noise_clip: A float, which is the min and max value of
                               the normal noise added to target actions
                               for gradient steps
            actor_update_infreq: An integer, which is the infrequency that
                               the actor is updated compared to the critic
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
        policy_noise_std = tf.constant(policy_noise_std,
                                       dtype=float_type)
        policy_noise_clip = tf.constant(policy_noise_clip,
                                        dtype=float_type)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            self.gradient_step_count += 1
            count = 0
            actor_update = self.gradient_step_count % actor_update_infreq
            actor_update = tf.constant(actor_update, dtype=tf.int32)
            for batch in batches:
                self._tf_train_step(*batch, policy_noise_std,
                                    policy_noise_clip,
                                    actor_update)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            critic_loss_results = self.metric_c.result()
            actor_loss_results = self.metric_a.result()
            self.metric_c.reset_states()
            if actor_update == 1:
                self.metric_a.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'critic_loss: {critic_loss_results} - '
                      f'actor_loss: {actor_loss_results}')
        return critic_loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1, target_update_interval=1,
              tau=1.0, policy_noise_std=.2, policy_noise_clip=.5,
              actor_update_infreq=2, verbose=True):
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
            policy_noise_std: A float, which is the standard deviation of the
                              noise to add to the target actions for gradient
                              steps
            policy_noise_clip: A float, which is the min and max value of
                               the normal noise added to target actions
                               for gradient steps
            actor_update_infreq: An integer, which is the infrequency that
                                 the actor is updated compared to the critic
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

            self._train(*arrays, epochs, batch_size,
                        policy_noise_std, policy_noise_clip,
                        actor_update_infreq, verbose=verbose)

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

    def save(self, path, save_model=True, save_data=True,
             note='T3DAgent Save'):
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
        return DDPGAgent.save(self, path, save_model=save_model,
                              save_data=save_data, note=note)


class Continuous:
    """This interface is used for the continuous action space
       variants of algorithms.
    """

    @staticmethod
    def scale(lower_bound, upper_bound, name=None):
        def _scale(x):
            x = tf.multiply(x, upper_bound - lower_bound)
            x = x + upper_bound + lower_bound
            return tf.multiply(.5, x)
        return _scale

    @staticmethod
    def sample(name=None):
        def _sample(x):
            mean, std = x
            eps = tf.random.normal(tf.shape(mean))
            action = eps * std + mean
            return action
        return _sample

    @staticmethod
    def clip(lower_bound, upper_bound, name=None):
        def _clip(x):
            return tf.clip_by_value(x, lower_bound, upper_bound)
        return _clip


class PGCAgent(PGAgent, Continuous):
    """This class is a continuous action space variant of the PGAgent.
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
        if len(amodel.output_shape) != 3:
            raise ValueError('The model must have three outputs: '
                             'mean, stddev, and actions')
        PGAgent.__init__(self, amodel, discounted_rate,
                         create_memory=create_memory)

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value or list of values, which is the action
                    the agent took
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
        self.actions.add(action)
        self.episode_rewards.append(reward)

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
            locs, scales, _ = self.amodel(states, training=True)
            normal = tfp.distributions.MultivariateNormalDiag(locs, tf.ones_like(scales))
            probs = normal.prob((actions - locs) / scales + locs)
            log_probs = tf.math.log(probs + keras.backend.epsilon())
            loss = -tf.reduce_mean(drewards * log_probs)
            entropy = tf.reduce_mean(probs * log_probs)
            loss += entropy * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric(loss)


class A2CCAgent(A2CAgent, Continuous):
    """This class is a continuous action space variant of the A2CAgent.
    """

    def __init__(self, amodel, cmodel, discounted_rate,
                 lambda_rate=0, create_memory=lambda shape, dtype: Memory()):
        """Initalizes the Policy Gradient Agent.

        Args:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            cmodel: A keras model, which takes the state as input and outputs
                    the value of that state
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            lambda_rate: A float within 0.0-1.0, which if nonzero will enable
                         generalized advantage estimation
            create_memory: A function, which returns a Memory instance
        """
        if len(amodel.output_shape) != 3:
            raise ValueError('The model must have three outputs: '
                             'mean, stddev, and actions')
        A2CAgent.__init__(self, amodel, cmodel, discounted_rate,
                          lambda_rate=lambda_rate,
                          create_memory=create_memory)

    def add_memory(self, state, action, new_state, reward, terminal):
        """Adds information from one step in the environment to the agent.

        Args:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            action: A value or list of values, which is the action
                    the agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        self.states.add(np.array(state))
        self.actions.add(action)
        self.episode_rewards.append(reward)
        if self.lambda_rate > 0:
            self.terminals.add(terminal)
            self.rewards.add(reward)

    def _train_step(self, states, drewards, advantages,
                    actions, entropy_coef):
        """Performs one gradient step with a batch of data.

        Args:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            advantages: A tensor, which if valid (lambda_rate > 0) contains
                        advantages for the actions performed
            actions: A tensor that contains onehot encodings of
                     the action performed
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
        """
        with tf.GradientTape() as tape:
            value_pred = tf.squeeze(self.cmodel(states, training=True))
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.compiled_loss._losses[0].fn(drewards,
                                                           value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            locs, scales, _ = self.amodel(states, training=True)
            normal = tfp.distributions.MultivariateNormalDiag(locs, tf.ones_like(scales))
            probs = normal.prob((actions - locs) / scales + locs)
            log_probs = tf.math.log(probs + keras.backend.epsilon())
            if self.lambda_rate == 0:
                advantages = (drewards - value_pred)
            loss = -tf.reduce_mean(advantages * log_probs)
            entropy = tf.reduce_mean(probs * log_probs)
            loss += entropy * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric(loss)
