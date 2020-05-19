"""
Author: Travis Hammond
Version: 5_19_2020
"""


import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

try:
    from paiutils.reinforcement import (
        Memory, PlayingData, DQNAgent,
        PGAgent, DDPGAgent, NoisePolicy,
    )
except ImportError:
    from reinforcement import (
        Memory, PlayingData, DQNAgent,
        PGAgent, DDPGAgent, NoisePolicy,
    )


class DQNPGAgent(DQNAgent, PGAgent):
    """This class is an Agent that uses a Deep Q Network by default to
       select actions, but can be easily be changed to a policy gradient
       based network to predict actions.
    """

    def __init__(self, policy, qmodel, amodel, discounted_rate,
                 create_memory=lambda: Memory(),
                 enable_target=True, enable_double=False,
                 enable_PER=False):
        """Initalizes the Deep Q Network and Policy Gradient Agent.
        params:
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
            enable_PER: A boolean, which determines if prioritized experience
                        replay should be used
                        (The implementation for this is not the normal tree
                         implementation, and only weights the probabilily of
                         being choosen)
        """
        DQNAgent.__init__(self, policy, qmodel, discounted_rate,
                          create_memory=create_memory,
                          enable_target=enable_target,
                          enable_double=enable_double,
                          enable_PER=enable_PER)
        self.amodel = amodel
        self.uses_dqn_method = True
        self.temp_rewards = create_memory()
        self.drewards = create_memory()
        self.pg_metric = tf.keras.metrics.Mean(name='pg_loss')
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
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()))
        )

    def use_dqn(self):
        self.uses_dqn_method = True

    def use_pg(self):
        self.uses_dqn_method = False

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
        if self.uses_dqn_method:
            def _select_action():
                qvalues = self.qmodel(np.expand_dims(state, axis=0),
                                      training=False)[0].numpy()
                return qvalues
            return self.policy.select_action(_select_action,
                                             training=training)
        else:
            actions = np.abs(self.amodel(np.expand_dims(state, axis=0),
                                         training=False)[0].numpy())
            return np.random.choice(np.arange(self.action_shape[0]),
                                    p=actions)

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
        DQNAgent.add_memory(self, state, action, new_state, reward, terminal)
        self.temp_rewards.add(reward)

    def forget(self):
        """Forgets or clears all memory."""
        DQNAgent.forget(self)
        self.temp_rewards.reset()
        self.drewards.reset()

    def end_episode(self):
        """Ends the episode, and creates drewards based
           on the episodes rewards.
        """
        if len(self.temp_rewards) > 0:
            dreward = 0
            dreward_list = []
            # hacky, assuming memory works with reversed
            for reward in reversed(self.temp_rewards.buffer):
                dreward *= self.discounted_rate
                dreward += reward
                dreward_list.append(dreward)
            self.temp_rewards.reset()
            for dreward in reversed(dreward_list):
                self.drewards.add(dreward)

    def _train_step(self, states, next_states, action_onehots,
                    terminals, rewards, drewards):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            next_states: A tensor that contains the states of
                         the environment after an action was performed
            action_onehots: A tensor that contains onehot encodings of
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
            y_true = (y_pred * (1 - action_onehots) +
                      qvalues[:, tf.newaxis] * action_onehots)
            loss = self.qmodel.loss_functions[0](y_true, y_pred) + reg_loss
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
                action_onehots *
                tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=1
            )
            loss = -tf.reduce_mean(drewards * log_probs)
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.pg_metric(loss)

        return abs_loss

    def _train(self, states, next_states, action_onehots, terminals,
               rewards, drewards, epochs, batch_size, verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            next_states: A numpy array that contains the states of
                         the environment after an action was performed
            action_onehots: A numpy array that contains onehot encodings of
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
        return: A list of floats, which are the absolute losses for all
                the data
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             next_states.astype(float_type),
             action_onehots.astype(float_type),
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
                                    target qmodel weights are updated
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
            if self.PER_losses is None:
                indexes = np.random.choice(np.arange(len(self.states)),
                                           size=length, replace=False)
            else:
                PER_losses_arr = self.PER_losses.array()
                self.max_loss = PER_losses_arr.max()
                PER_losses_arr = PER_losses_arr / PER_losses_arr.sum()
                indexes = np.random.choice(np.arange(len(self.states)),
                                           size=length, replace=False,
                                           p=PER_losses_arr)
            if length >= 10000:  # depends on cpu and other factors
                next_states_arr = self.next_states.array()[indexes]
                states_arr = self.states.array()[indexes]
                action_onehots = self.action_identity[
                    self.actions.array()[indexes]
                ]
                rewards_arr = self.rewards.array()[indexes]
                drewards_arr = self.drewards.array()[indexes]
                terminals_arr = self.terminals.array()[indexes]
            else:
                next_states_arr = np.empty((length, *self.states[0].shape))
                states_arr = np.empty((length, *self.states[0].shape))
                action_onehots = np.empty((length, self.action_shape[0]))
                rewards_arr = np.empty(length)
                drewards_arr = np.empty(length)
                terminals_arr = np.empty(length)
                for ndx, rndx in enumerate(indexes):
                    states_arr[ndx] = self.states[rndx]
                    next_states_arr[ndx] = self.next_states[rndx]
                    action_onehots[ndx] = self.action_identity[
                        self.actions[rndx]
                    ]
                    rewards_arr[ndx] = self.rewards[rndx]
                    drewards_arr[ndx] = self.drewards[rndx]
                    terminals_arr[ndx] = self.terminals[rndx]

            std = drewards_arr.std()
            if std == 0:
                return False
            drewards_arr = (drewards_arr - drewards_arr.mean()) / std

            losses = self._train(states_arr, next_states_arr, action_onehots,
                                 terminals_arr, rewards_arr, drewards_arr,
                                 epochs, batch_size, verbose=verbose)
            if self.PER_losses is not None:
                for ndx, loss in zip(indexes, losses):
                    self.PER_losses[ndx] = loss
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
        DQNAgent.load(self, path, load_model=load_model, load_data=False)
        if load_model:
            with open(os.path.join(path, 'amodel.json'), 'r') as file:
                self.amodel = model_from_json(file.read())
            self.amodel.load_weights(os.path.join(path, 'aweights.h5'))
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
                for dreward in file['drewards']:
                    self.drewards.add(dreward)
                for terminal in file['terminals']:
                    self.terminals.add(terminal)
                if self.PER_losses is not None:
                    if 'PER_losses' in file:
                        for loss in file['PER_losses']:
                            self.PER_losses.add(loss)
                    else:
                        for _ in range(len(self.states)):
                            self.PER_losses.add(self.max_loss)

    def save(self, path, save_model=True,
             save_data=True, note='DQNPGAgent Save'):
        """Saves a note, model weights, and memory to a new folder.
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
        path = DQNAgent.save(self, path, save_model=save_model,
                             save_data=False, note=note)
        if save_model:
            with open(os.path.join(path, 'amodel.json'), 'w') as file:
                file.write(self.amodel.to_json())
            self.amodel.save_weights(os.path.join(path, 'aweights.h5'))
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset('next_states',
                                    data=self.next_states.array())
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('rewards', data=self.rewards.array())
                file.create_dataset('drewards', data=self.drewards.array())
                file.create_dataset('terminals', data=self.terminals.array())
                if self.PER_losses is not None:
                    file.create_dataset(
                        'PER_losses', data=self.PER_losses.array()
                    )
        return path


class A2CAgent(PGAgent):
    """This class (Advantage Actor-Critic) is like the PGAgent, but it also
       has a critic network which is used to estimate the value function
       in order to train the Actor network on the advantages instead of
       the discounted rewards.
    """

    def __init__(self, amodel, cmodel, discounted_rate,
                 lambda_rate=0, create_memory=lambda: Memory()):
        """Initalizes the Policy Gradient Agent.
        params:
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
                         create_memory=create_memory,
                         policy=None)
        self.cmodel = cmodel
        self.lambda_rate = lambda_rate
        self.temp_rewards = create_memory()
        self.terminals = create_memory()
        self.metric_c = tf.keras.metrics.Mean(name='critic_loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=tf.keras.backend.floatx()))
        )

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
        self.actions.add(action)
        self.temp_rewards.add(reward)
        if self.lambda_rate > 0:
            self.terminals.add(terminal)
            self.rewards.add(reward)

    def forget(self):
        """Forgets or clears all memory."""
        PGAgent.forget(self)
        self.temp_rewards.reset()
        self.terminals.reset()

    def end_episode(self):
        """Ends the episode, and creates drewards based
           on the episodes rewards.
        """
        if len(self.temp_rewards) > 0:
            dreward = 0
            dreward_list = []
            # hacky, assuming memory works with reversed
            for reward in reversed(self.temp_rewards.buffer):
                dreward *= self.discounted_rate
                dreward += reward
                dreward_list.append(dreward)
            self.temp_rewards.reset()
            for dreward in reversed(dreward_list):
                self.drewards.add(dreward)
            if self.lambda_rate > 0:
                self.terminals[-1] = True

    def _train_step(self, states, drewards, advantages,
                    action_onehots, entropy_coef):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            advantages: A tensor, which if valid (lambda_rate > 0) contains
                        advantages for the actions performed
            action_onehots: A tensor that contains onehot encodings of
                            the action performed
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
        """
        with tf.GradientTape() as tape:
            value_pred = tf.reshape(self.cmodel(states, training=True), [-1])
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.loss_functions[0](drewards, value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            action_pred = self.amodel(states, training=True)
            # log_softmax may be mathematically correct, but in practice
            # seems to give worse results
            log_probs = tf.reduce_sum(
                action_onehots *
                tf.math.log(action_pred + tf.keras.backend.epsilon()), axis=1
            )
            if self.lambda_rate == 0:
                advantages = (drewards - value_pred)
            loss = -tf.reduce_mean(advantages * log_probs)
            entropy = tf.reduce_sum(
                action_pred *
                tf.math.log(action_pred + tf.keras.backend.epsilon()),
                axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )
        self.metric(loss)

    def _train(self, states, drewards, advantages, action_onehots,
               epochs, batch_size, entropy_coef, verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted rewards
                      for the actions performed in the environment
            advantages: A numpy array, which if valid (lambda_rate > 0)
                        contains advantages for the actions performed
            action_onehots: A numpy array that contains onehot encodings of
                            the action performed
            epochs: An integer, which is the number of complete gradient
                    steps to perform
            batch_size: An integer, which is the size of the batch for
                        each partial gradient step
            entropy_coef: A float, which is the coefficent of entropy to add
                          to the actor loss
            verbose: A boolean, which determines if information should
                     be printed to the screen
        return: A float, which is the mean critic loss of the batches
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             drewards.astype(float_type),
             advantages.astype(float_type),
             action_onehots.astype(float_type))
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

            if self.lambda_rate == 0:
                advantages_arr = np.empty(length)
            else:
                # cmodel predict on batches if large?
                values = tf.reshape(
                    self.cmodel(self.states.array()), [-1]
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

            indexes = np.random.choice(np.arange(len(self.states)),
                                       size=length, replace=False)
            if length >= 20000:  # depends on cpu and other factors
                states_arr = self.states.array()[indexes]
                action_onehots = self.action_identity[
                    self.actions.array()[indexes]
                ]
                drewards_arr = self.drewards.array()[indexes]
                if self.lambda_rate > 0:
                    advantages_arr = advantages[indexes]
            else:
                states_arr = np.empty((length, *self.states[0].shape))
                action_onehots = np.empty((length, self.action_shape[0]))
                drewards_arr = np.empty(length)
                if self.lambda_rate > 0:
                    advantages_arr = np.empty(length)
                for ndx in range(length):
                    states_arr[ndx] = self.states[indexes[ndx]]
                    action_onehots[ndx] = self.action_identity[
                        self.actions[indexes[ndx]]
                    ]
                    drewards_arr[ndx] = self.drewards[indexes[ndx]]
                    if self.lambda_rate > 0:
                        advantages_arr[ndx] = advantages[indexes[ndx]]

            if self.lambda_rate == 0:
                std = drewards_arr.std()
                if std == 0:
                    return False
                drewards_arr = (drewards_arr - drewards_arr.mean()) / std
            else:
                std = advantages_arr.std()
                if std == 0:
                    return False
                advantages_arr = (advantages_arr - advantages_arr.mean()) / std

            self._train(states_arr, drewards_arr, advantages_arr,
                        action_onehots, epochs, batch_size, entropy_coef,
                        verbose=verbose)

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
        PGAgent.load(self, path, load_model=load_model, load_data=False)
        if load_model:
            with open(os.path.join(path, 'cmodel.json'), 'r') as file:
                self.cmodel = model_from_json(file.read())
            self.cmodel.load_weights(os.path.join(path, 'cweights.h5'))
        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for state in file['states']:
                    self.states.add(state)
                for action in file['actions']:
                    self.actions.add(action)
                for dreward in file['drewards']:
                    self.drewards.add(dreward)
                if self.lambda_rate > 0:
                    for reward in file['rewards']:
                        self.rewards.add(reward)
                    for terminal in file['terminals']:
                        self.terminals.add(terminal)

    def save(self, path, save_model=True,
             save_data=True, note='A2CAgent Save'):
        """Saves a note, models, weights, and memory to a new folder.
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
        path = PGAgent.save(self, path, save_model=save_model,
                            save_data=False, note=note)
        if save_model:
            with open(os.path.join(path, 'cmodel.json'), 'w') as file:
                file.write(self.cmodel.to_json())
            self.cmodel.save_weights(os.path.join(path, 'cweights.h5'))
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('drewards', data=self.drewards.array())
                if self.lambda_rate > 0:
                    file.create_dataset('rewards', data=self.rewards.array())
                    file.create_dataset('terminals',
                                        data=self.terminals.array())
        return path


class PPOAgent(A2CAgent):
    """This class (Proximal Policy Optimization) is like the A2CAgent
       but attempts to avoid taking large gradient steps that would
       collapse the performacne of the agent. (this is the clip variant)
    """

    def __init__(self, amodel, cmodel, discounted_rate,
                 lambda_rate=0, clip_ratio=.2,
                 create_memory=lambda: Memory()):
        """Initalizes the Policy Gradient Agent.
        params:
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
        print('WARNING: This implementation may be incorrect.')
        A2CAgent.__init__(self, amodel, cmodel, discounted_rate,
                          lambda_rate=lambda_rate,
                          create_memory=create_memory)
        self.clip_ratio = clip_ratio
        self.old_probs = create_memory()
        self.prob = None
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.amodel.input_shape,
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
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
        actions = self.amodel(np.expand_dims(state, axis=0),
                              training=False)[0].numpy()
        action = np.random.choice(np.arange(self.action_shape[0]),
                                  p=actions)
        self.prob = actions[action]
        return action

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
        A2CAgent.add_memory(self, state, action, new_state, reward, terminal)
        if self.prob is None:
            actions = self.amodel(np.expand_dims(state, axis=0),
                                  training=False)[0].numpy()
            prob = actions[action]
            self.old_probs.add(prob)

            # Assuming a uniform distribution
            # self.old_probs.add(1 / self.action_shape[0])
        else:
            self.old_probs.add(self.prob)
            self.prob = None

    def forget(self):
        """Forgets or clears all memory."""
        A2CAgent.forget(self)
        self.old_probs.forget()

    def _train_step(self, states, drewards, advantages, action_onehots,
                    old_probs, entropy_coef):
        """Performs one gradient step with a batch of data.
        params:
            states: A tensor that contains environment states
            drewards: A tensor that contains the discounted reward
                      for the action performed in the environment
            advantages: A tensor, which if valid (lambda_rate > 0) contains
                        advantages for the actions performed
            action_onehots: A tensor that contains onehot encodings of
                            the action performed
            old_probs: A tensor of the old probs
            entropy_coef: A tensor constant float, which is the
                          coefficent of entropy to add to the
                          actor loss
        return: A tensor of the new probs
        """
        with tf.GradientTape() as tape:
            value_pred = tf.reshape(self.cmodel(states, training=True), [-1])
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss = self.cmodel.loss_functions[0](drewards, value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.cmodel.trainable_variables)
        self.cmodel.optimizer.apply_gradients(
            zip(grads, self.cmodel.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            action_pred = self.amodel(states, training=True)
            probs = tf.reduce_sum(action_onehots * action_pred, axis=1)
            ratio = probs / (old_probs + tf.keras.backend.epsilon())
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio,
                                             1.0 + self.clip_ratio)
            if self.lambda_rate == 0:
                advantages = drewards - value_pred
            loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            entropy = tf.reduce_sum(
                action_pred *
                tf.math.log(action_pred + tf.keras.backend.epsilon()),
                axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.amodel.trainable_variables)
        self.amodel.optimizer.apply_gradients(
            zip(grads, self.amodel.trainable_variables)
        )

        self.metric(loss)
        return probs

    def _train(self, states, drewards, advantages, action_onehots,
               old_probs, epochs, batch_size, entropy_coef,
               verbose=True):
        """Performs multiple gradient steps of all the data.
        params:
            states: A numpy array that contains environment states
            drewards: A numpy array that contains the discounted reward
                      for the action performed in the environment
            advantages: A numpy array, which if valid (lambda_rate > 0)
                        contains advantages for the actions performed
            action_onehots: A numpy array that contains onehot encodings of
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
        return: A tuple of a float (mean critic loss of the batches) and
                a numpy ndarray of probs
        """
        length = states.shape[0]
        float_type = tf.keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             drewards.astype(float_type),
             advantages.astype(float_type),
             action_onehots.astype(float_type),
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

            if self.lambda_rate == 0:
                advantages_arr = np.empty(length)
            else:
                # cmodel predict on batches if large?
                values = tf.reshape(
                    self.cmodel(self.states.array()), [-1]
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

            indexes = np.random.choice(np.arange(len(self.states)),
                                       size=length, replace=False)
            if length >= 15000:  # depends on cpu and other factors
                states_arr = self.states.array()[indexes]
                action_onehots = self.action_identity[
                    self.actions.array()[indexes]
                ]
                drewards_arr = self.drewards.array()[indexes]
                old_probs_arr = self.old_probs.array()[indexes]
                if self.lambda_rate > 0:
                    advantages_arr = advantages[indexes]
            else:
                states_arr = np.empty((length, *self.states[0].shape))
                action_onehots = np.empty((length, self.action_shape[0]))
                drewards_arr = np.empty(length)
                if self.lambda_rate > 0:
                    advantages_arr = np.empty(length)
                old_probs_arr = np.empty(length)
                for ndx in range(length):
                    states_arr[ndx] = self.states[indexes[ndx]]
                    action_onehots[ndx] = self.action_identity[
                        self.actions[indexes[ndx]]
                    ]
                    drewards_arr[ndx] = self.drewards[indexes[ndx]]
                    if self.lambda_rate > 0:
                        advantages_arr[ndx] = advantages[indexes[ndx]]
                    old_probs_arr[ndx] = self.old_probs[indexes[ndx]]

            if self.lambda_rate == 0:
                std = drewards_arr.std()
                if std == 0:
                    return False
                drewards_arr = (drewards_arr - drewards_arr.mean()) / std
            else:
                std = advantages_arr.std()
                if std == 0:
                    return False
                advantages_arr = (advantages_arr - advantages_arr.mean()) / std

            loss, new_probs = self._train(
                states_arr, drewards_arr, advantages_arr, action_onehots,
                old_probs_arr, epochs, batch_size, entropy_coef,
                verbose=verbose
            )
            for ndx in range(length):
                self.old_probs[indexes[ndx]] = new_probs[ndx]

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
        A2CAgent.load(self, path, load_model=load_model, load_data=False)
        if load_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'r') as file:
                for state in file['states']:
                    self.states.add(state)
                for action in file['actions']:
                    self.actions.add(action)
                for dreward in file['drewards']:
                    self.drewards.add(dreward)
                for old_prob in file['old_probs']:
                    self.old_probs.add(old_prob)
                if self.lambda_rate > 0:
                    for reward in file['rewards']:
                        self.rewards.add(reward)
                    for terminal in file['terminals']:
                        self.terminals.add(terminal)

    def save(self, path, save_model=True,
             save_data=True, note='PPOAgent Save'):
        """Saves a note, models, weights, and memory to a new folder.
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
        path = A2CAgent.save(self, path, save_model=save_model,
                             save_data=False, note=note)
        if save_data:
            with h5py.File(os.path.join(path, 'data.h5'), 'w') as file:
                file.create_dataset('states', data=self.states.array())
                file.create_dataset('actions', data=self.actions.array())
                file.create_dataset('drewards', data=self.drewards.array())
                file.create_dataset('old_probs',
                                    data=self.old_probs.array())
                if self.lambda_rate > 0:
                    file.create_dataset('rewards', data=self.rewards.array())
                    file.create_dataset('terminals',
                                        data=self.terminals.array())
        return path


class PGCAAgent(PGAgent):
    """This class is a PGAgent adapted for continuous action spaces."""

    def __init__(self, amodel, discounted_rate, max_action,
                 create_memory=lambda: Memory(),
                 policy=None):
        """Initalizes the Policy Gradient Agent.
        params:
            amodel: A keras model, which takes the state as input and outputs
                    actions (regularization losses are not applied,
                    and compiled loss are not used)
            discounted_rate: A float within 0.0-1.0, which is the rate that
                             future rewards should be counted for the current
                             reward
            max_action: A float/integer, which is the max output of amodel
            create_memory: A function, which returns a Memory instance
        """
        raise NotImplementedError('This Agent has not been '
                                  'implemented in this version.')
        PGAgent.__init__(self, amodel, discounted_rate,
                         create_memory=create_memory,
                         policy=None)
        self.max_action = max_action


class TD3Agent(DDPGAgent):
    """This class (Twin Delayed DDPG Agent) attempts to mitigate
       the problems that a DDPGAgent faces through clipping Q targets
       between two Q models, delaying policy updates, and adding noise
       to target actions.
    """

    def __init__(self, policy, amodel, cmodel, discounted_rate,
                 create_memory=lambda: Memory()):
        """Initalizes the DDPG Agent.
        params:
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
        if not isinstance(policy, NoisePolicy):
            raise ValueError('The policy parameter must be a '
                             'instance of NoisePolicy.')
        DDPGAgent.__init__(self, policy, amodel, cmodel, discounted_rate,
                           create_memory=create_memory,
                           enable_target=True)
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
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=tf.keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=tf.keras.backend.floatx()),
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
        params:
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
        next_qvalues = tf.minimum(next_qvalues1, next_qvalues2)
        qvalues_true = (rewards +
                        self.discounted_rate * next_qvalues * terminals)
        # Critic
        with tf.GradientTape() as tape:
            qvalues_pred1, qvalues_pred2 = self.cmodel(
                [states, actions], training=True
            )
            if len(self.cmodel.losses) > 0:
                reg_loss = tf.math.add_n(self.cmodel.losses)
            else:
                reg_loss = 0
            loss1 = self.cmodel.loss_functions[0](qvalues_true, qvalues_pred1)
            loss2 = self.cmodel.loss_functions[0](qvalues_true, qvalues_pred2)
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
        if mini_batch > 0 and len(self.states) > mini_batch:
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat+1):
            if verbose:
                print(f'Repeat {count}/{repeat}')
            indexes = np.random.choice(np.arange(len(self.states)),
                                       size=length, replace=False)
            if length >= 10000:  # depends on cpu and other factors
                next_states_arr = self.next_states.array()[indexes]
                states_arr = self.states.array()[indexes]
                actions_arr = self.actions.array()[indexes]
                rewards_arr = self.rewards.array()[indexes]
                terminals_arr = self.terminals.array()[indexes]
            else:
                next_states_arr = np.empty((length, *self.states[0].shape))
                states_arr = np.empty((length, *self.states[0].shape))
                actions_arr = np.empty((length, *self.actions[0].shape))
                rewards_arr = np.empty(length)
                terminals_arr = np.empty(length)
                for ndx, rndx in enumerate(indexes):
                    states_arr[ndx] = self.states[rndx]
                    next_states_arr[ndx] = self.next_states[rndx]
                    actions_arr[ndx] = self.actions[rndx]
                    rewards_arr[ndx] = self.rewards[rndx]
                    terminals_arr[ndx] = self.terminals[rndx]

            self._train(states_arr, next_states_arr, actions_arr,
                        terminals_arr, rewards_arr, epochs, batch_size,
                        policy_noise_std, policy_noise_clip,
                        actor_update_infreq, verbose=verbose)

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

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
        return DDPGAgent.save(self, path, save_model=save_model,
                              save_data=save_data, note='T3DAgent Save')


class MultipleAgentWrapper(Agent):
    """This class is a wrapper class, which allows mutiple agents to be combined for
       outputing mutiple actions.
    """

    def __init__(self, agents):
        """Initalizes the agent.
        params:
            agents: A list of agents
        """
        self.agents = agents
        for agent in self.agents:
            assert isinstance(agent, Agent), (
                'Invalid agent value. Must be an Agent.'
            )

    def select_action(self, state, training=False):
        """Returns the actions the agents "believe" to be
           suited for the given state.
        params:
            state: A value or list of values, which is the
                   state to get the action for
            training: A boolean, which determines if all
                      agents are training
        return: A list of values, which are the selected actions
        """
        actions = []
        for agent in self.agents:
            actions.append(agent.select_action(state, training))
        return actions

    def set_playing_data(self, *args, **kwargs):
        """Sets the episode data for all agents.
           (all agents must take same params)
        """
        for agent in self.agents:
            agent.set_playing_data(*args, **kwargs)

    def add_memory(self, state, actions, new_state, reward, terminal):
        """Adds information from one step in the environment to all agents.
        params:
            state: A value or list of values, which is the
                   state of the environment before the
                   action was performed
            actions: A list of values, which are the actions each agent took
            new_state: A value or list of values, which is the
                       state of the environment after performing
                       the action
            reward: A float/integer, which is the evaluation of
                    the action performed
            terminal: A boolean, which determines if this call to
                      add memory is the last for the episode
        """
        for agent, action in zip(self.agents, actions):
            agent.add_memory(state, action, new_state, reward, terminal)
        
    def forget(self, agent_ndx=None):
        """Forgets or clears all memory for all agents."""
        for agent in self.agents:
            agent.forget()

    def end_episode(self):
        """Ends the episode for all agents."""
        for agent in self.agents:
            agent.end_episode()

    def learn(self, *args, **kwargs):
        """Trains all the agents on a batch of their experiences.
           (all agents must take same params)
        """
        for agent in self.agents:
            agent.learn(*args, **kwargs)
        
    def load(self, path):
        """Loads a save from a folder for each agent.
        params:
            path: A string, which is the path to a folder to load
        """
        for agent in sorted(os.listdir(path)):
            agent.load(os.path.join(path, agent))

    def save(self, path, notes):
        """Saves agents to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            notes: A list of strings, which are saved in an agent's folder
        return: A string, which is the complete path of the save
        """
        time = datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        for agent, note in zip(self.agents, notes):
            agent.save(path, note)
        return path


class HeadAgentWrapper(Agent):
    """This class is a wrapper class, which allows one agent to handle
       multiple states at one time.
    """

    def __init__(self, agent, num_minions, memory_order_matters=None):
        """Initalizes the agent.
        params:
            agent: An agent
            num_minions: An integer, which is the number of minions
                         this agent acts for (if zero, the amount can vary)
            memory_order_matters: A boolean, which determines if
                                  the order memory is added matters
                                  (if true, basically the agent's 
                                  end_episode method is important)
        """
        self.agent = agent
        assert isinstance(self.agent, Agent), (
            'Invalid agent value. Must be an Agent.'
        )
        self.num_minions = num_minions
        if memory_order_matters is None:
            memory_order_matters = isinstance(self.agent, PGAgent)
        self.memory_order_matters = memory_order_matters
        if self.memory_order_matters:
            if self.num_minions > 0:
                self.minion_memory = []
                for _ in range(self.num_minions):
                    self.minion_memory.append([])
            else:
                raise ValueError(
                    'num_minions cannot be zero if memory order matters'
                )

    def select_action(self, states, training=False):
        """Returns the actions the agent "believes" to be
           suited for the given states.
        params:
            states: A list of states (if a value is None
                    action will be None)
            training: A boolean, which determines if the
                      agent is training
        return: A list of values, which are the selected actions
        """
        actions = []
        for state in states:
            if state is None:
                actions.append(None)
            else:
                actions.append(self.agent.select_action(state, training))
        return actions

    def set_playing_data(self, *args, **kwargs):
        """Sets the episode data for the agent."""
        self.agent.set_playing_data(*args, **kwargs)

    def add_memory(self, states, actions, new_states, rewards, terminals):
        """Adds information from one step in the environment to the agent.
        params:
            states: A list of states (if a value is None
                    that memory will not be added)
            actions: A list of actions (if a value is None
                    that memory will not be added)
            new_states: A list of new_states 
            rewards: A list of rewards
            terminals: A list of terminals
        """
        if self.memory_order_matters:
            for ndx in range(self.num_minions):
                if states[ndx] is not None and actions[ndx] is not None:
                    memory = (states[ndx], actions[ndx], new_states[ndx],
                              rewards[ndx], terminals[ndx])
                    self.minion_memory[ndx].append(memory)
                    if terminals[ndx]:
                        for memory in self.minion_memory[ndx]:
                            self.agent.add_memory(*memory)
                        self.minion_memory[ndx].clear()  
                        self.agent.end_episode()    
        else:
            memories = zip(states, actions, new_states, rewards, terminals)
            for memory in memories:
                if memory[0] is not None and memory[1] is not None:
                    self.agent.add_memory(*memory)
        
    def forget(self):
        """Forgets or clears all memory for the agent."""
        self.agent.forget()
        if self.memory_order_matters:
            for ndx in range(self.num_minions):
                self.minion_memory[ndx].clear()

    def end_episode(self):
        """Ends the episode for the agent."""
        if self.memory_order_matters:
            for ndx in range(self.num_minions):
                for memory in self.minion_memory[ndx]:
                    self.agent.add_memory(*memory)
                self.minion_memory[ndx].clear()
                self.agent.end_episode()

    def learn(self, *args, **kwargs):
        """Trains the agent on a batch of its experiences.
        """
        self.agent.learn(*args, **kwargs)
        
    def load(self, path):
        """Loads a save from a folder for each agent.
        params:
            path: A string, which is the path to a folder to load
        """
        self.agent.load(path)

    def save(self, path, note):
        """Saves agents to a new folder.
        params:
            path: A string, which is the path to a folder to save within
            note: A string, which is saved in the agent's folder
        return: A string, which is the complete path of the save
        """
        return self.agent.save(path, note)
