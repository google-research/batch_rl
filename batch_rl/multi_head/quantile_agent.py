# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributional RL agent using quantile regression.

This loss is computed as in "Distributional Reinforcement Learning with Quantile
Regression" - Dabney et. al, 2017"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class QuantileAgent(rainbow_agent.RainbowAgent):
  """An extension of Rainbow to perform quantile regression."""

  def __init__(self,
               sess,
               num_actions,
               kappa=1.0,
               network=atari_helpers.QuantileNetwork,
               num_atoms=200,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=50000,
               update_period=4,
               target_update_period=10000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.1,
               epsilon_eval=0.05,
               epsilon_decay_period=1000000,
               replay_scheme='prioritized',
               tf_device='/cpu:0',
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00005, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Args:
      sess: A `tf.Session` object for running associated ops.
      num_actions: Int, number of actions the agent can take at any state.
      kappa: Float, Huber loss cutoff.
      network: tf.Keras.Model, expects 3 parameters: num_actions, num_atoms,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See atari_helpers.QuantileNetwork
        as an example.
      num_atoms: Int, the number of buckets for the value function distribution.
      gamma: Float, exponential decay factor as commonly used in the RL
        literature.
      update_horizon: Int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: Int, number of stored transitions for training to
        start.
      update_period: Int, period between DQN updates.
      target_update_period: Int, ppdate period for the target network.
      epsilon_fn: Function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon), and which returns the epsilon value used for
        exploration during training.
      epsilon_train: Float, final epsilon for training.
      epsilon_eval: Float, epsilon during evaluation.
      epsilon_decay_period: Int, number of steps for epsilon to decay.
      replay_scheme: String, replay memory scheme to be used. Choices are:
        uniform - Standard (DQN) replay buffer (Mnih et al., 2015)
        prioritized - Prioritized replay buffer (Schaul et al., 2015)
      tf_device: Tensorflow device with which the value function is computed
        and trained.
      optimizer: A `tf.train.Optimizer` object for training the model.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.kappa = kappa

    super(QuantileAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        network=network,
        num_atoms=num_atoms,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _create_network(self, name):
    """Builds a Quantile ConvNet.

    Equivalent to Rainbow ConvNet, only now the output logits are interpreted
    as quantiles.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.

    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_actions, self._num_atoms, name=name)
    return network

  def _build_target_distribution(self):
    batch_size = tf.shape(self._replay.rewards)[0]
    # size of rewards: batch_size x 1
    rewards = self._replay.rewards[:, None]
    # size of tiled_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        self._replay_next_target_net_outputs.q_values, axis=1)[:, None]
    batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
    # size of next_qt_argmax: batch_size x 2
    batch_indexed_next_qt_argmax = tf.concat(
        [batch_indices, next_qt_argmax], axis=1)
    # size of next_logits (next quantiles): batch_size x num_atoms
    next_logits = tf.gather_nd(
        self._replay_next_target_net_outputs.logits,
        batch_indexed_next_qt_argmax)
    return rewards + gamma_with_terminal * next_logits

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training.
    """
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    bellman_errors = (target_distribution[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
    huber_loss = (  # Eq. 9 of paper.
        tf.to_float(tf.abs(bellman_errors) <= self.kappa) *
        0.5 * bellman_errors ** 2 +
        tf.to_float(tf.abs(bellman_errors) > self.kappa) *
        self.kappa * (tf.abs(bellman_errors) - 0.5 * self.kappa))

    tau_hat = ((tf.range(self._num_atoms, dtype=tf.float32) + 0.5) /
               self._num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.

    quantile_huber_loss = (  # Eq. 10 of paper.
        tf.abs(tau_hat[None, :, None] - tf.to_float(bellman_errors < 0)) *
        huber_loss)

    # Sum over tau dimension, average over target value dimension.
    loss = tf.reduce_sum(tf.reduce_mean(quantile_huber_loss, 2), 1)

    if self._replay_scheme == 'prioritized':
      target_priorities = self._replay.tf_get_priority(self._replay.indices)
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      loss_weights = 1.0 / tf.sqrt(target_priorities + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))

      # Weight loss by inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss
