# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Multi Head DQN agent."""

import os

from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class MultiHeadDQNAgent(dqn_agent.DQNAgent):
  """DQN agent with multiple heads."""

  def __init__(self,
               sess,
               num_actions,
               num_heads=1,
               transform_strategy='IDENTITY',
               num_convex_combinations=1,
               network=atari_helpers.MultiHeadQNetwork,
               init_checkpoint_dir=None,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_heads: int, Number of heads per action output of the Q function.
      transform_strategy: str, Possible options include (1)
      'STOCHASTIC' for multiplication with a left stochastic matrix. (2)
      'IDENTITY', in which case the heads are not transformed.
      num_convex_combinations: If transform_strategy is 'STOCHASTIC',
        then this argument specifies the number of random
        convex combinations to be created. If None, `num_heads` convex
        combinations are created.
      network: tf.Keras.Model. A call to this object will return an
        instantiation of the network provided. The network returned can be run
        with different inputs to create different outputs. See
        atari_helpers.MultiHeadQNetwork as an example.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      **kwargs: Arbitrary keyword arguments.
    """
    tf.logging.info('Creating MultiHeadDQNAgent with following parameters:')
    tf.logging.info('\t num_heads: %d', num_heads)
    tf.logging.info('\t transform_strategy: %s', transform_strategy)
    tf.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
    tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
    self.num_heads = num_heads
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self._q_heads_transform = None
    self._num_convex_combinations = num_convex_combinations
    self.transform_strategy = transform_strategy
    super(MultiHeadDQNAgent, self).__init__(
        sess, num_actions, network=network, **kwargs)

  def _create_network(self, name):
    """Builds a multi-head Q-network that outputs Q-values for multiple heads.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    kwargs = {}  # Used for passing the transformation matrix if any
    if self._q_heads_transform is None:
      if self.transform_strategy == 'STOCHASTIC':
        tf.logging.info('Creating q_heads transformation matrix..')
        self._q_heads_transform = atari_helpers.random_stochastic_matrix(
            self.num_heads, num_cols=self._num_convex_combinations)
    if self._q_heads_transform is not None:
      kwargs.update({'transform_matrix': self._q_heads_transform})
    network = self.network(
        num_actions=self.num_actions,
        num_heads=self.num_heads,
        transform_strategy=self.transform_strategy,
        name=name,
        **kwargs)
    return network

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension for each head.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_heads, axis=1)
    is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
    is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
    rewards = tf.expand_dims(self._replay.rewards, axis=-1)
    return rewards + (
        self.cumulative_gamma * replay_next_qt_max * is_non_terminal)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    actions = self._replay.actions
    indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
    replay_chosen_q = tf.gather_nd(
        self._replay_net_outputs.q_heads, indices=indices)
    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    q_head_losses = tf.reduce_mean(loss, axis=0)
    final_loss = tf.reduce_mean(q_head_losses)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', final_loss)
    return self.optimizer.minimize(final_loss)
