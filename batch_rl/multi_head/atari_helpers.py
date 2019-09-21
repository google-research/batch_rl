# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Helper functions for multi head (Ensemble-DQN and REM) agents."""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def multi_head_network(
    num_actions, num_heads, network_type, state,
    transform_strategy=None, **kwargs):
  """The convolutional network used to compute agent's multi-head Q-values.

  Args:
    num_actions: int, number of actions.
    num_heads: int, the number of buckets of the value function distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    transform_strategy: str, Possible options include (1) 'SORT' for sorting
      the heads, (2) 'ROTATE' for rotating the heads randomly. If None,
      then the heads are not reordered.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  net = slim.conv2d(
      net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  net = slim.flatten(net)
  net = slim.fully_connected(net, 512, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_heads,
      activation_fn=None,
      weights_initializer=weights_initializer)

  q_heads = tf.reshape(net, [-1, num_actions, num_heads])
  unordered_q_heads = q_heads

  q_heads, q_values = combine_q_functions(
      unordered_q_heads, transform_strategy, **kwargs)
  return network_type(q_heads, unordered_q_heads, q_values)


def combine_q_functions(q_functions, transform_strategy, **kwargs):
  """Utility function for combining multiple Q functions.

  Args:
    q_functions: Multiple Q-functions concatenated.
    transform_strategy: str, Possible options include (1) 'SORT' for sorting
      the heads, (2) 'ROTATE' for rotating the heads randomly. If None,
      then the heads are not reordered.
    **kwargs: Arbitrary keyword arguments.
  Returns:
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
  """
  # Create q_values before reordering the heads for training
  q_values = tf.reduce_mean(q_functions, axis=-1)

  if transform_strategy == 'STOCHASTIC':
    left_stochastic_matrix = kwargs.get('transform_matrix')
    if left_stochastic_matrix is None:
      raise ValueError('None value provided for stochastic matrix')
    q_functions = tf.tensordot(
        q_functions, left_stochastic_matrix, axes=[[2], [0]])
  elif transform_strategy == 'IDENTITY':
    tf.logging.info('Not sorting Q-function heads')
  else:
    raise ValueError(
        '{} is not a valid reordering strategy'.format(transform_strategy))
  return q_functions, q_values


def nature_dqn_network(state, num_actions):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    state: `tf.Tensor`, contains the agent's current state.
    num_actions: int, number of actions.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  net = tf.contrib.slim.conv2d(net, 32, [8, 8], stride=4)
  net = tf.contrib.slim.conv2d(net, 64, [4, 4], stride=2)
  net = tf.contrib.slim.conv2d(net, 64, [3, 3], stride=1)
  net = tf.contrib.slim.flatten(net)
  net = tf.contrib.slim.fully_connected(net, 512)
  q_values = tf.contrib.slim.fully_connected(
      net, num_actions, activation_fn=None)
  return q_values


def multi_network_dqn(
    num_actions, num_networks, network_type, state,
    transform_strategy=None, **kwargs):
  """Create a Q function using multiple Q-networks."""

  q_networks = []
  device_fn = kwargs.pop('device_fn', lambda i: '/gpu:0')
  for i in range(num_networks):
    with tf.device(device_fn(i)):
      with tf.variable_scope('network_{}'.format(i)):
        q_networks.append(nature_dqn_network(state, num_actions))
  q_networks = tf.stack(q_networks, axis=-1)
  unordered_q_networks = q_networks

  q_networks, q_values = combine_q_functions(
      q_networks, transform_strategy, **kwargs)
  return network_type(q_networks, unordered_q_networks, q_values)


def random_stochastic_matrix(dim, num_cols=None, dtype=tf.float32):
  """Generates a random left stochastic matrix."""
  mat_shape = (dim, dim) if num_cols is None else (dim, num_cols)
  mat = tf.random.uniform(shape=mat_shape, dtype=dtype)
  mat /= tf.norm(mat, ord=1, axis=0, keepdims=True)
  return mat
