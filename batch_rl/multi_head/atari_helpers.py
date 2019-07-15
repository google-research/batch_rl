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

"""Helper functions for fixed replay agents."""

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

  # Create q_values before reordering the heads for training
  q_values = tf.reduce_mean(q_heads, axis=-1)

  # Apply the reorder strategy
  if transform_strategy == 'STOCHASTIC':
    left_stochastic_matrix = kwargs.get('transform_matrix')
    if left_stochastic_matrix is None:
      raise ValueError('None value provided for stochastic matrix')
    q_heads = tf.tensordot(q_heads, left_stochastic_matrix, axes=[[2], [0]])
  elif transform_strategy == 'IDENTITY':
    tf.logging.info('Not transforming Q-function heads')
  else:
    raise ValueError(
        '{} is not a valid reordering strategy'.format(transform_strategy))

  return network_type(q_heads, unordered_q_heads, q_values)


def uniform_stochastic_matrix(dim, num_cols=1, dtype=tf.float32):
  """Generates a uniform distribution over the simplex."""
  mat = tf.concat([
      tf.zeros(shape=(1, num_cols), dtype=dtype),
      tf.random_uniform(shape=(dim - 1, num_cols), dtype=dtype),
      tf.ones(shape=(1, num_cols), dtype=dtype)
  ], axis=0)
  mat = tf.sort(mat, axis=0)
  mat = mat[1:] - mat[:-1]  # Consecutive differences
  return mat


def random_stochastic_matrix(dim, num_cols=None, dtype=tf.float32):
  """Generates a random left stochastic matrix."""
  mat_shape = (dim, dim) if num_cols is None else (dim, num_cols)
  mat = tf.random.uniform(shape=mat_shape, dtype=dtype)
  mat /= tf.norm(mat, ord=1, axis=0, keepdims=True)
  return mat
