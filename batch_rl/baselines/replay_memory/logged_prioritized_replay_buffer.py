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

"""Logged Prioritized Replay Buffer."""
# pytype: skip-file


from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gzip
import pickle

from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory import prioritized_replay_buffer

import gin
import numpy as np
import tensorflow.compat.v1 as tf


STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX


class OutOfGraphLoggedPrioritizedReplayBuffer(
    prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer):
  """A logged out-of-graph Replay Buffer for Prioritized Experience Replay."""

  def __init__(self, log_dir, *args, **kwargs):
    """Initializes OutOfGraphLoggedPrioritizedReplayBuffer."""
    super(OutOfGraphLoggedPrioritizedReplayBuffer, self).__init__(
        *args, **kwargs)
    self._log_count = 0
    self._log_dir = log_dir
    tf.gfile.MakeDirs(self._log_dir)

  def add(self, observation, action, reward, terminal, *args):
    super(OutOfGraphLoggedPrioritizedReplayBuffer, self).add(
        observation, action, reward, terminal, *args)
    # Log the replay buffer every time the replay buffer is filled to capacity.
    cur_size = self.add_count % self._replay_capacity
    if cur_size == self._replay_capacity - 1:
      self._log_buffer()
      self._log_count += 1

  def load(self, checkpoint_dir, suffix):
    super(OutOfGraphLoggedPrioritizedReplayBuffer, self).load(
        checkpoint_dir, suffix)
    self._log_count = self.add_count // self._replay_capacity

  def _log_buffer(self):
    """This method will save all the replay buffer's state in a single file."""
    checkpointable_elements = self._return_checkpointable_elements()
    for attr in checkpointable_elements:
      filename = self._generate_filename(self._log_dir, attr, self._log_count)
      with tf.gfile.Open(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
          # Some numpy arrays might not be part of storage
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)
    tf.logging.info('Replay buffer logged to ckpt {number} in {dir}'.format(
        number=self._log_count, dir=self._log_dir))

  def log_final_buffer(self):
    """Logs the replay buffer at the end of training."""
    add_count = self.add_count
    self.add_count = np.array(self.cursor())
    self._log_buffer()
    self._log_count += 1
    self.add_count = add_count


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedLoggedPrioritizedReplayBuffer(
    circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of OutOfGraphLoggedPrioritizedReplayBuffer with in-graph sampling."""

  def __init__(self,
               log_dir,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes WrappedLoggedPrioritizedReplayBuffer."""

    memory = OutOfGraphLoggedPrioritizedReplayBuffer(
        log_dir, observation_shape, stack_size, replay_capacity, batch_size,
        update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype)
    super(WrappedLoggedPrioritizedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        wrapped_memory=memory,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

  def tf_set_priority(self, indices, priorities):
    """Sets the priorities for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].
      priorities: tf.Tensor with dtype float and shape [n].

    Returns:
       A tf op setting the priorities for prioritized sampling.
    """
    return tf.py_func(
        self.memory.set_priority, [indices, priorities], [],
        name='prioritized_replay_set_priority_py_func')

  def tf_get_priority(self, indices):
    """Gets the priorities for the given indices.

    Args:
      indices: tf.Tensor with dtype int32 and shape [n].

    Returns:
      priorities: tf.Tensor with dtype float and shape [n], the priorities at
        the indices.
    """
    return tf.py_func(
        self.memory.get_priority, [indices],
        tf.float32,
        name='prioritized_replay_get_priority_py_func')
