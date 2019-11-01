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

"""Random agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent
import numpy as np

import gin


@gin.configurable
class RandomAgent(dqn_agent.DQNAgent):
  """Random Agent."""

  def __init__(self, sess, num_actions, replay_log_dir, **kwargs):
    """This maintains all the DQN default argument values."""
    self._replay_log_dir = replay_log_dir
    super(RandomAgent, self).__init__(sess, num_actions, **kwargs)

  def step(self, reward, observation):
    """Returns a random action."""
    return np.random.randint(self.num_actions)

  def log_final_buffer(self):
    pass
