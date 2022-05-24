# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Environments for experiments with RCE.
"""

import inspect
import os

from absl import logging
import d4rl  # pylint: disable=unused-import
import gin
import gym
from metaworld.envs.mujoco import sawyer_xyz
import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
import tqdm
from env import *

# We need to import d4rl so that gym registers the environments.
os.environ['SDL_VIDEODRIVER'] = 'dummy'


def _get_image_obs(self):
  # The observation returned here should be in [0, 255].
  obs = self.get_image(width=84, height=84)
  return obs[::-1]


@gin.configurable
def load_env(env_name, max_episode_steps=None):
  """Loads an environment.

  Args:
    env_name: Name of the environment.
    max_episode_steps: Maximum number of steps per episode.
  Returns:
    tf_env: A TFPyEnvironment.
  """
  if env_name == 'ant':
    gym_env = AntNoBonusEnv()
    max_episode_steps = gym_env._max_episode_steps
  elif env_name == 'cheetah':
    gym_env = CheetahNoFlipEnv()
    max_episode_steps = gym_env._max_episode_steps
  elif env_name == 'hopper':
    gym_env = HopperNoBonusEnv()
    max_episode_steps = gym_env._max_episode_steps
  elif env_name == 'humanoid':
    gym_env = HumanoidNoBonusEnv()
    max_episode_steps = gym_env._max_episode_steps
  else:
    gym_spec = gym.spec(env_name)
    gym_env = gym_spec.make()
    max_episode_steps = gym_spec.max_episode_steps

  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=max_episode_steps)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  return tf_env


@gin.configurable(denylist=['env', 'env_name'])
def get_data(env, env_name, num_expert_obs=200, terminal_offset=50):
  """Loads the success examples.

  Args:
    env: A PyEnvironment for which we want to generate success examples.
    env_name: The name of the environment.
    num_expert_obs: The number of success examples to generate.
    terminal_offset: For the d4rl datasets, we randomly subsample the last N
      steps to use as success examples. The terminal_offset parameter is N.
  Returns:
    expert_obs: Array with the success examples.
  """
  if env_name in ['hammer-human-v0', 'door-human-v0', 'relocate-human-v0']:
    dataset = env.get_dataset()
    terminals = np.where(dataset['terminals'])[0]
    expert_obs = np.concatenate(
        [dataset['observations'][t - terminal_offset:t] for t in terminals],
        axis=0)
    indices = np.random.choice(
        len(expert_obs), size=num_expert_obs, replace=False)
    expert_obs = expert_obs[indices]
  else:
    # For environments where we generate the expert dataset on the fly, we can
    # improve performance but only generating the number of expert
    # observations that we'll actually use. Not all environments support this
    # function, so we first have to check whether the environment's
    # get_dataset method accepts a num_obs kwarg.
    get_dataset_args = inspect.getfullargspec(env.get_dataset).args
    if 'num_obs' in get_dataset_args:
      dataset = env.get_dataset(num_obs=num_expert_obs)
    else:
      dataset = env.get_dataset()
    indices = np.random.choice(
        dataset['observations'].shape[0], size=num_expert_obs, replace=False)
    expert_obs = dataset['observations'][indices]
  if 'image' in env_name:
    expert_obs = expert_obs.astype(np.uint8)
  logging.info('Done loading expert observations')
  return expert_obs
