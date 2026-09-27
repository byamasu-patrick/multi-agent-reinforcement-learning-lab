from functools import partial
import random

import gymnasium as gym
from omegaconf import DictConfig

from ac.utils import wrappers as mwrappers


def _make_single_env(
    name,
    time_limit,
    clear_info,
    observe_id,
    standardise_rewards,
    wrappers,
    seed,
    enable_video,
    **kwargs,
):
    env = gym.make(name, render_mode="rgb_array" if enable_video else None, **kwargs)
    if clear_info:
        env = mwrappers.ClearInfo(env)
    if time_limit:
        env = gym.wrappers.TimeLimit(env, time_limit)
    env = mwrappers.RecordEpisodeStatistics(env)
    if observe_id:
        env = mwrappers.ObserveID(env)
    if standardise_rewards:
        env = mwrappers.StandardiseReward(env)
    if wrappers is not None:
        for wrapper in wrappers:
            wrapper = (
                getattr(mwrappers, wrapper)
                if hasattr(mwrappers, wrapper)
                else getattr(gym.wrappers, wrapper)
            )
            env = wrapper(env)

    env.reset(seed=seed)
    return env


def _make_parallel_envs(parallel_envs, seed, **env_config):
    if seed is None:
        seed = random.randint(0, 99999)

    # each sub-process builds its own copy of the environment with a distinct seed
    return gym.vector.AsyncVectorEnv(
        [
            partial(_make_single_env, seed=seed + i, **env_config)
            for i in range(parallel_envs)
        ]
    )


def make_env(seed, enable_video=False, **env_config):
    env_config = DictConfig(env_config)
    if "parallel_envs" in env_config and env_config.parallel_envs:
        return _make_parallel_envs(**env_config, enable_video=enable_video, seed=seed)
    return _make_single_env(**env_config, enable_video=enable_video, seed=seed)
