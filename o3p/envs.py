from typing import Dict, Tuple
import gymnasium
from gymnasium.spaces import Box as GymnaBox
from gymnasium.spaces import Dict as GymnaDict
import minari
import numpy as np
import enum
import ogbench

from o3p.buffers import DefaultBuffer, proper_reshape


class _EnvType(enum.Enum):
    BOX = 0
    GOALENV = 1


def flatten_recursive(obs):
    if type(obs)==dict:
        keys = list(sorted(obs.keys()))
        tuple_out = [
                flatten_recursive(obs[key]) for key in keys
            ]
        return np.hstack(tuple_out)
    else:
        return obs


def flatten_goalenv_obs(observation):
    return np.hstack((
        flatten_recursive(observation["achieved_goal"]), 
        flatten_recursive(observation["desired_goal"]), 
        flatten_recursive(observation["observation"])
    ))


def dict_flatten_goalenv_obs(observation):
    return {
        "achieved_goal": flatten_recursive(observation["achieved_goal"]), 
        "desired_goal": flatten_recursive(observation["desired_goal"]), 
        "observation": flatten_recursive(observation["observation"])
    }


def get_env_type(obs_space):
    if isinstance(obs_space, GymnaBox):
        env_type = _EnvType.BOX
    elif isinstance(obs_space, GymnaDict) \
            and "observation" in obs_space.spaces \
            and "desired_goal" in obs_space.spaces \
            and "achieved_goal" in obs_space.spaces:
        env_type = _EnvType.GOALENV
    else:
        raise ValueError("Unsupported environment type.")
    return env_type


def compute_mean_std(x: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(0)
    std = x.std(0) + eps
    return mean, std


def wrap_env(
    env : gymnasium.Env,
    env_type,
    obs_mean: Dict,
    obs_std: Dict,
) -> gymnasium.Env:
    
    if env_type == _EnvType.GOALENV:
        def normalize_state(obs):
            # Note: epsilon should be already added in std.
            transformed_obs = {}
            transformed_obs["desired_goal"] = (
                obs["desired_goal"] - obs_mean["observations.desired_goal"]
                ) / obs_std["observations.desired_goal"]
            transformed_obs["achieved_goal"] = (
                obs["achieved_goal"] - obs_mean["observations.achieved_goal"]
                ) / obs_std["observations.achieved_goal"]           
            transformed_obs["observation"] = (
                obs["observation"] - obs_mean["observations.observation"]
                ) / obs_std["observations.observation"]
            return transformed_obs
    else:
        def normalize_state(obs):
            # Note: epsilon should be already added in std.
            return ( obs - obs_mean["observations"]) / obs_std["observations"]
    
    env = gymnasium.wrappers.TransformObservation(
        env, normalize_state, env.observation_space)
    return env


def get_minari(
    env_name: str,
    normalize_reward: bool = True,
    normalize_obs : bool = False,
):

    _dataset = minari.load_dataset(env_name, download=True)
    env = _dataset.recover_environment()

    buffer = DefaultBuffer(_dataset.total_steps)

    def compute_pre_observations(observations, max_step=None):
        # __import__("IPython").embed()
        if max_step is None:
            if type(observations) == dict:
                return {
                    key: flatten_recursive(observations[key])[:-1] \
                        for key in observations
                }
            else:
                return observations[:-1]
        else:
            if type(observations) == dict:
                return {
                    key: flatten_recursive(observations[key])[:max_step-1] \
                        for key in observations
                }
            else:
                return observations[:max_step-1]

    def compute_next_observations(observations, max_step=None):
        if max_step is None:
            if type(observations) == dict:
                return {
                    key: flatten_recursive(observations[key])[1:] \
                        for key in observations
                }
            else:
                return observations[1:]
        else:
            if type(observations) == dict:
                return {
                    key: flatten_recursive(observations[key])[1:max_step] \
                        for key in observations
                }
            else:
                return observations[1:max_step]

    def infos_flatten_recursive(x: Dict, max_step=None):
        if max_step is None:
            return {
                key: flatten_recursive(x[key])[1:] for key in x
            }
        else:
            return {
                key: flatten_recursive(x[key])[1:max_step] for key in x
            }

    ref_max_score = -np.inf
    for ep in _dataset:
        max_step = None

        # if "antmaze" in env_name:
        #     max_step = ep.infos["success"].argmax()
        #     if max_step == 0:
        #         max_step = None
        
        transition_dict = {
            "observations": compute_pre_observations(
                ep.observations, max_step + 1 if max_step is not None else None),
            "actions": proper_reshape(ep.actions[:max_step]),
            "rewards": proper_reshape(ep.rewards[:max_step]),
            "terminations": proper_reshape(ep.terminations[:max_step]),
            "truncations": proper_reshape(ep.truncations[:max_step]),
            "next_observations": compute_next_observations(
                ep.observations, max_step + 1 if max_step is not None else None),
            "infos": infos_flatten_recursive(
                ep.infos, max_step + 1 if max_step is not None else None)
        }
        score = transition_dict["rewards"].sum()
        if score > ref_max_score: 
            ref_max_score = score

        if "antmaze" in env_name:
            transition_dict["rewards"] -= 1.

        buffer.insert(transition_dict["rewards"].shape[0], transition_dict)

    obs_mean = None
    obs_std = None

    if normalize_reward:
        reward_max = buffer.buffers["rewards"].max()
        reward_min = buffer.buffers["rewards"].min()
        normalizing_factor = 1000. / (reward_max - reward_min)
        buffer.buffers["rewards"] = \
            buffer.buffers["rewards"] * normalizing_factor
    else:
        normalizing_factor = 1.

    if normalize_obs:
        obs_mean = {}
        obs_std = {}
        for key in buffer.buffers:
            if key.startswith("observations"):
                obs_mean[key], obs_std[key] = compute_mean_std(
                    buffer.buffers[key], 1e-3)
                buffer.buffers[key] = (
                    buffer.buffers[key] - obs_mean[key]) / obs_std[key]
                
            # buffer.buffers["next_observations"] = (
            #     buffer.buffers["next_observations"] - obs_mean) / (obs_std + 1e-5)

        for key in buffer.buffers:
            if key.startswith("next_observations"):
                buffer.buffers[key] = (
                    buffer.buffers[key] - obs_mean[key[5:]]) / obs_std[key[5:]]

    buffer.to_jax()

    env_type = get_env_type(env.observation_space)

    def env_creator():
        if obs_mean is not None and obs_std is not None:
            return wrap_env(_dataset.recover_environment(), env_type, obs_mean, obs_std)
        else:
            return _dataset.recover_environment()

    return (
        buffer, 
        env, 
        env_creator, 
        normalizing_factor, 
        obs_mean, 
        obs_std, 
        ref_max_score
    )


def get_ogbench(
    env_name: str,
    normalize_reward: bool = True,
    normalize_obs : bool = False,
):

    env, _dataset, _val_dataset = ogbench.make_env_and_datasets(env_name)

    buffer_size = _dataset["rewards"].shape[0]
    buffer = DefaultBuffer(buffer_size)

    transition_dict = {
        "observations": proper_reshape(_dataset["observations"][:]),
        "actions": proper_reshape(_dataset["actions"][:]),
        "rewards": proper_reshape(_dataset["rewards"][:]),
        "terminations": proper_reshape(_dataset["terminals"][:]),
        "truncations": proper_reshape(_dataset["terminals"][:]) * proper_reshape(_dataset["masks"][:]),
        "next_observations": proper_reshape(_dataset["next_observations"][:]),
        "infos": {}
    }
    ref_max_score = 0.

    buffer.insert(transition_dict["rewards"].shape[0], transition_dict)

    obs_mean = None
    obs_std = None

    if normalize_reward:
        reward_max = buffer.buffers["rewards"].max()
        reward_min = buffer.buffers["rewards"].min()
        normalizing_factor = 1000. / (reward_max - reward_min)
        buffer.buffers["rewards"] = \
            buffer.buffers["rewards"] * normalizing_factor
    else:
        normalizing_factor = 1.

    if normalize_obs:
        obs_mean = {}
        obs_std = {}
        for key in buffer.buffers:
            if key.startswith("observations"):
                obs_mean[key], obs_std[key] = compute_mean_std(
                    buffer.buffers[key], 1e-3)
                buffer.buffers[key] = (
                    buffer.buffers[key] - obs_mean[key]) / obs_std[key]
                
            # buffer.buffers["next_observations"] = (
            #     buffer.buffers["next_observations"] - obs_mean) / (obs_std + 1e-5)

        for key in buffer.buffers:
            if key.startswith("next_observations"):
                buffer.buffers[key] = (
                    buffer.buffers[key] - obs_mean[key[5:]]) / obs_std[key[5:]]

    buffer.to_jax()

    env_type = get_env_type(env.observation_space)

    def env_creator():
        if obs_mean is not None and obs_std is not None:
            return wrap_env(
                ogbench.make_env_and_datasets(env_name, env_only=True), 
                env_type, 
                obs_mean, 
                obs_std)
        else:
            return ogbench.make_env_and_datasets(env_name, env_only=True)

    return (
        buffer, 
        env, 
        env_creator, 
        normalizing_factor, 
        obs_mean, 
        obs_std, 
        ref_max_score
    )