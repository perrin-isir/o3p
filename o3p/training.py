import gymnasium
import numpy as np
import jax
import jax.numpy as jnp
import tqdm
from typing import Callable, Union, Any, Tuple, Optional, Dict
import optax
import flax
import time
import os

from o3p.agents import AgentTrainState, AgentNetworks, Agent, create_agent_train_state, save_agent_train_state, save_agent_config
from o3p.envs import flatten_recursive, flatten_goalenv_obs, _EnvType, get_env_type, dict_flatten_goalenv_obs
from o3p.buffers import DefaultBuffer, DefaultEpisodicBuffer
from o3p.samplers import DefaultSampler, DefaultEpisodicSampler, HER
from o3p.logs import o3p_log, o3p_log_reset


def time_stamp():
    return time.strftime(
            "%Y-%m-%d_%H:%M:%S", time.localtime()
        ) + "_" + chr(ord('`')+int(time.time() % 1 * 26 + 1))


def optimize_multi_models(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: flax.core.FrozenDict,
    opt2: Any,
    opt_state2: Any,
    params_to_update2: flax.core.FrozenDict,
    opt3: Any,
    opt_state3: Any,
    params_to_update3: flax.core.FrozenDict,
    opt4: Any,
    opt_state4: Any,
    params_to_update4: flax.core.FrozenDict,
) -> Tuple[Any, flax.core.FrozenDict, Any, flax.core.FrozenDict, jnp.ndarray, Any]:
    (loss, aux), grad = jax.value_and_grad(fn_loss, argnums=(0, 1, 2, 3), has_aux=True)(
        params_to_update,
        params_to_update2,
        params_to_update3,
        params_to_update4,
    )
    update, opt_state = opt(grad[0], opt_state)
    params_to_update = optax.apply_updates(params_to_update, update)
    update2, opt_state2 = opt2(grad[1], opt_state2)
    params_to_update2 = optax.apply_updates(params_to_update2, update2)
    update3, opt_state3 = opt3(grad[2], opt_state3)
    params_to_update3 = optax.apply_updates(params_to_update3, update3)
    update4, opt_state4 = opt4(grad[3], opt_state4)
    params_to_update4 = optax.apply_updates(params_to_update4, update4)
    return (
        opt_state, 
        params_to_update, 
        opt_state2, 
        params_to_update2, 
        opt_state3, 
        params_to_update3,
        opt_state4, 
        params_to_update4,
        aux   
    )


def grad_update(train_state: AgentTrainState, 
                networks: AgentNetworks, 
                loss_fn: Callable):
    (
        opt_state_value, params_value,
        opt_state_critic, params_critic, 
        opt_state_actor, params_actor,
        opt_state_scalars, scalars,
        update_info
    ) = optimize_multi_models(
        loss_fn,
        networks.opt_value,
        train_state.opt_state_value,
        train_state.params_value,
        networks.opt_critic,
        train_state.opt_state_critic,
        train_state.params_critic,
        networks.opt_actor,
        train_state.opt_state_actor,
        train_state.params_actor,
        networks.opt_scalars,
        train_state.opt_state_scalars,
        train_state.scalars
    )

    return train_state._replace(
        opt_state_value=opt_state_value, 
        params_value=params_value, 
        opt_state_critic=opt_state_critic, 
        params_critic=params_critic, 
        opt_state_actor=opt_state_actor, 
        params_actor=params_actor, 
        opt_state_scalars=opt_state_scalars, 
        scalars=scalars,
    ), update_info


def evaluate_parallel(
    algo: Agent,
    train_state: AgentTrainState,    
    act_fn: Callable, 
    envs: gymnasium.vector.vector_env.VectorEnv, 
    env_type: _EnvType,
    rng: jax.random.PRNGKey,
    networks: AgentNetworks,
    obs_mean: float, 
    obs_std: float
) -> float:
    iterations = algo.config.eval_episodes // envs.num_envs
    rng, subkey = jax.random.split(rng)
    observation, _ = envs.reset(seed=subkey[0].item())
    if env_type == _EnvType.GOALENV:
        observation = flatten_goalenv_obs(observation)
    finished = False
    counts = np.zeros(observation.shape[0])
    sum_rewards = np.zeros(observation.shape[0])
    #TODO: correct big issue with coef
    coef = np.ones(observation.shape[0]) 
    while not finished:
        rng, subkey = jax.random.split(rng)
        action = act_fn(train_state, algo.config, observation, subkey, networks, 
                        deterministic=True, max_action=1.0).squeeze()
        observation, reward, terminated, truncated, info = envs.step(
            np.asarray(action))
        if env_type == _EnvType.GOALENV:
            observation = flatten_goalenv_obs(observation)
        done = logical_or(terminated, truncated)
        sum_rewards += coef * reward
        if done.max():
            rng, subkey = jax.random.split(rng)
            observation, _ = envs.reset(
                options={"reset_mask": done.flatten()}, 
                seed=subkey[0].item()
            )
            if env_type == _EnvType.GOALENV:
                observation = flatten_goalenv_obs(observation)
            counts += done
            coef = counts >= iterations
            finished = coef.min()
    return sum_rewards.sum() / algo.config.eval_episodes


def train_offline(buffer, 
                  env_name,
                  env,
                  env_creator,
                  algo, 
                  obs_mean=None,
                  obs_std=None, 
                  ref_max_score=None):

    env_type = get_env_type(env.observation_space)
    stamp = algo.__class__.__name__ + "_" + env_name + "_" + time_stamp()

    if env_type == _EnvType.GOALENV:
        observation_dim = len(flatten_goalenv_obs(env.observation_space.sample()))
        # buffer.sampler = HER(env.unwrapped.compute_reward)
        # buffer.sampler = DefaultEpisodicSampler()
    else:
        observation_dim = len(env.observation_space.sample())
    action_dim = len(env.action_space.sample())

    eval_envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: env_creator()] * algo.config.eval_episodes,
        autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED
    ) 

    train_state, networks = create_agent_train_state(
        jax.random.PRNGKey(algo.config.seed),
        observation_dim,
        action_dim,
        algo.config,
    )
    rng = jax.random.PRNGKey(algo.config.seed)
    rng, subkey = jax.random.split(rng)

    if algo.config.log_interval is not None:
        assert algo.config.log_interval % algo.config.eval_interval == 0, \
            "algo.config.log_interval must be a multiple of algo.config.eval_interval."

    num_steps = algo.config.max_steps // algo.config.n_jitted_updates
    eval_interval = algo.config.eval_interval // algo.config.n_jitted_updates
    if algo.config.log_interval is not None:
        log_interval = eval_interval * (
            algo.config.log_interval // algo.config.eval_interval)
    else:
        log_interval = None

    if algo.config.save_train_state_interval is not None:
        save_train_state_interval = \
            algo.config.save_train_state_interval // algo.config.n_jitted_updates
    else:
        save_train_state_interval = None
    
    if env_type == _EnvType.GOALENV:
        multi_update_fn = jax.jit(algo.offline_update_n_times_goalenv, 
                                  static_argnums=(3,5,6,7))
        # multi_update_fn = jax.jit(algo.update_n_times, static_argnums=(4,5,6))
        # multi_update_fn = jax.jit(algo.offline_episodic_update_n_times, 
        #                           static_argnums=(3,6,7,8))
        # multi_update_fn = algo.offline_episodic_update_n_times
        # p = jnp.array(
        #     np.asarray(buffer.buffers["episode_length"][:buffer.current_size, 0, 0], 
        #                dtype='float64') / np.asarray(
        #         buffer.buffers["episode_length"][:buffer.current_size, 0, 0], 
        #         dtype='float64').sum())
    else:
        multi_update_fn = jax.jit(algo.offline_update_n_times, 
                                  static_argnums=(3,5,6,7))
    act_fn = jax.jit(algo.get_action, static_argnums=(1,4))

    o3p_log_reset()
    save_agent_config(algo.config, os.path.join(algo.config.save_dir, stamp))

    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        
        if env_type == _EnvType.GOALENV:
            train_state, update_info = multi_update_fn(
                i * algo.config.n_jitted_updates, 
                subkey, 
                buffer.buffers, 
                buffer.current_size, 
                train_state, 
                # p,
                networks, 
                algo.config, 
                algo.config.n_jitted_updates)
        else:
            train_state, update_info = multi_update_fn(
                i * algo.config.n_jitted_updates, 
                subkey, 
                buffer.buffers, 
                buffer.current_size, 
                train_state, 
                networks, 
                algo.config, 
                algo.config.n_jitted_updates)

        if i % eval_interval == 0:
            rng, subkey_eval = jax.random.split(rng)
            avg_return = evaluate_parallel(
                algo, 
                train_state, 
                act_fn, 
                eval_envs,
                env_type,
                subkey_eval, 
                networks,
                None,
                None
            )
            iters = i * algo.config.n_jitted_updates
            print(f"iterations: {iters}\n"
                f"avg return ({algo.config.eval_episodes} "
                f"eps): [ {avg_return:.4f} ]"
                f" (max score: {ref_max_score:.4f})")
            
            if log_interval is not None and i % log_interval == 0:
                o3p_log(
                    {"avg_return": avg_return, "iterations": iters} | update_info, 
                    os.path.join(algo.config.save_dir, stamp)
                )

        if save_train_state_interval is not None and \
                i % save_train_state_interval == 0:
            save_agent_train_state(train_state,
                                   os.path.join(algo.config.save_dir, stamp))

    return train_state, networks


def logical_or(
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.logical_or(x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.logical_or(x, y)
    else:
        raise TypeError("Incorrect or non-matching input types.")


def eval_sequential(algo, train_state, act_fn, env, env_type, rng, networks, num_episodes=2):
    episode_returns = []
    for _ in range(num_episodes):
        rng, key = jax.random.split(rng)
        episode_return = 0
        observation, _ = env.reset(seed=key[0].item())
        done = False
        while not done:
            rng, key = jax.random.split(rng)
            action = act_fn(
                train_state, 
                algo.config, 
                flatten_goalenv_obs(observation) 
                    if env_type == _EnvType.GOALENV else observation, 
                key, 
                networks, 
                deterministic=True, 
                max_action=1.0).squeeze()

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        episode_returns.append(episode_return)

    return np.mean(episode_returns), np.min(episode_returns)
 

def train_online(env_name, 
                 algo,
                 parallel_envs: int = 1, 
                 reward_multiplier: float = 1., 
                 obs_mean: Optional[np.ndarray] = None, 
                 obs_std: Optional[np.ndarray] = None):

    dummy_env = gymnasium.make(env_name)
    env_type = get_env_type(dummy_env.observation_space)
    stamp = algo.__class__.__name__ + "_" + env_name + "_" + time_stamp()

    if env_type == _EnvType.GOALENV:
        compute_reward = dummy_env.unwrapped.compute_reward
        max_episode_steps = dummy_env.spec.max_episode_steps
        algo.buffer = DefaultEpisodicBuffer(max_episode_steps, algo.config.buffer_size)
        algo.buffer.sampler = HER(compute_reward)
        # algo.buffer.sampler = DefaultEpisodicSampler()
        observation_dim = len(flatten_goalenv_obs(dummy_env.observation_space.sample()))
    else:
        algo.buffer = DefaultBuffer(algo.config.buffer_size)
        algo.buffer.sampler = DefaultSampler()
        observation_dim = len(dummy_env.observation_space.sample())
    action_dim = len(dummy_env.action_space.sample())

    envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: gymnasium.make(env_name)] * parallel_envs,
        autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED
    )
    eval_envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: gymnasium.make(env_name)] * algo.config.eval_episodes,
        autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED
    )

    train_state, networks = create_agent_train_state(
        jax.random.PRNGKey(algo.config.seed),
        observation_dim,
        action_dim,
        algo.config,
    )

    rng = jax.random.PRNGKey(algo.config.seed)
    rng, subkey_env = jax.random.split(rng)
    num_envs = envs.num_envs
    observation, reset_info = envs.reset(seed=subkey_env[0].item())

    assert algo.config.n_jitted_updates % num_envs == 0, \
        "algo.config.n_jitted_updates must be a multiple of the nr of parallel envs."
    if algo.config.log_interval is not None:
        assert algo.config.log_interval % algo.config.eval_interval == 0, \
            "algo.config.log_interval must be a multiple of algo.config.eval_interval."
       
    num_steps = algo.config.max_steps // algo.config.n_jitted_updates
    intermediate_steps = algo.config.n_jitted_updates // num_envs
    eval_interval = algo.config.eval_interval // algo.config.n_jitted_updates
    if algo.config.log_interval is not None:
        log_interval = eval_interval * (
            algo.config.log_interval // algo.config.eval_interval)
    else:
        log_interval = None

    if algo.config.save_train_state_interval is not None:
        save_train_state_interval = \
            algo.config.save_train_state_interval // algo.config.n_jitted_updates
    else:
        save_train_state_interval = None

    multi_update_fn = jax.jit(algo.update_n_times, static_argnums=(4,5,6))
    act_fn = jax.jit(algo.get_action, static_argnums=(1,4))

    o3p_log_reset()
    save_agent_config(algo.config, os.path.join(algo.config.save_dir, stamp))

    for i in tqdm.tqdm(range(
        1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        for j in range(intermediate_steps):
            rng, subkey = jax.random.split(rng)
            if num_envs * (i * intermediate_steps + j) > algo.config.random_steps:

                action = act_fn(
                    train_state, algo.config, 
                    flatten_goalenv_obs(observation) \
                        if env_type == _EnvType.GOALENV else observation, 
                    subkey, networks, 
                    deterministic=False, max_action=1.0)
                
            else:
                action = envs.action_space.sample()
            next_observation, reward, terminated, truncated, info = envs.step(
                np.asarray(action))

            done = logical_or(terminated, truncated)

            if env_type == _EnvType.GOALENV:
                transition = {
                    "observations": dict_flatten_goalenv_obs(observation),
                    "actions": action,
                    "rewards": reward,
                    "terminations": terminated,
                    "truncations": truncated,
                    "next_observations": dict_flatten_goalenv_obs(next_observation),
                }
                if algo.config.use_infos:
                    transition["infos"] = {
                        key: flatten_recursive(info[key]) for key in info}
                algo.buffer.insert(num_envs, transition, done)
            else:
                transition = {
                    "observations": observation,
                    "actions": action,
                    "rewards": reward,
                    "terminations": terminated,
                    "truncations": truncated,
                    "next_observations": next_observation
                }
                if algo.config.use_infos:
                    transition["infos"] = {
                        key: flatten_recursive(info[key]) for key in info}
                algo.buffer.insert(num_envs, transition)

            if done.max():
                rng, subkey = jax.random.split(rng)
                observation, reset_info = envs.reset(
                    options={"reset_mask": done.flatten()}, 
                    seed=subkey[0].item()
                )
            else:
                observation = next_observation

        if num_envs * (i+1) * intermediate_steps > algo.config.random_steps:

            batches = []
            for _ in range(algo.config.n_jitted_updates):
                batch = algo.buffer.sample(algo.config.batch_size)
                if env_type == _EnvType.GOALENV:
                    s, a, r, d, n_s = (
                        np.hstack((
                            batch["observations.achieved_goal"],
                            batch["observations.desired_goal"],
                            batch["observations.observation"]
                        )),
                        batch["actions"],
                        batch["rewards"],
                        batch["terminations"],
                        np.hstack((
                            batch["next_observations.achieved_goal"],
                            batch["next_observations.desired_goal"],
                            batch["next_observations.observation"]
                        ))
                    )
                    base_batch = {
                        "observations": s,
                        "actions": a,
                        "rewards": r,
                        "terminations": d,
                        "next_observations": n_s
                    }
                    if algo.config.use_infos:
                        base_batch = base_batch | {
                            key: batch[key] for key in batch if key.startswith("infos.")
                            }
                    batches.append(base_batch)
                else:
                    base_batch = {
                        "observations": batch["observations"],
                        "actions": batch["actions"],
                        "rewards": batch["rewards"],
                        "terminations": batch["terminations"],
                        "next_observations": batch["next_observations"]
                    }
                    if algo.config.use_infos:
                        base_batch = base_batch | {
                            key: batch[key] for key in batch if key.startswith("infos.")
                            }
                    batches.append(base_batch)

            rng, key = jax.random.split(rng)
            train_state, update_info = multi_update_fn(
                i * algo.config.n_jitted_updates,
                key,
                batches,
                train_state,
                networks,
                algo.config,
                algo.config.n_jitted_updates
            )

            if i % eval_interval == 0:
                rng, subkey_eval = jax.random.split(rng)
                avg_return = evaluate_parallel(
                    algo, 
                    train_state, 
                    act_fn, 
                    eval_envs,
                    env_type,
                    subkey_eval, 
                    networks,
                    obs_mean,
                    obs_std
                )
                iters = i * algo.config.n_jitted_updates
                print(
                    f"iterations: {iters}\n"
                    f"avg return ({algo.config.eval_episodes} "
                    f"eps): [ {avg_return:.4f} ]")
                
                if log_interval is not None and i % log_interval == 0:
                    o3p_log(
                        {"avg_return": avg_return, "iterations": iters} | update_info, 
                        os.path.join(algo.config.save_dir, stamp)
                    )
            
            if save_train_state_interval is not None and \
                    i % save_train_state_interval == 0:
                save_agent_train_state(train_state,
                    os.path.join(algo.config.save_dir, stamp))
    
    return train_state, networks