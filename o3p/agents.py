from typing import NamedTuple, Tuple, Dict, Any, List, Optional
# import haiku as hk
import flax
import jax
import abc
import numpy as np
import jax.numpy as jnp
import optax
from gymnasium.spaces import Box as GymnaBox
from gymnasium.spaces import Dict as GymnaDict
import os
import joblib
from omegaconf import OmegaConf

from o3p.models import (
    ContinuousVFunction, ContinuousQFunction, 
    StateDependentGaussianPolicyTanh, StateDependentGaussianPolicy, DeterministicPolicy,
    AgentConfig, AgentTrainState, AgentNetworks
)


def save_agent_config(agent_config: AgentConfig, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "agent_config.yaml")
    fout = open(save_file, "w")
    fout.write(OmegaConf.to_yaml(dict(agent_config)))
    fout.close()
    print(f"agent_config saved in {save_file}")


# def target_update(
#     online_params: hk.Params,
#     target_params: hk.Params,
#     tau: float,
# ) -> hk.Params:
#     return jax.tree_util.tree_map(
#         lambda t, s: (1 - tau) * t + tau * s, target_params, online_params)

def target_update(
    online_params: flax.core.FrozenDict,
    target_params: flax.core.FrozenDict,
    tau: float,
) -> flax.core.FrozenDict:
    return jax.tree_util.tree_map(
        lambda t, s: (1 - tau) * t + tau * s,
        target_params,
        online_params,
    )

class Agent(object):

    def __init__(self, config_dict: dict):
        self.configure(config_dict)
        self.buffer = None

    @abc.abstractmethod
    def configure(
        self, config_dict: dict
    ) -> None:
        return

    @abc.abstractmethod
    def update_models(
        self, 
        rng: jax.random.PRNGKey,
        iteration: int,
        batch: Dict, 
        train_state: AgentTrainState, 
        networks: AgentNetworks,
        config: AgentConfig
    ) -> Tuple["AgentTrainState", Dict]:
        return

    @classmethod
    def update_on_batch(
        self,
        iteration: int,
        key: jax.random.PRNGKey,
        batch: Dict,
        train_state: AgentTrainState,
        networks: AgentNetworks,
        config: AgentConfig,
    ) -> Tuple["AgentTrainState", Dict]:
        train_state, update_info = self.update_models(
            key, iteration, batch, train_state, networks, config)
        if config.target_critic:
            new_params_critic_target = target_update(
                train_state.params_critic, train_state.params_critic_target, config.tau
            )
            train_state = train_state._replace(
                params_critic_target=new_params_critic_target)
        if config.target_value:
            new_params_value_target = target_update(
                train_state.params_value, train_state.params_value_target, config.tau
            )
            train_state = train_state._replace(
                params_value_target=new_params_value_target)
        if config.target_actor:
            new_params_actor_target = target_update(
                train_state.params_actor, train_state.params_actor_target, config.tau
            )
            train_state = train_state._replace(
                params_actor_target=new_params_actor_target)
        return train_state, update_info


    @classmethod
    def offline_update_n_times(
        self,
        iteration: int,
        rng: jax.random.PRNGKey,
        buffers: Dict,
        buffer_size: int,
        train_state: AgentTrainState,
        networks: AgentNetworks,
        config: AgentConfig,
        n: int
    ) -> Tuple["AgentTrainState", Dict]:
        new_train_state = train_state
        for k in range(n):
            _, subkey1, subkey2 = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(
                subkey1, (config.batch_size,), 0, buffer_size
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], buffers)
            new_train_state, update_info = self.update_on_batch(
                iteration + k,
                subkey2,
                batch,
                new_train_state,
                networks,
                config,
            )
        return new_train_state, update_info
    
    @classmethod
    def offline_update_n_times_goalenv(
        self,
        iteration: int,
        rng: jax.random.PRNGKey,
        buffers: Dict,
        buffer_size: int,
        train_state: AgentTrainState,
        networks: AgentNetworks,
        config: AgentConfig,
        n: int
    ) -> Tuple["AgentTrainState", Dict]:
        new_train_state = train_state
        for k in range(n):
            _, subkey1, subkey2 = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(
                subkey1, (config.batch_size,), 0, buffer_size
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], buffers)
            s, a, r, d, n_s = (
                jnp.hstack((
                    batch["observations.achieved_goal"],
                    batch["observations.desired_goal"],
                    batch["observations.observation"]
                )),
                batch["actions"],
                batch["rewards"],
                batch["terminations"],
                jnp.hstack((
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
                "next_observations": n_s,
            }
            if config.use_infos:
                base_batch = base_batch | {
                    key: batch[key] for key in batch if key.startswith("infos.")
                    }
            new_train_state, update_info = self.update_on_batch(
                iteration + k,
                subkey2,
                base_batch,
                new_train_state,
                networks,
                config,
            )
        return new_train_state, update_info

    @classmethod
    def offline_episodic_update_n_times(
        self,
        iteration: int,
        rng: jax.random.PRNGKey,
        buffers: Dict,
        buffer_size: int,
        train_state: AgentTrainState,
        p: jnp.array,
        networks: AgentNetworks,
        config: AgentConfig,
        n: int
    ) -> Tuple["AgentTrainState", Dict]:
        new_train_state = train_state
        for k in range(n):
            _, subkey1, subkey2, subkey3 = jax.random.split(rng, 4)

            episode_idxs = jax.random.choice(
                subkey1, jnp.arange(buffer_size), 
                shape=(config.batch_size,), 
                replace=True, 
                p=p)
            t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
            t_samples = jax.random.randint(
                subkey2, 
                (config.batch_size,), 
                minval=jnp.zeros(config.batch_size, dtype=int),
                maxval=t_max_episodes)
            batch = jax.tree_util.tree_map(
                lambda x: x[episode_idxs, t_samples], buffers)
            s, a, r, d, n_s = (
                jnp.hstack((
                    batch["observations.achieved_goal"],
                    batch["observations.desired_goal"],
                    batch["observations.observation"]
                )),
                batch["actions"],
                batch["rewards"],
                batch["terminations"],
                jnp.hstack((
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
                "next_observations": n_s,
            }
            if config.use_infos:
                base_batch = base_batch | {
                    key: batch[key] for key in batch if key.startswith("infos.")
                    }
            new_train_state, update_info = self.update_on_batch(
                iteration + k,
                subkey3,
                base_batch,
                new_train_state,
                networks,
                config,
            )
        return new_train_state, update_info
    
    @classmethod
    def update_n_times(
        self,
        iteration: int,
        rng: jax.random.PRNGKey,
        batch_list: List[Dict],
        train_state: AgentTrainState,
        networks: AgentNetworks,
        config: AgentConfig,
        n: int
    ) -> Tuple["AgentTrainState", Dict]:
        new_train_state = train_state
        for k in range(n):
            rng, subkey = jax.random.split(rng)
            new_train_state, update_info = self.update_on_batch(
                iteration + k,
                subkey,
                batch_list[k],
                new_train_state,
                networks,
                config,
            )
        return new_train_state, update_info


    @staticmethod
    def get_action(
        train_state: AgentTrainState,
        config: AgentConfig,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        networks: AgentNetworks,
        deterministic: bool = False,
        max_action: float = 1.0,  # Actions should be in [-1, 1] accross all dimensions
    ) -> jnp.ndarray:
        return networks.actor.get_action(
            train_state,
            config,
            observations,
            seed,
            networks,
            deterministic,
            max_action
        )
        

    # @classmethod
    # def get_distributional_action(
    #     self,
    #     train_state: AgentTrainState,
    #     config: AgentConfig,
    #     observations: np.ndarray,
    #     seed: jax.random.PRNGKey,
    #     networks: AgentNetworks,
    #     deterministic: bool = False,
    #     max_action: float = 1.0,  # Actions should be in [-1, 1] accross all dimensions
    # ) -> jnp.ndarray:
    #     dist, deterministic_actions = networks.actor.apply(
    #         train_state.params_actor, observations
    #     )
    #     rnd_actions = dist.sample(seed=seed)
    #     rnd_actions = jnp.clip(rnd_actions, -max_action, max_action)
    #     deterministic_actions = jnp.clip(deterministic_actions, -max_action, max_action)
    #     return rnd_actions * (1. - deterministic) + \
    #           deterministic * deterministic_actions
    
    # @classmethod
    # def get_action(
    #     self,
    #     train_state: AgentTrainState,
    #     config: AgentConfig,
    #     observations: np.ndarray,
    #     seed: jax.random.PRNGKey,
    #     networks: AgentNetworks,
    #     deterministic: bool = False,
    #     max_action: float = 1.0,  # Actions should be in [-1, 1] accross all dimensions
    # ) -> jnp.ndarray:
    #     if not config.distributional_actor:
    #         actions = networks.actor.apply(
    #             train_state.params_actor, observations) * max_action
    #         policy_noise = (config.policy_noise_std * max_action
    #             * (1. - deterministic) * jax.random.normal(seed, actions.shape))
    #         actions = jnp.clip(actions + policy_noise.clip(
    #             -config.policy_noise_clip, config.policy_noise_clip), 
    #             -max_action, max_action)
    #         return actions
    #     else:
    #         return self.get_distributional_action(
    #             train_state, 
    #             config, 
    #             observations, 
    #             seed, 
    #             networks, 
    #             deterministic, 
    #             max_action
    #         )


def fake_args(dim: int):
    return np.zeros((1, dim))


def create_agent_train_state(
    rng: jax.random.PRNGKey,
    observation_dim: int,
    action_dim: int,
    config: AgentConfig,
) -> AgentTrainState:
    
    def fn_value():
        return ContinuousVFunction(
            num_values=config.num_values,
            hidden_units=tuple(config.value_hidden_dims),
        )

    def fn_critic():
        return ContinuousQFunction(
            num_critics=config.num_critics,
            hidden_units=config.critic_hidden_dims,
        )

    if config.distributional_actor:
        if config.tanh_actor:
            def fn_actor():
                return StateDependentGaussianPolicyTanh(
                    num_actors=config.num_actors,
                    action_dim=action_dim,
                    hidden_units=config.actor_hidden_dims,
                    log_std_min=config.policy_log_std_min,
                    log_std_max=config.policy_log_std_max,
                )
        else:
            def fn_actor():
                return StateDependentGaussianPolicy(
                    num_actors=config.num_actors,
                    action_dim=action_dim,
                    hidden_units=config.actor_hidden_dims,
                    log_std_min=config.policy_log_std_min,
                    log_std_max=config.policy_log_std_max,
                )
    else:
        def fn_actor():
            return DeterministicPolicy(
                num_actors=config.num_actors,
                action_dim=action_dim,
                hidden_units=config.actor_hidden_dims,
            )

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    
    def optimizer(lr):
        if config.opt_decay_schedule:
            schedule_fn = optax.cosine_decay_schedule(-lr, config.max_steps)
            return optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        else:
            return optax.adam(lr)

    fake_args_critic = (fake_args(observation_dim), fake_args(action_dim))
    fake_args_value = fake_args_actor = (fake_args(observation_dim),)

    s_critic = fn_critic()
    variables = s_critic.init(key1, *fake_args_critic)
    s_params_critic = variables
    # s_params_critic = variables['params']
    s_params_critic_target = s_params_critic
    opt_init, s_opt_critic = optimizer(config.critic_lr)
    s_opt_state_critic = opt_init(s_params_critic)

    s_value = fn_value()
    variables = s_value.init(key2, *fake_args_value)
    s_params_value = variables
    # s_params_value = variables['params']
    s_params_value_target = s_params_value
    opt_init, s_opt_value = optimizer(config.value_lr)
    s_opt_state_value = opt_init(s_params_value)

    s_actor = fn_actor()
    variables = s_actor.init(key3, *fake_args_actor)
    s_params_actor = variables
    # s_params_actor = variables['params']
    s_params_actor_target = s_params_actor
    opt_init, s_opt_actor = optimizer(config.actor_lr)
    s_opt_state_actor = opt_init(s_params_actor)

    s_scalars = jnp.zeros(config.num_scalars, dtype=jnp.float32)
    opt_init, s_opt_scalars = optimizer(config.scalars_lr)
    s_opt_state_scalars = opt_init(s_scalars)

    return AgentTrainState(
        rng = rng,
        params_critic = s_params_critic,
        params_critic_target = s_params_critic_target,
        opt_state_critic = s_opt_state_critic,
        params_value = s_params_value,
        params_value_target = s_params_value_target,
        opt_state_value = s_opt_state_value,
        params_actor = s_params_actor,
        params_actor_target = s_params_actor_target,
        opt_state_actor = s_opt_state_actor,
        scalars = s_scalars,
        opt_state_scalars = s_opt_state_scalars
    ), AgentNetworks(
        critic = s_critic,
        opt_critic = s_opt_critic,
        value = s_value,
        opt_value = s_opt_value,
        actor = s_actor,
        opt_actor = s_opt_actor,
        opt_scalars = s_opt_scalars,
    )


def save_agent_train_state(train_state: AgentNetworks, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "agent_train_state.joblib") 
    with open(save_file, "wb") as f_:
        joblib.dump(train_state, f_, compress=True)
    print(f"train_state saved in {save_file}")


def load_agent_train_state(save_dir: str):
    return joblib.load(os.path.join(save_dir, "agent_train_state.joblib"))
