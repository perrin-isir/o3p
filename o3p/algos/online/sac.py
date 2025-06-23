from omegaconf import OmegaConf
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
import gymnasium
import haiku as hk
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from o3p.buffers import extract_from_batch
from o3p.agents import AgentConfig, AgentTrainState, AgentNetworks, Agent
from o3p.training import grad_update


class SAC(Agent):
    @classmethod
    def configure(
        self, config_dict: dict
    ) -> None:
        config_dict = self.CustomConfig(**config_dict)
        # non-modifiable params:
        config_dict.distributional_actor = True
        config_dict.tanh_actor = True
        config_dict.num_critics = 2
        config_dict.target_critic = True
        config_dict.target_value = False
        config_dict.target_actor = False

        self.config = config_dict

    class CustomConfig(AgentConfig):
        # default and specific params:
        critic_lr: float = 3e-3
        scalars_lr: float = 3e-3
        actor_lr: float = 3e-3
        tau: float = 1e-2
        policy_freq: int = 2

    @classmethod
    def update_models(
        self, 
        key: jax.random.PRNGKey,
        iteration: int,
        batch: Dict, 
        train_state: AgentTrainState, 
        networks: AgentNetworks,
        config: AgentConfig
    ) -> Tuple["AgentTrainState", Dict]:

        (observations, actions, rewards, next_observations, dones, infos) = \
            extract_from_batch(batch)
        
        def _loss(
            params_value: hk.Params,    
            params_critic: hk.Params,
            params_actor: hk.Params,
            scalars: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, Dict]:

            key1, key2 = jax.random.split(key)

            log_alpha = scalars[0]
            alpha = jnp.exp(log_alpha)

            next_dist, _ = networks.actor.apply(params_actor, next_observations)
            next_action, next_log_pi = next_dist.sample_and_log_prob(seed=key1)

            next_q_min = jnp.asarray(networks.critic.apply(
                train_state.params_critic_target, next_observations, next_action
            )).min(axis=0)
            next_q_min -= alpha * next_log_pi
            target_q = jax.lax.stop_gradient(
                rewards + (1.0 - dones) * config.discount * next_q_min)

            q_list = networks.critic.apply(params_critic, observations, actions)
            loss_critic = sum((jnp.square(target_q - q)).mean() for q in q_list)

            dist, _ = networks.actor.apply(params_actor, observations)
            action_actor, log_pi = dist.sample_and_log_prob(seed=key2)

            mean_q = (
                jnp.asarray(networks.critic.apply(
                    jax.lax.stop_gradient(params_critic), observations, action_actor)
                ).min(axis=0)
            ).mean()
            mean_log_pi = log_pi.mean()
            loss_actor = (jax.lax.stop_gradient(alpha) * mean_log_pi - mean_q)

            target_ent = float(-actions.shape[-1])
            loss_alpha = -log_alpha * (target_ent + jax.lax.stop_gradient(mean_log_pi))
            return (
                loss_critic + \
                    (iteration % config.policy_freq == 0) * (loss_actor + loss_alpha), 
                {
                    "loss_critic": loss_critic, 
                    "loss_actor": loss_actor, 
                    "loss_alpha": loss_alpha
                }
            )

        return grad_update(train_state, networks, _loss)