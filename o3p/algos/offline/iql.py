import jax
import sys
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any
import flax
import gymnasium

from o3p.buffers import extract_from_batch
from o3p.models import AgentConfig, AgentTrainState, AgentNetworks
from o3p.agents import Agent
from o3p.training import grad_update


def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQL(Agent):
    @classmethod
    def configure(
        self, config_dict: dict
    ) -> None:
        config_dict = self.CustomConfig(**config_dict)
        # non-modifiable params:
        config_dict.distributional_actor = True
        config_dict.tanh_actor = False
        config_dict.num_critics = 2
        config_dict.num_actors = 1
        config_dict.target_critic = False
        config_dict.target_value = True
        config_dict.target_actor = False

        self.config = config_dict

    class CustomConfig(AgentConfig):
        # default and specific params:
        value_lr: float = 3e-4
        critic_lr: float = 3e-4
        actor_lr: float = 3e-4
        tau: float = 0.005
        expectile: float = 0.9
        beta: float = 10.0

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

        next_v = jax.lax.stop_gradient(networks.value.apply(
            train_state.params_value_target, next_observations)[0])
        target_q = rewards + config.discount * (1 - dones) * next_v

        q = jnp.asarray(jax.lax.stop_gradient(networks.critic.apply(
            train_state.params_critic, observations, actions
        ))).min(axis=0)

        def _loss(
            params_value: flax.core.FrozenDict,
            params_critic: flax.core.FrozenDict,
            params_actor: flax.core.FrozenDict,
            scalars: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, Dict]:

            q1, q2 = networks.critic.apply(params_critic, observations, actions)
            loss_critic = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            v = networks.value.apply(params_value, observations, index=0)
            loss_value = expectile_loss(q - v, config.expectile).mean()

            dist, predicted_actions = networks.actor.apply(params_actor, observations, index=0)
            log_probs = dist.log_prob(actions)
            # log_probs = dist.log_prob(actions.clip(-0.9999,0.9999))
            exp_a = jnp.exp((q - jax.lax.stop_gradient(v)) * config.beta)
            exp_a = jnp.minimum(exp_a, 100.0)
            loss_actor = -(exp_a * log_probs).mean()

            # loss_actor += (exp_a * (predicted_actions - actions)**2).mean()

            return loss_critic + loss_value + loss_actor, {
                    "loss_critic": loss_critic, 
                    "loss_value": loss_value, 
                    "loss_actor": loss_actor
                }

        return grad_update(train_state, networks, _loss)