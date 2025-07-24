import jax
import sys
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any
import flax
import gymnasium

from o3p.buffers import extract_from_batch
from o3p.models import AgentConfig, AgentTrainState, AgentNetworks, ActorType
from o3p.agents import Agent
from o3p.training import grad_update


class FQL(Agent):
    @classmethod
    def configure(
        self, config_dict: dict
    ) -> None:
        config_dict = self.CustomConfig(**config_dict)
        # non-modifiable params:
        # config_dict.distributional_actor = True
        # config_dict.tanh_actor = False
        config_dict.actor_type = ActorType.ActorVectorField
        config_dict.num_critics = 2
        config_dict.num_actors = 2
        config_dict.target_critic = True
        config_dict.target_value = False
        config_dict.target_actor = False

        self.config = config_dict

    class CustomConfig(AgentConfig):
        # default and specific params:
        value_lr: float = 3e-4
        critic_lr: float = 3e-4
        actor_lr: float = 3e-4
        tau: float = 0.005
        flow_steps: int = 10
        normalize_q_loss: bool = True
        alpha: float = 0.1
        # critic_hidden_dims: Tuple[int, int] = (512, 512, 512, 512)
        critic_hidden_dims: Tuple[int, int] = (256, 256)
        # actor_hidden_dims: Tuple[int, int] = (512, 512, 512, 512)
        actor_hidden_dims: Tuple[int, int] = (256, 256)

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

        action_dim = networks.actor.action_dim
        batch_size = config.batch_size
    
        (observations, actions, rewards, next_observations, dones, infos) = \
            extract_from_batch(batch)

        max_action = config.max_action

        rng, key_critic = jax.random.split(key)
        next_actions = networks.actor.get_action(
            train_state,
            config,
            next_observations,
            key_critic,
            networks,
            deterministic = False,
            max_action = max_action,
        )
        # noises = jax.random.normal(key_critic, (batch_size, action_dim))
        # next_actions = networks.actor.apply(train_state.params_actor, next_observations, noises, None, index=0)
        # next_actions = jnp.clip(next_actions, -max_action, max_action)

        next_q = jnp.asarray(networks.critic.apply(
            train_state.params_critic_target, next_observations, next_actions
        )).mean(axis=0)

        target_q = jax.lax.stop_gradient(
            rewards + (1.0 - dones) * config.discount * next_q)

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = noises
        for i in range(config.flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / config.flow_steps)
            vels = networks.actor.apply(train_state.params_actor, observations, target_flow_actions, t, is_encoded=True, index=1)
            target_flow_actions = target_flow_actions + vels / config.flow_steps
        target_flow_actions = jnp.clip(target_flow_actions, -max_action, max_action)

        def _loss(
            params_value: flax.core.FrozenDict,
            params_critic: flax.core.FrozenDict,
            params_actor: flax.core.FrozenDict,
            scalars: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, Dict]:

            q1, q2 = networks.critic.apply(params_critic, observations, actions)
            loss_critic = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            pred = networks.actor.apply(params_actor, observations, x_t, t, index=1)
            # # BC flow loss
            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            actor_actions = networks.actor.apply(params_actor, observations, noises, None, index=0)
            # Distillation loss
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions = jnp.clip(actor_actions, -1, 1)
            qs1, qs2 = jax.lax.stop_gradient(
                networks.critic.apply(params_critic, observations, actor_actions)
            )
            q = jnp.mean(qs1 + qs2, axis=0)
            # Q loss
            q_loss = -q.mean()
            if config.normalize_q_loss:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
            
            loss_actor = bc_flow_loss + config.alpha * distill_loss + q_loss

            return loss_critic + loss_actor, {
                    "loss_critic": loss_critic,
                    "loss_actor": loss_actor
                }

        return grad_update(train_state, networks, _loss)