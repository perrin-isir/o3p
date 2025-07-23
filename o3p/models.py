# import haiku as hk
from jax import nn
import numpy as np
import distrax
from flax import linen as nn
import jax.numpy as jnp
# import jax
from typing import Sequence, Callable, Optional, Tuple
# import functools


class MLP(nn.Module):
    output_dim: int
    hidden_units: Sequence[int]
    hidden_activation: Callable = nn.relu
    output_activation: Optional[Callable] = None
    hidden_scale: float = 1.0
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        orthogonal_hidden = nn.initializers.orthogonal(scale=self.hidden_scale)
        orthogonal_output = nn.initializers.orthogonal(scale=self.output_scale)

        for units in self.hidden_units:
            x = nn.Dense(
                features=units,
                kernel_init=orthogonal_hidden
            )(x)
            x = self.hidden_activation(x)

        x = nn.Dense(
            features=self.output_dim,
            kernel_init=orthogonal_output
        )(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

# class MLP(hk.Module):
#     def __init__(
#         self,
#         output_dim,
#         hidden_units,
#         hidden_activation=nn.relu,
#         output_activation=None,
#         hidden_scale=1.,
#         output_scale=1.,
#     ):
#         super(MLP, self).__init__()
#         self.output_dim = output_dim
#         self.hidden_units = hidden_units
#         self.hidden_activation = hidden_activation
#         self.output_activation = output_activation
#         self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
#         self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

#     def __call__(self, x):
#         for _, unit in enumerate(self.hidden_units):
#             x = hk.Linear(unit, **self.hidden_kwargs)(x)
#             x = self.hidden_activation(x)
#         x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
#         if self.output_activation is not None:
#             x = self.output_activation(x)
#         return x


class ContinuousVFunction(nn.Module):
    num_values: int = 1
    hidden_units: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        outputs = []
        for i in range(self.num_values):
            mlp = MLP(
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )
            outputs.append(mlp(x))
        return outputs
    

# class ContinuousVFunction(hk.Module):

#     def __init__(
#         self,
#         num_values=1,
#         hidden_units=(256, 256),
#     ):
#         super(ContinuousVFunction, self).__init__()
#         self.num_critics = num_values
#         self.hidden_units = hidden_units

#     def __call__(self, x):
#         def _fn(x):
#             return MLP(
#                 1,
#                 self.hidden_units,
#                 hidden_activation=nn.relu,
#                 hidden_scale=np.sqrt(2),
#             )(x)

#         return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQFunction(nn.Module):
    num_critics: int = 2
    hidden_units: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)

        outputs = []
        for i in range(self.num_critics):
            q_net = MLP(
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )
            outputs.append(q_net(x))
        return outputs
    


# class ContinuousQFunction(hk.Module):

#     def __init__(
#         self,
#         num_critics=2,
#         hidden_units=(256, 256),
#     ):
#         super(ContinuousQFunction, self).__init__()
#         self.num_critics = num_critics
#         self.hidden_units = hidden_units

#     def __call__(self, s, a):
#         def _fn(x):
#             return MLP(
#                 1,
#                 self.hidden_units,
#                 hidden_activation=nn.relu,
#                 hidden_scale=np.sqrt(2),
#             )(x)

#         x = jnp.concatenate([s, a], axis=1)
#         return [_fn(x) for _ in range(self.num_critics)]


# class DeterministicPolicy(hk.Module):

#     def __init__(
#         self,
#         action_dim,
#         hidden_units=(256, 256),
#     ):
#         super(DeterministicPolicy, self).__init__()
#         self.action_dim = action_dim
#         self.hidden_units = hidden_units

#     def __call__(self, x):
#         return MLP(
#             self.action_dim,
#             self.hidden_units,
#             hidden_activation=nn.relu,
#             hidden_scale=np.sqrt(2),
#             output_activation=jnp.tanh,
#         )(x)

class DeterministicPolicy(nn.Module):
    action_dim: int
    hidden_units: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        return MLP(
            output_dim=self.action_dim,
            hidden_units=self.hidden_units,
            hidden_activation=nn.relu,
            hidden_scale=np.sqrt(2),
            output_activation=jnp.tanh,
        )(x)


# class StateDependentGaussianPolicy(hk.Module):

#     def __init__(
#         self,
#         action_dim,
#         hidden_units=(256, 256),
#         log_std_min=-10.0,
#         log_std_max=2.0,
#         clip_log_std=True,
#     ):
#         super(StateDependentGaussianPolicy, self).__init__()
#         self.action_dim = action_dim
#         self.hidden_units = hidden_units
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.clip_log_std = clip_log_std

#     def __call__(self, x):
#         if len(x.shape) < 2:
#             x = x[None]
#         x = MLP(
#             2 * self.action_dim,
#             self.hidden_units,
#             hidden_activation=nn.relu,
#             hidden_scale=np.sqrt(2),
#         )(x)
        
#         mean, log_std = jnp.split(x, 2, axis=1)
#         if self.clip_log_std:
#             log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
#         else:
#             log_std = self.log_std_min + \
#                 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1.0)
#         std = jnp.exp(log_std)
#         distribution = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
#         return distribution, mean


class StateDependentGaussianPolicy(nn.Module):
    action_dim: int
    hidden_units: Sequence[int] = (256, 256)
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    clip_log_std: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Distribution, jnp.ndarray]:
        if x.ndim < 2:
            x = jnp.expand_dims(x, axis=0)

        out = MLP(
            output_dim=2 * self.action_dim,
            hidden_units=self.hidden_units,
            hidden_activation=nn.relu,
            hidden_scale=np.sqrt(2),
        )(x)

        mean, log_std = jnp.split(out, 2, axis=-1)

        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            # Soft clipping via tanh squashing
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1.0)

        std = jnp.exp(log_std)
        distribution = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        return distribution, mean
    

# class StateDependentGaussianPolicyTanh(StateDependentGaussianPolicy):

#     def __init__(
#         self,
#         action_dim,
#         hidden_units=(256, 256),
#         log_std_min=-10.0,
#         log_std_max=2.0,
#         clip_log_std=True,
#     ):
#         super(StateDependentGaussianPolicyTanh, self).__init__(
#             action_dim, 
#             hidden_units=hidden_units, 
#             log_std_min=log_std_min, 
#             log_std_max=log_std_max, 
#             clip_log_std=clip_log_std
#         )

#     def __call__(self, x):
#         distribution, mean = super().__call__(x)
#         dist = distrax.Transformed(
#             distribution, 
#             distrax.Block(distrax.Tanh(), 1))
#         deterministic_action = jnp.tanh(mean)
#         # TODO: scale with max_action
#         return dist, deterministic_action
    

class StateDependentGaussianPolicyTanh(StateDependentGaussianPolicy):
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Distribution, jnp.ndarray]:
        # Get original Gaussian distribution and mean
        distribution, mean = super().__call__(x)

        # Apply tanh transformation to the distribution
        transformed_dist = distrax.Transformed(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        # Deterministic action via tanh of mean
        deterministic_action = jnp.tanh(mean)

        # TODO: scale with max_action if needed
        return transformed_dist, deterministic_action