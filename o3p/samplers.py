from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from typing import Union, Dict


class Sampler(ABC):
    def __init__(self, *, seed: Union[int, None] = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(
        self,
        buffer,
        batch_size: int,
    ) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
        """Return a batch of transitions"""
        pass


class DefaultSampler(Sampler):
    def __init__(self, *, seed: Union[int, None] = None):
        super().__init__(seed=seed)

    def sample(
        self,
        buffer: Dict[str, Union[np.ndarray]],
        batch_size: int,
    ) -> Dict[str, Union[np.ndarray]]:
        buffer_size = next(iter(buffer.values())).shape[0]
        idxs = self.rng.choice(
            buffer_size,
            size=batch_size,
            replace=True,
        )
        transitions = {key: buffer[key][idxs] for key in buffer.keys()}
        return transitions


class DefaultEpisodicSampler(Sampler):
    def __init__(self, *, seed: Union[int, None] = None):
        super().__init__(seed=seed)

    def sample(
        self,
        buffers: Dict[str, Union[np.ndarray]],
        batch_size: int,
    ) -> Dict[str, Union[np.ndarray]]:
        rollout_batch_size = buffers["episode_length"].shape[0]
        episode_idxs = self.rng.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p=np.asarray(buffers["episode_length"][:, 0, 0], dtype='float64')
                / np.asarray(buffers["episode_length"][:, 0, 0], dtype='float64').sum(),
        )
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
        t_samples = self.rng.integers(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        return transitions


class HER(Sampler):
    def __init__(
        self,
        compute_reward,
        seed: Union[int, None] = None
    ):
        super().__init__(seed=seed)
        self.future_p = 0.8
        self.reward_func = compute_reward

    def sample(self, buffers, batch_size_in_transitions):
        rollout_batch_size = buffers["episode_length"].shape[0]
        batch_size = batch_size_in_transitions
        # select rollouts and steps
        episode_idxs = self.rng.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p = np.asarray(buffers["episode_length"][:, 0, 0], dtype='float64')
                / np.asarray(buffers["episode_length"][:, 0, 0], dtype='float64').sum(),
        )
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
        
        t_samples = self.rng.integers(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        # HER indexes
        her_indexes = np.where(self.rng.uniform(size=batch_size) < self.future_p)

        future_offset = self.rng.uniform(size=batch_size) * (
            t_max_episodes - t_samples
        )
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace desired goal with achieved goal
        future_ag = buffers["observations.achieved_goal"][
            episode_idxs[her_indexes], future_t
        ]

        transitions["observations.desired_goal"][her_indexes] = future_ag
        transitions["next_observations.desired_goal"][her_indexes] = future_ag

        # recomputing rewards
        transitions["rewards"] = np.expand_dims(
            self.reward_func(
                transitions["observations.achieved_goal"],
                transitions["observations.desired_goal"],
                {}  #TODO: retrieve info which should be put here
            ),
            1,
        )
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        transitions.pop("episode_length")
        return transitions