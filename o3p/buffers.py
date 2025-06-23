from typing import NamedTuple, Optional
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from typing import Union, Dict, Any
import joblib
import os
from enum import Enum

from o3p.samplers import Sampler


class DataType(Enum):
    NUMPY = "data represented as numpy arrays"
    JAX = "data represented as jax.numpy arrays"


def get_datatype(x: Union[np.ndarray, jnp.ndarray]) -> DataType:
    if isinstance(x, jnp.ndarray):
        return DataType.JAX
    elif isinstance(x, np.ndarray):
        return DataType.NUMPY
    else:
        raise TypeError(f"{type(x)} not handled.")


def datatype_convert(
    x: Union[np.ndarray, jnp.ndarray, list, float],
    datatype: Union[DataType, None] = DataType.NUMPY,
) -> Union[np.ndarray, jnp.ndarray]:
    if datatype is None:
        return x
    elif datatype == DataType.NUMPY:
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
    elif datatype == DataType.JAX:
        if isinstance(x, jnp.ndarray):
            return x
        else:
            return jnp.array(x)


def proper_reshape(x):
    # Trick to add a singleton dimension only to flat arrays
    return x.reshape((x.shape + (1,1))[0:2])


def extract_from_batch(batch: Dict):
    return (
        proper_reshape(batch["observations"]),
        proper_reshape(batch["actions"]),
        proper_reshape(batch["rewards"]),
        proper_reshape(batch["next_observations"]),
        proper_reshape(batch["terminations"]),
        {
            key: proper_reshape(batch[key]) for \
                key in batch if key.startswith("infos.")
        }
    )

class Buffer(ABC):
    """Base class for buffers"""

    def __init__(
        self,
        buffer_size: int,
        sampler: Optional[Sampler]=None,
    ):
        self.size = buffer_size
        self.sampler = sampler
        self.current_size = 0
        self.next_insert_idx = 0
        self.buffers = {}

    @abstractmethod
    def insert(self, step: Dict[str, Any]):
        """Inserts a transition in the buffer"""
        pass

    @abstractmethod
    def sample(self, batch_size) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
        """Uses the sampler to returns a batch of transitions"""
        pass


class EpisodicBuffer(Buffer):
    """Base class for episodic buffers"""

    def __init__(
        self,
        max_episode_steps: int,
        buffer_size: int,
        sampler: Optional[Sampler]=None,
    ):
        super().__init__(int(buffer_size // max_episode_steps), sampler)
        self.T = max_episode_steps 


class DefaultBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int,
        sampler: Optional[Sampler]=None,
    ):
        super().__init__(buffer_size, sampler)
        self.dict_sizes = None
        self.keys = None
        self.subkeys = None
        self.zeros = None
        self.first_insert_done = False

    def _init_buffer(self, num_steps: int, transitions: Dict[str, Any]):
        self.dict_sizes = {}
        self.keys = list(transitions.keys())
        self.subkeys = {}
        for key in self.keys:
            if isinstance(transitions[key], dict):
                self.subkeys[key] = list(transitions[key].keys())
                for k in self.subkeys[key]:
                    t = proper_reshape(transitions[key][k].reshape(num_steps, -1))
                    assert len(t.shape) == 2
                    self.dict_sizes[key + "." + k] = t.shape[1]
            else:
                self.subkeys[key] = []
                t = proper_reshape(transitions[key].reshape(num_steps, -1))
                assert len(t.shape) == 2
                self.dict_sizes[key] = t.shape[1]
        for key in self.dict_sizes:
            self.buffers[key] = np.zeros([self.size, self.dict_sizes[key]])
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.first_insert_done = True

    def insert(self, num_steps: int, transitions: Dict[str, Any]):
        if not self.first_insert_done:
            self._init_buffer(num_steps, transitions)
        v = next(iter(transitions.values()))
        if type(v) == dict:
            v = next(iter(v.values()))
        num_steps = v.shape[0]
        idxs = self._get_storage_idx(inc=num_steps)
        for key in self.keys:
            if isinstance(transitions[key], dict):
                for k in transitions[key]:
                    self.buffers[key + "." + k][idxs, :] = datatype_convert(
                        transitions[key][k].reshape(num_steps, -1), DataType.NUMPY
                    ).reshape((num_steps, self.dict_sizes[key + "." + k]))
            else:
                self.buffers[key][idxs, :] = datatype_convert(
                    transitions[key].reshape(num_steps, -1), DataType.NUMPY
                ).reshape((num_steps, self.dict_sizes[key]))

    def pre_sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][: self.current_size]
        return temp_buffers

    def sample(self, batch_size):
        if self.sampler is not None:
            return self.sampler.sample(self.pre_sample(), batch_size)
        else:
            raise ValueError("This buffer has no sampler.")

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.next_insert_idx + inc <= self.size:
            idx = np.arange(self.next_insert_idx, self.next_insert_idx + inc)
        else:
            overflow = inc - (self.size - self.next_insert_idx)
            idx_a = np.arange(self.next_insert_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
        self.current_size = min(self.size, self.current_size + inc)
        self.next_insert_idx = (self.next_insert_idx + inc) % self.size
        return idx

    def to_jax(self):
        for key in self.buffers.keys():
            self.buffers[key] = jnp.asarray(self.buffers[key])

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        list_vars = [
            self.current_size,
            self.next_insert_idx,
            self.buffers,
            self.size,
            self.dict_sizes,
            self.keys,
            self.subkeys,
            self.first_insert_done,
        ]
        if self.sampler is not None:
            list_vars.append(self.sampler.rng)
        with open(os.path.join(directory, "buffer.joblib"), "wb") as f_:
            joblib.dump(list_vars, f_, compress=True)

    def load(self, directory: str):
        list_vars = joblib.load(os.path.join(directory, "buffer.joblib"))
        self.current_size = list_vars[0]
        self.next_insert_idx = list_vars[1]
        self.buffers = list_vars[2]
        self.size = list_vars[3]
        self.dict_sizes = list_vars[4]
        self.keys = list_vars[5]
        self.subkeys = list_vars[6]
        self.first_insert_done = list_vars[7]
        if len(list_vars) == 9:
            self.sampler.rng = list_vars[8]
        self.zeros = lambda i: np.zeros(i).astype("int")

#TODO: Other possibility???:  (not adding parallel transitions, we will have
# num_envs different buffers; when transitions are inserted: we put an 'episode_id' key
# with some random identifiers, and whenever a transition has a truncated or a terminated,
# we change the episode_id for future transitions, and we also give the info of the to-go
# length for all transitions of the episode; if a bunch of transitions are added at once,
# maybe a cumsum() will be useful

class DefaultEpisodicBuffer(EpisodicBuffer):
    def __init__(
        self,
        max_episode_steps: int,
        buffer_size: int,
        sampler: Optional[Sampler] = None,
    ):
        super().__init__(max_episode_steps, buffer_size, sampler)
        self.dict_sizes = None
        self.num_envs = None
        self.keys = None
        self.subkeys = None
        self.current_t = None
        self.zeros = None
        self.current_idxs = None
        self.first_insert_done = False

    def _init_buffer(self, num_envs: int, transitions: Dict[str, Any]):
        self.dict_sizes = {}
        self.keys = list(transitions.keys())
        self.subkeys = {}
        self.num_envs = num_envs
        for key in self.keys:
            if isinstance(transitions[key], dict):
                self.subkeys[key] = list(transitions[key].keys())
                for k in transitions[key]:
                    t = proper_reshape(transitions[key][k].reshape(num_envs, -1))
                    assert len(t.shape) == 2
                    self.dict_sizes[key + "." + k] = t.shape[1]
            else:
                self.subkeys[key] = []
                t = proper_reshape(transitions[key].reshape(num_envs, -1))
                assert len(t.shape) == 2
                self.dict_sizes[key] = t.shape[1]
        self.dict_sizes["episode_length"] = 1
        for key in self.dict_sizes:
            if key not in self.buffers:
                self.buffers[key] = np.zeros([self.size, self.T, self.dict_sizes[key]])
        self.current_t = np.zeros(self.num_envs).astype("int")
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.current_idxs = self._get_storage_idx(inc=self.num_envs)
        self.first_insert_done = True

    def _store_done(self, done):
        if done.max():
            where_done = np.where(datatype_convert(done, DataType.NUMPY) == 1)[0]
            k_envs = len(where_done)
            new_idxs = self._get_storage_idx(inc=k_envs)
            self.current_idxs[where_done] = new_idxs.reshape((1, len(new_idxs)))
            self.current_t[where_done] = 0

    def insert(self, num_envs: int, transitions: Dict[str, Any], done):
        if not self.first_insert_done:
            self._init_buffer(num_envs, transitions)
        else:
            assert(num_envs == self.num_envs)
        for key in self.keys:
            if isinstance(transitions[key], dict):
                for k in transitions[key]:
                    self.buffers[key + "." + k][
                        self.current_idxs, self.current_t, :
                    ] = datatype_convert(transitions[key][k], DataType.NUMPY).reshape(
                        (self.num_envs, self.dict_sizes[key + "." + k])
                    )
            else:
                self.buffers[key][
                    self.current_idxs, self.current_t, :
                ] = datatype_convert(transitions[key], DataType.NUMPY).reshape(
                    (self.num_envs, self.dict_sizes[key])
                )
        self.current_t += 1
        self.buffers["episode_length"][
            self.current_idxs, self.zeros(self.num_envs), :
        ] = self.current_t.reshape((self.num_envs, 1))
        self._store_done(done)

    def insert_episode(self, transitions: Dict[str, Any]):
        insert_idx = self.next_insert_idx
        self.next_insert_idx = (self.next_insert_idx + 1) % self.size
        self.current_size = min(self.size, self.current_size + 1)
        episode_length = transitions["rewards"].shape[0]
        if self.dict_sizes is None:
            self.dict_sizes = {}
        if "episode_length" not in self.dict_sizes:
            self.dict_sizes["episode_length"] = 1
            self.buffers["episode_length"] = np.zeros(
                [self.size, self.T, self.dict_sizes["episode_length"]])
        self.buffers["episode_length"][insert_idx, 0, :] = episode_length
        for key in list(transitions.keys()):
            if isinstance(transitions[key], dict):
                for k in transitions[key]:
                    t = proper_reshape(transitions[key][k].reshape(episode_length, -1))
                    assert len(t.shape) == 2
                    if key + "." + k not in self.dict_sizes:
                        self.dict_sizes[key + "." + k] = t.shape[1]
                    if key + "." + k not in self.buffers:
                        self.buffers[key + "." + k] = np.zeros(
                            [self.size, self.T, self.dict_sizes[key + "." + k]])
                    self.buffers[key + "." + k][
                        insert_idx, :episode_length, :
                    ] = datatype_convert(transitions[key][k], DataType.NUMPY).reshape(
                        (episode_length, self.dict_sizes[key + "." + k])
                    )
            else:
                t = proper_reshape(transitions[key].reshape(episode_length, -1))
                assert len(t.shape) == 2
                if key not in self.dict_sizes:
                    self.dict_sizes[key] = t.shape[1]
                if key not in self.buffers:
                    self.buffers[key] = np.zeros(
                        [self.size, self.T, self.dict_sizes[key]])
                self.buffers[key][
                    insert_idx, :episode_length, :
                ] = datatype_convert(transitions[key], DataType.NUMPY).reshape(
                    (episode_length, self.dict_sizes[key])
                )

    def pre_sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][: self.current_size]
        return temp_buffers

    def sample(self, batch_size):
        return self.sampler.sample(self.pre_sample(), batch_size)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.next_insert_idx + inc <= self.size:
            idx = np.arange(self.next_insert_idx, self.next_insert_idx + inc)
        else:
            overflow = inc - (self.size - self.next_insert_idx)
            idx_a = np.arange(self.next_insert_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
        self.buffers["episode_length"][idx, self.zeros(inc), :] = self.zeros(
            inc
        ).reshape((inc, 1))
        self.current_size = min(self.size, self.current_size + inc)
        self.next_insert_idx = (self.next_insert_idx + inc) % self.size
        return idx
    
    def to_jax(self):
        for key in self.buffers.keys():
            self.buffers[key] = jnp.asarray(self.buffers[key])
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        list_vars = [
            self.current_size,
            self.next_insert_idx,
            self.buffers,
            self.T,
            self.size,
            self.dict_sizes,
            self.num_envs,
            self.keys,
            self.subkeys,
            self.current_t,
            self.current_idxs,
            self.first_insert_done,
        ]
        if self.sampler is not None:
            list_vars.append(self.sampler.rng)
        with open(os.path.join(directory, "buffer.joblib"), "wb") as f_:
            joblib.dump(list_vars, f_, compress=True)

    def load(self, directory: str):
        list_vars = joblib.load(os.path.join(directory, "buffer.joblib"))
        self.current_size = list_vars[0]
        self.next_insert_idx = list_vars[1]
        self.buffers = list_vars[2]
        self.T = list_vars[3]
        self.size = list_vars[4]
        self.dict_sizes = list_vars[5]
        self.num_envs = list_vars[6]
        self.keys = list_vars[7]
        self.subkeys = list_vars[8]
        self.current_t = list_vars[9]
        self.current_idxs = list_vars[10]
        self.first_insert_done = list_vars[11]
        if len(list_vars) == 13:
            self.sampler.rng = list_vars[12]
        self.zeros = lambda i: np.zeros(i).astype("int")