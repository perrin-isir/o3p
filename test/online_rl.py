from omegaconf import OmegaConf
from o3p.algos import SAC
from o3p.algos import SAC
from o3p.training import train_online
from omegaconf import OmegaConf

conf_dict = OmegaConf.from_cli()
conf_dict.n_jitted_updates = 8
conf_dict.max_steps = 500_000
conf_dict.eval_interval = 10000
conf_dict.log_interval = 10000
conf_dict.save_train_state_interval = 10000
conf_dict.eval_episodes = 8

algo = SAC(conf_dict)

train_state, networks = train_online("HalfCheetah-v5", algo, parallel_envs=4)