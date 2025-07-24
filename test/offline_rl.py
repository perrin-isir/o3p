from o3p.algos import IQL, FQL
from o3p.envs import get_minari
from o3p.training import train_offline
from omegaconf import OmegaConf


conf_dict = OmegaConf.from_cli()
conf_dict.n_jitted_updates = 8
conf_dict.max_steps = 500_000
conf_dict.eval_interval = 50_000
conf_dict.log_interval = 50_000
conf_dict.save_train_state_interval = 50_000
conf_dict.eval_episodes = 100

algo = FQL(conf_dict)
env_name = "mujoco/halfcheetah/medium-v0"
# env_name = "D4RL/antmaze/umaze-v1"
(
    dataset, 
    env, 
    env_creator,
    reward_normalizing_factor, 
    obs_mean, 
    obs_std, 
    ref_max_score
    ) = get_minari(
        env_name, 
        normalize_reward=False, 
        normalize_obs=True)
train_offline(
    dataset, env_name, env, env_creator, algo, obs_mean, obs_std, ref_max_score)