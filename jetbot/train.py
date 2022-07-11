# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from env import JetBotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = "./cnn_policy"
# set headles to false to visualize training
my_env = JetBotEnv(headless=False)


policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[32, dict(pi=[64, 32], vf=[64, 32])])
policy = CnnPolicy
total_timesteps = 500000

if args.test is True:
    total_timesteps = 10000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=1000,
    #learning_rate=0.00025,
    learning_rate=0.001,
    gamma=0.9, #0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)



try:
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
except:
    pass

model.save(log_dir + "/jetbot_policy")

my_env.close()
