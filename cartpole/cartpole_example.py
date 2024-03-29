from omni.isaac.kit import SimulationApp


### Perform any omniverse imports here after the helper loads ###




def testPPO():
    from omni.isaac.gym.vec_env import VecEnvBase
    from cartpole_task import CartpoleTask
    from stable_baselines3 import PPO

    env = VecEnvBase(headless=False)
    task = CartpoleTask(name="Cartpole")
    env.set_task(task, backend="torch")

    # create agent from stable baselines
    model = PPO(
            "MlpPolicy",
            env,
            n_steps=1000,
            batch_size=1000,
            n_epochs=20,
            learning_rate=0.001,
            gamma=0.99,
            device="cuda:0",
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=1.0,
            verbose=1,
            tensorboard_log="./cartpole_tensorboard"
    )
    model.learn(total_timesteps=100000)
    model.save("ppo_cartpole")

    env.close()




if __name__=='__main__':
    # Simple example showing how to start and stop the helper
    simulation_app = SimulationApp({"headless": False})

    testPPO()