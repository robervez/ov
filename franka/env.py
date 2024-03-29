# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import gym
from gym import spaces
import numpy as np
import math

RVDEBUG = False


class FrankaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.franka import Franka
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path


        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            
            
        franka_asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd" 
        self.franka=self._my_world.scene.add(
            Franka(prim_path="/franka", name="franka", #usd_path=franka_asset_path,
                   position=np.array([0, 0.0, 0.020]),
                   orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                   )
        )
        
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.025]),
                size=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None

        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        # observation space -> 6 or 12 (if using orientation or not), inside a cube around the robot
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(6,), dtype=np.float32)
        self._my_world.reset()
        self._franka_articulation_controller = self.franka.get_articulation_controller()
        self.maxefforts = self._franka_articulation_controller.get_max_efforts()

        # no grippers
        self.maxefforts[-1] = 0
        self.maxefforts[-2] = 0

        print (self.maxefforts)

        # 9 dof
        self.action_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        #self.action_space = spaces.Box(
        #    np.array([-1.0, 0, 0 ,0 ,0,0,0,0,0]),
        #    np.array([ 1.0, 0, 0 ,0 ,0,0,0,0,0]))

        print(self.action_space )

        return

    def get_dt(self):
        return self._dt

    def step(self, action):

        previous_position = self.franka.end_effector.get_world_pose()[0]
        action[-1] = 0
        action[-2] = 0

        # final_action =np.concatenate((action, [0,0])) * 5.0
        final_action = np.multiply(action, self.maxefforts)
        if RVDEBUG:
            print("Action:", action)
            print("Final Action:", final_action)

        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction
            if RVDEBUG:
                print(action)
            #print(type(action))
            #print(action)
            #self.franka.apply_action (ArticulationAction(joint_efforts=final_action))
            self.franka.apply_action(ArticulationAction(joint_velocities=action * 0.5))

            #self.franka.set_joint_positions(self.startpositions[1:], joint_indices=[1, 2, 3, 4, 5, 6, 7, 8])

            self._my_world.step(render=False)
        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
            print("Done")
        goal_world_position, _ = self.goal.get_world_pose()
        current_position, current_orientation = self.franka.end_effector.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_position)

        # vorrei arrivarci con gripper rivolto verso il basso
        rewardOrientation = math.exp(-np.dot(current_orientation,self.startendEffectorOrientation))

        reward = previous_dist_to_goal - current_dist_to_goal + rewardOrientation

        return observations, reward, done, info

    def reset(self):
        print("RESET")
        self._my_world.reset()
        self._franka_articulation_controller.switch_control_mode('velocity')
        from omni.isaac.core.utils.types import ArticulationAction
        self.franka.apply_action(ArticulationAction(joint_velocities=np.zeros(9)))

        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = .80 * math.sqrt(np.random.rand()) + 0.20
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.025]))
        observations = self.get_observations()

        self.startpositions = self.franka.get_joint_positions()
        self.startendEffectorOrientation = self.franka.end_effector.get_world_pose()[1]

        return observations

    def get_observations(self):
        self._my_world.render()

        #print("Joint Positions after first reset: " + str(self.franka.get_joint_positions()))
        #print("pose after first reset: " + str(self.franka.get_world_pose()))
        #print("pose after first reset: " + str(self.franka.end_effector.get_world_pose()))

        mypos = self.franka.end_effector.get_world_pose()[0]
        goalpos= self.goal.get_world_pose()[0]

        #print(mypos)
        #print(type(mypos))
        #print(goalpos)
        #print(type(goalpos))

        # qui devo anche aggiungere il target....
        obs = np.concatenate((mypos,goalpos))
        return obs

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]



