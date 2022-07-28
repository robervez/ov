import time

import gym
from gym import spaces
import numpy as np
import math

from omni.isaac.kit import SimulationApp

headless = False
dt= 1.0 / 60.0
simulation_app = SimulationApp({"headless": headless, "anti_aliasing": 0})

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path

my_world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
assets_root_path = get_assets_root_path()
import itertools

cnt = itertools.count(0)  # start counting at 0

print("step {}".format(next(cnt)))

franka_asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
franka = my_world.scene.add(
    Franka(prim_path="/franka", name="franka",  # usd_path=franka_asset_path,
           position=np.array([0, 0.0, 0.020]),
           orientation=np.array([1.0, 0.0, 0.0, 0.0]),
           )
)

print("step {}".format(next(cnt)))
goal = my_world.scene.add(
    VisualCuboid(
        prim_path="/new_cube_1",
        name="visual_cube",
        position=np.array([0.60, 0.30, 0.025]),
        size=np.array([0.05, 0.05, 0.05]),
        color=np.array([1.0, 0, 0]),
    )
)
print("step {}".format(next(cnt)))
my_world.reset()
print("step {}".format(next(cnt)))

from omni.isaac.core.utils.types import ArticulationAction
action = np.array([0.9, 0,0,0,0,0,0,0,0])
controller=franka.get_articulation_controller()
controller.switch_control_mode('velocity')
franka.apply_action(ArticulationAction(joint_velocities=action))
startpositions = franka.get_joint_positions()


print("step {}".format(next(cnt)))
for i in range(10*60):
    franka.set_joint_positions(startpositions[1:],joint_indices=[1,2,3,4,5,6,7,8])
    print("step {}".format(next(cnt)))
    time.sleep(dt)
    #print("step ", franka.get_applied_action())
    print("pos ", franka.get_joint_positions())
    my_world.step(render=True)

