__credits__ = ["Kallinteris-Andreas"]

import math,sys,time
from typing import Any, Dict, Optional, Tuple, Union
from inspect import getsourcefile
from os.path import abspath,dirname
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 1.5, }


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qx, qy, qz, qw])


MODEL_OBSTACLES = 3

CONFIGS = {
    'FT': {
        'target': (-0.075, 0.185),
    },
    'FTO1': {
        'target': (-0.075, -0.185),
        'obstacles': [(0.12, -0.15)],
    },
    'FTO2': {
        'target': (-0.075, 0.185),
        'obstacles': [(0.12,0.17),(0.0,-0.1)]
    },
    'FTO3': {
        'target': (-0.15, 0.17),
        'obstacles': [(0.0,0.2),(-0.02,0.1),(0.12,0.17)]
    },
    'FTO3b' : {
        'target': (-0.1, 0.1),
        'obstacles': [(0.12,0.2),(0,0.1),(-0.25,0.1)]
    },
    # used for hardest env in RM
    'FTO4b': {
        'target': (0.06, 0.6), # unused target, RM will set it
        'obstacles': [(0.15, 0.15), (-0.18, 0.18), (-0.18, -0.18)],
    },
}

class EnvConfig():

    def __init__(self,
                 njoints=2,
                 config_str = 'FT',
                 ):

        self.njoints = njoints
        self.config_str = config_str

        self.model_obstacles = MODEL_OBSTACLES
        self.uobstacle = 'U' in self.config_str

        self.seed = None
        self.np_random = None

    def random_target(self):
        g = None
        while True:
            dj = min(3,self.njoints)
            g = self.np_random.uniform(low=-0.1*dj, high=0.1*dj, size=2)
            if np.linalg.norm(g) < 0.1*self.njoints and \
                    np.linalg.norm(g) > 0.1:
                break
        return g

    def random_obstacle(self, d=3):
        d = max(d, 1.5)
        while True:
            dj = min(3,self.njoints)
            opos = self.np_random.uniform(low=-0.1*dj, high=0.1*dj, size=2)
            if np.linalg.norm(opos) > 0.15 and \
                    np.linalg.norm(opos) < 0.1*dj and \
                    (opos[0]<0 or abs(opos[1])>0.12) and \
                    np.linalg.norm(opos-self.target) > 0.15 and \
                    np.linalg.norm(opos-self.target) < d*0.15:
                break
        return opos


    def read_config(self):

        # obstacles
        self.obstacles = [None] * self.model_obstacles
        safe_x = 0.10*self.njoints + 0.05
        safe_y = 0.10*self.njoints + 0.05

        for jo in range(self.model_obstacles):
            self.obstacles[jo] = np.array([safe_x-jo*0.05, safe_y])

        if self.config_str in CONFIGS.keys():
            cfg = CONFIGS[self.config_str]
            self.target = cfg['target']

            self.nobstacles = 0
            if 'obstacles' in CONFIGS[self.config_str].keys():
                self.nobstacles = len(CONFIGS[self.config_str]['obstacles'])
                for jo in range(self.nobstacles):
                    self.obstacles[jo] = CONFIGS[self.config_str]['obstacles'][jo]

        self.uobstacle_pose = {
            'pos': np.array([-0.075, 0.25, 0.01]),
            'quat': euler_to_quaternion(-math.pi/2, 0, 0)  # don't know why yaw is the first param ???
        }


    def old_function(self):

        # target
        if self.fixed_target == 4:
            self.target = self.np_random.choice([np.array([0.1, 0.15]),
                                                 np.array([0, 0.15]),np.array([0.1, -0.15]),np.array([-0.1, -0.15])])
        elif self.fixed_target:
            self.target = np.array([-0.075, 0.175])
        else:
            self.target = self.random_target()
            # print(f"Random target: {self.target}")

        # obstacles
        self.obstacles = [None] * self.model_obstacles
        safe_x = 0.10*self.njoints + 0.05
        safe_y = 0.10*self.njoints + 0.05

        for jo in range(self.model_obstacles):
            self.obstacles[jo] = np.array([safe_x-jo*0.05, safe_y])

        if not self.fixed_obstacles:
            for jo in range(self.nobstacles):
                obsposok = False
                while not obsposok:
                    self.obstacles[jo] = self.random_obstacle(d=1.5+jo*0.5)
                    dog = np.linalg.norm(self.target - self.obstacles[jo])
                    obsposok = dog>0.15
                    for ojo in range(jo):
                        doo = np.linalg.norm(self.obstacles[ojo] - self.obstacles[jo])
                        obsposok = obsposok and doo>0.15


        else:
            if self.nobstacles==2:
                if self.uobstacle:
                    if self.obstacle_config == 'b':
                        self.obstacles[0] = np.array([0.17,-0.18])
                        self.obstacles[1] = np.array([-0.18,0.02])
                    else:
                        self.obstacles[0] = np.array([0.17,-0.18])   # FTUO2
                        self.obstacles[1] = np.array([-0.22,-0.12])
                else:
                    if self.obstacle_config == 'c':
                        self.obstacles[0] = np.array([-0.17,-0.16])
                        self.obstacles[1] = np.array([0.02,0.13])
                    elif self.obstacle_config == 'b':
                        self.obstacles[0] = np.array([-0.12,-0.17])  # FTO2b
                        self.obstacles[1] = np.array([0,0.1])
                    else:
                        self.obstacles[0] = np.array([0.12,0.17])  # FTO2
                        self.obstacles[1] = np.array([0,-0.1])

            elif self.nobstacles==1:

                if self.uobstacle:
                    if self.obstacle_config == 'b':
                        self.obstacles[0] = np.array([-0.17,-0.16])
                    else:
                        self.obstacles[0] = np.array([0.17,-0.18])   # FTUO1
                else:
                    if self.obstacle_config == 'b':
                        self.obstacles[0] = np.array([-0.13, 0.06])
                    else:
                        self.obstacles[0] = np.array([0.12,0.17])  # FTO1


        self.uobstacle_pose = {
            'pos': np.array([-0.075, 0.25, 0.01]),
            'quat': euler_to_quaternion(-math.pi/2, 0, 0)  # don't know why yaw is the first param ???
        }


        # check target and obtscle pos

        if not self.fixed_target:
            obst_target_ok = False
            while not obst_target_ok:
                # check target-obstacle cells
                obst_target_ok = True
                for jo in range(self.nobstacles):
                    dog = np.linalg.norm(self.target - self.obstacles[jo])
                    if dog<0.15:
                        obst_target_ok = False

                # choose new random goal
                if not obst_target_ok:
                    self.target = self.random_target()


    def get_config(self):
        return self.target, self.obstacles, self.uobstacle_pose


    def reset(self, seed):
        if self.np_random == None or seed != self.seed:
            self.seed = seed
            self.np_random = np.random.default_rng(self.seed)
        if self.config_str in CONFIGS.keys():
            self.read_config()
        else:
            self.old_function()

        return self.get_config()



class ReacherEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    "Reacher" is a two-jointed robot arm.
    The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.


    ## Action Space
    ```{figure} action_space_figures/reacher.png
    :name: reacher
    ```

    The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

    | Num | Action                                                                          | Control Min | Control Max |Name (in corresponding XML file)| Joint | Type (Unit)  |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
    | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1          | 1           | joint0                         | hinge | torque (N m) |
    | 1   | Torque applied at the second hinge (connecting the two links)                   | -1          | 1           | joint1                         | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *cos(qpos) (2 elements):* The cosine of the angles of the two arms.
    - *sin(qpos) (2 elements):* The sine of the angles of the two arms.
    - *qpos (2 elements):* The coordinates of the target.
    - *qvel (2 elements):* The angular velocities of the arms (their derivatives).
    - *xpos (2 elements):* The vector between the target and the reacher's.

    The observation space is a `Box(-Inf, Inf, (10,), float64)` where the elements are as follows:

    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
    | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
    | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
    | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
    | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
    | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                         | slide | position (m)             |
    | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
    | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | excluded | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d)        | -Inf | Inf | NA                               | slide | position (m)             |


    Most Gymnasium environments just return the positions and velocities of the joints in the `.xml` file as the state of the environment.
    In reacher, however, the state is created by combining only certain elements of the position and velocity and performing some function transformations on them.
    The `reacher.xml` contains these 4 joints:

    | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
    |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
    | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
    | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
    | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
    | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |


    ## Rewards
    The total reward is: ***reward*** *=* *reward_distance + reward_control*.

    - *reward_distance*:
    This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
    with a more negative value assigned if the reacher's *fingertip* is further away from the target.
    It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
    where $w_{near}$ is the `reward_near_weight` (default is $1$).
    - *reward_control*:
    A negative reward to penalize the walker for taking actions that are too large.
    It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
    where $w_{control}$ is the `reward_control_weight`. (default is $0.1$)

    `info` contains the individual reward terms.

    ## Starting State
    The initial position state of the reacher arm is $\mathcal{U}_{[-0.1 \times I_{2}, 0.1 \times I_{2}]}$.
    The position state of the goal is (permanently) $\mathcal{S}(0.2)$.
    The initial velocity state of the Reacher arm is $\mathcal{U}_{[-0.005 \times 1_{2}, 0.005 \times 1_{2}]}$.
    The velocity state of the object is (permanently) $0_2$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution and $\mathcal{S}$ is the uniform continuous spherical distribution.

    The default frame rate is $2$, with each frame lasting $0.01$, so *dt = 5 * 0.01 = 0.02*.


    ## Episode End
    ### Termination
    The Reacher never terminates.

    ### Truncation
    The default duration of an episode is 50 timesteps.


    ## Arguments
    Reacher provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Reacher-v5', xml_file=...)
    ```

    | Parameter               | Type       | Default       | Description                                              |
    |-------------------------|------------|---------------|----------------------------------------------------------|
    | `xml_file`              | **str**    |`"reacher.xml"`| Path to a MuJoCo model                                   |
    | `reward_dist_weight`    | **float**  | `1`           | Weight for _reward_dist_ term (see `Rewards` section)    |
    | `reward_control_weight` | **float**  | `0.1`         | Weight for _reward_control_ term (see `Rewards` section) |

    ## Version History
    * v6:
        - New reward function goal + dist + ctrl + vel
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `reward_distance` was based on the state before the physics step, now it is based on the state after the physics step (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).
        - Removed `"z - position_fingertip"` from the observation space since it is always 0 and therefore provides no useful information to the agent, this should result is slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
        - Added `xml_file` argument.
        - Added `reward_dist_weight`, `reward_control_weight` arguments to configure the reward function (defaults are effectively the same as in `v4`).
        - Fixed `info["reward_ctrl"]`  not being multiplied by the reward weight.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
            self,
            xml_file: str = "reacher.xml",
            frame_skip: int = 1,
            default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
            njoints = 2,
            ndim = 2,
            config_str = 'FT',
            #        fixed_target = False,
            #        fixed_obstacles = False,
            #        nobstacles = 0,
            #        obstacle_config = ' ',
            #        uobstacle = False,
            **kwargs,
    ):

        self.njoints = njoints
        self.ndim = ndim

        dd = dirname(getsourcefile(lambda:0))  # dir of this file

        if self.njoints == 3:
            xml_file = "reacher3.xml"
        if self.njoints == 5:
            xml_file = "marrtino.xml"

        xml_file_abs = dd+"/assets/"+xml_file

        #print(xml_file_abs)

        utils.EzPickle.__init__(
            self,
            xml_file_abs,
            frame_skip,
            default_camera_config,
            **kwargs,
        )

        self.config_str = config_str

        #self.fixed_target = fixed_target
        #self.fixed_obstacles = fixed_obstacles
        #self.nobstacles = nobstacles
        #self.obstacle_config = obstacle_config
        #self.uobstacle = uobstacle
        # if the env is deterministic
        self.deterministic = True

        self.model_obstacles = MODEL_OBSTACLES
        self.nobstacles = 0
        if 'obstacles' in CONFIGS[self.config_str].keys():
            self.nobstacles = len(CONFIGS[self.config_str]['obstacles'])

        obssize = self.njoints * 4 + self.ndim * 3 + self.nobstacles * 2
        # joints pos & vel + fingertip pos&or + target + obstacles        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obssize,), dtype=np.float64)

        self.env_config = EnvConfig(
            self.njoints,
            self.config_str,
            #fixed_target=self.fixed_target,
            #fixed_obstacles=self.fixed_obstacles,
            #nobstacles = self.nobstacles,
            #obstacle_config = self.obstacle_config,
            #uobstacle = self.uobstacle,
        )

        MujocoEnv.__init__(
            self,
            xml_file_abs,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )


        self.max_vel = 10

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        if self.render_mode == "human":
            # hide menu
            self.mujoco_renderer._get_viewer("human")._hide_menu = True


    # reset override not needed in gym v. 1.0.0, use:
    # self.seed = self.np_random_seed
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.seed = seed
        return super().reset(seed=seed)


    def reset_model(self):

        # robot init 
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.env_config.reset(self.seed)
        self.target = self.env_config.target
        self.obstacles = self.env_config.obstacles
        self.uobstacle = self.env_config.uobstacle

        if self.env_config.uobstacle:
            # u-shape obstacle
            b = self.data.model.body("obstacleu")
            b.pos = self.env_config.uobstacle_pose['pos']
            b.quat = self.env_config.uobstacle_pose['quat']

        # max possible distance between ee and target
        self.max_dist = 0.22*min(3,self.njoints)

        if not self.deterministic:
            qpos[0:self.njoints] += self.np_random.uniform(low=-0.1, high=0.1, size=self.njoints)
            qvel[0:self.njoints] += self.np_random.uniform(
                low=-0.005, high=0.005, size=self.njoints)

        pobs = - self.model_obstacles*2
        pgoal = pobs - 2

        if pgoal+2<0:
            qpos[pgoal:pgoal+2] = self.target
        else:
            qpos[pgoal:] = self.target
        for jo in range(self.model_obstacles):
            if pobs+jo*2+2<0:
                qpos[pobs+jo*2:pobs+jo*2+2] = self.obstacles[jo]
            else:
                qpos[pobs+jo*2:] = self.obstacles[jo]
        qvel[pgoal:] = 0

        # print(f"pos: {qpos} - vel {qvel}")

        self.set_state(qpos, qvel)

        #print(self.data.body("target"))
        assert (self.data.body("target").cvel==0).all(), "Target vel non zero!!!"

        # any hit during this episode
        self.episode_hit = False

        return self._get_obs()



    def step(self, action):

        #theta = self.data.qpos.flatten()[:self.njoints] # joints pos
        #jvel = self.data.qvel.flatten()[:self.njoints]  # joints vel

        self.do_simulation(action, self.frame_skip)

        termination = False

        self.hit = 0
        if self.detect_contact():
            self.hit = 1
            self.episode_hit = True
            termination = True

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if termination:
            reward -= 100

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, termination, False, info



    def _get_obs(self):
        theta = self.data.qpos.flatten()[:self.njoints]
        if self.ndim==2:
            quat = np.concatenate( [
                self.data.body("fingertip").xquat[0:1],  # fingertip 2D orientation
                self.data.body("fingertip").xquat[3:4] ] )
        else:
            quat = self.data.body("fingertip").xquat    # fingertip 3D orientation

        obs = np.concatenate(
            [
                theta,              # joint angles
                np.cos(theta),      # cos joint angles
                np.sin(theta),      # sin joint angles
                self.data.qvel.flatten()[0:self.njoints],   # joint velocities
                self.data.body("fingertip").xpos[0:self.ndim],      # fingertip position
                quat,               # fingertip orientation
                self.data.body("target").xpos[:2],         # target 2D position
            ]
        )

        if self.nobstacles>=1:
            obs = np.concatenate([obs,
                                  self.data.body("obstacle1").xpos[:2],
                                  ])
        if self.nobstacles>=2:
            obs = np.concatenate([obs,
                                  self.data.body("obstacle2").xpos[:2],
                                  ])
        if self.nobstacles>=3:
            obs = np.concatenate([obs,
                                  self.data.body("obstacle3").xpos[:2],
                                  ])


        return obs


    def _get_rew(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")  # diff to target
        vec_norm = np.linalg.norm(vec)

        jvel = self.data.qvel.flatten()[:self.njoints]  # joints vel
        jvel_norm = np.linalg.norm(jvel) # joints vel norm

        self.dist_tolerance = 0.08
        self.jvel_tolerance = 0.8

        # goal reached: extra reward
        reward_goal = 1.0 if (vec_norm<self.dist_tolerance and jvel_norm<self.jvel_tolerance and not self.episode_hit) else 0


        # Reacher-v4 original reward + goal reward + hit penalty
        reward_dist = -vec_norm
        reward_ctrl = -np.square(action).sum()
        reward_hit = -self.hit
        reward = reward_dist + reward_ctrl + reward_goal + reward_hit
        reward_info = {
            "dist": vec_norm,
            "hit": self.hit,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_goal": reward_goal,
            "reward_hit": reward_hit,
        }

        return reward, reward_info


    def detect_contact(self):
        excluded = [('link0','root'), ('jcolor0','root'), ('ground','target'), ('ground','ball')]
        r = False
        d = self.data
        for coni in range(d.ncon):
            con = d.contact[coni]
            geom1name = d.model.geom(con.geom1).name
            geom2name = d.model.geom(con.geom2).name
            if (geom1name,geom2name) not in excluded:
                # print(f" -- contact {geom1name} {geom2name}")
                r = True
        return r

    def set_target(self, pos_xyz):
        """
        Update the active target used by shaping/obs and move the target body in MuJoCo.
        Expects a 3D position [x, y, z]; shaping uses x,y.
        """
        import numpy as np
        pos = np.asarray(pos_xyz, float).reshape(3)

        # 1) update the 2D target used by reward/obs
        self.target = pos[:2]

        # 2) move the two slide DOFs that place the 'target' body
        qpos = self.data.qpos.copy().flatten()
        pobs = - self.model_obstacles * 2
        pgoal = pobs - 2
        qpos[pgoal:pgoal+2] = self.target
        self.set_state(qpos, self.data.qvel.copy().flatten())

        # 3) best-effort: if a site named 'target' exists, move it for visualization
        try:
            sid = self.model.site("target").id
            self.model.site_pos[sid] = pos
            self.sim.forward()
        except Exception:
            pass



# env registration 

from gymnasium.envs.registration import register

def reacher_v6(**args):
    return ReacherEnv(**args)

def env_register(idreg, max_episode_steps=300):
    vid = idreg.split('_')
    cfstr = vid[1]
    nj = 2
    ndim = 2
    if vid[0] == 'Reacher3-v6':
        nj = 3
    elif vid[0] == 'MARRtinoArm':
        nj = 5
        ndim = 3

    register(id=idreg,
             entry_point="reacher_obstacles.envs.reacher_v6:reacher_v6",
             max_episode_steps=max_episode_steps,
             kwargs =  {
                 'njoints': nj,
                 'ndim': ndim,
                 'config_str': cfstr,
             } )



for conf in CONFIGS.keys():
    env_register(f"Reacher-v6_{conf}")
    env_register(f"Reacher3-v6_{conf}")
    env_register(f"MARRtinoArm_{conf}")