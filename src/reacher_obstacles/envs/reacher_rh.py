__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from queue import Queue
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.envs.registration import register

class RewardHeuristic(Wrapper):

    def __init__(self, env,
                 reward_shaping=False,
                 time_beta: float = 1.0,
                 absorb_goal: bool = False,
                 **kwargs):
        super().__init__(env)
        self.bins = 9
        self.gmap = np.zeros((self.bins, self.bins))
        self.last_potential = 0.0
        self.reward_shaping = reward_shaping
        self.abs_gamma = 0.9

        self.time_beta = float(time_beta)
        self.absorb_goal = bool(absorb_goal)


    def _to_r_c(self, pos):
        # continuous x,y to discrete r,c
        c = int((pos[0] + 0.45) / 0.1)
        r = int((0.45 - pos[1]) / 0.1)
        return r, c
    # ---------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed)

        # compute goal map
        self.gmap = np.zeros((self.bins, self.bins)) - 1  # reset to -1

        # obstacles
        for jo in range(self.env.unwrapped.nobstacles):
            obsr, obsc = self._to_r_c(self.unwrapped.obstacles[jo])
            self.gmap[obsr, obsc] = 2 * self.bins

        # zero position
        zr, zc = self._to_r_c(np.array([0, 0]))
        self.gmap[zr, zc] = 2 * self.bins

        # optional U-obstacle map (unchanged)
        if self.env.unwrapped.uobstacle:
            uobstmap = np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 18, 18, 18, 0, 0, 0, 0],
                 [0, 0, 18, 0, 18, 0, 0, 0, 0],
                 [0, 0, 18, 0, 18, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
            )
            self.gmap = self.gmap + uobstmap

        tr, tc = self._to_r_c(self.env.unwrapped.target)
        assert self.gmap[tr, tc] == -1, "ERROR - reset model - target cell occupied by an obstacle!"
        self.gmap[tr, tc] = 0  # goal cell to 0

        # BFS fill with L1 distances
        q = Queue(maxsize=self.bins**2)
        q.put((tr, tc))
        while not q.empty():
            (r, c) = q.get()
            v = self.gmap[r, c]
            if r > 0 and self.gmap[r-1, c] == -1:
                self.gmap[r-1, c] = v+1; q.put((r-1, c))
            if r < self.bins-1 and self.gmap[r+1, c] == -1:
                self.gmap[r+1, c] = v+1; q.put((r+1, c))
            if c > 0 and self.gmap[r, c-1] == -1:
                self.gmap[r, c-1] = v+1; q.put((r, c-1))
            if c < self.bins-1 and self.gmap[r, c+1] == -1:
                self.gmap[r, c+1] = v+1; q.put((r, c+1))

        # reset potential to min value
        self.last_potential = 1.0 * (1.0 * self.abs_gamma**(self.bins*2) - 1.0)
        return obs, info

    def reward_heuristic(self):
        ftpos = self.env.unwrapped.data.body('fingertip').xpos
        ftr, ftc = self._to_r_c(ftpos)
        dvec = self.gmap[ftr, ftc]
        if not self.reward_shaping:
            rh = 1.0 * (1.0 * self.abs_gamma**dvec - 1.0)  # <= 0
        else:
            potential = 1.0 * (1.0 * self.abs_gamma**dvec - 1.0)
            rh = self.abs_gamma * potential - self.last_potential
            self.last_potential = potential
        return float(rh), int(dvec)

    def step(self, action):
        observation, reward, term, trunc, info = super().step(action)

        # remove original shapers from base env
        reward -= info["reward_dist"]
        reward -= info["reward_ctrl"]

        # RH value + principled time cost
        rh, dvec = self.reward_heuristic()
        dt = float(getattr(self.env.unwrapped, "dt"))
        time_penalty = - self.time_beta * dt

        reward += time_penalty + rh
        # reward += -1 + rh

        # lightweight near-goal shaping
        if dvec <= 3:
            reward += 0.5 + np.clip(info["reward_dist"], -0.5, 0.0)
            if dvec == 0:
                reward += 0.5

        # optional absorbing goal
        if self.absorb_goal and dvec == 0:
            term = True
            info["done_reason"] = "goal_absorbing"

        # log useful bits
        info["time_penalty"] = time_penalty
        info["reward_heuristic"] = rh
        info["dt"] = getattr(self.env.unwrapped, "dt", None)

        return observation, reward, term, trunc, info


def reacher_rh(**args):
    render_mode = args.get('render_mode', None)
    env = gym.make(args['envid'], render_mode=render_mode)
    env = RewardHeuristic(env,
                          reward_shaping=args.get('reward_shaping', False),
                          time_beta=args.get('time_beta', 1.0),
                          absorb_goal=args.get('absorb_goal', False))
    return env


def env_register(idreg, max_episode_steps=50, time_beta=1.0, absorb_goal=False):
    v = idreg.split('_')
    envid = v[0] + "_" + v[1]
    rs = (v[2] == 'rsV')
    register(
        id=idreg,
        entry_point="reacher_obstacles.envs.reacher_rh:reacher_rh",
        max_episode_steps=max_episode_steps,
        kwargs={
            'envid': envid,
            'reward_shaping': rs,
            'time_beta': time_beta,
            'absorb_goal': absorb_goal,
        }
    )


rew_list = ['rhV', 'rsV']

for conf in ["FT", "FTO1", "FTO1b", "FTO2", "FTO2b", "FTO2c", "FTU", "FTO3", "FTO3b"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}")
        env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["FTUO1", "FTUO1b", "FTUO2", "FTUO2b"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}", max_episode_steps=100)
        env_register(f"Reacher3-v6_{conf}_{rew}", max_episode_steps=100)

for conf in ["FTRO1", "FTRO2", "FTRO3"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}")
        env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["FT4RO1", "FT4RO2", "FT4RO3"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}")
        env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["RTO1", "RTO1b", "RTO2", "RTO2b", "RTO2c", "RTU", "RTUO1", "RTUO1b", "RTUO2"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}")
        env_register(f"Reacher3-v6_{conf}_{rew}")

for conf in ["RTRO1", "RTRO2", "RTRO3"]:
    for rew in rew_list:
        env_register(f"Reacher-v6_{conf}_{rew}")
        env_register(f"Reacher3-v6_{conf}_{rew}")
