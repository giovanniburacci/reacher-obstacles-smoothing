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
        self.gmap = np.zeros((self.bins, self.bins))  # base mask gets built on reset
        self.last_potential = 0.0
        self.reward_shaping = bool(reward_shaping)
        self.abs_gamma = 0.9

        self.time_beta = float(time_beta)
        self.absorb_goal = bool(absorb_goal)

        # route goals cache (no per-goal maps exposed)
        self._route_goals = None

    # ---------------------------
    # Helpers
    # ---------------------------

    def _to_r_c(self, pos):
        """continuous (x,y) -> discrete (r,c) in [0..8]."""
        c = int((pos[0] + 0.45) / 0.1)
        r = int((0.45 - pos[1]) / 0.1)
        r = np.clip(r, 0, self.bins - 1)
        c = np.clip(c, 0, self.bins - 1)
        return int(r), int(c)

    def _build_base_mask(self):
        """
        Build base mask into self.gmap:
          free cells  = -1
          obstacles   = large positive (2*bins)
        """
        self.gmap = np.zeros((self.bins, self.bins)) - 1

        # obstacles
        for jo in range(self.env.unwrapped.nobstacles):
            r, c = self._to_r_c(self.unwrapped.obstacles[jo])
            self.gmap[r, c] = 2 * self.bins

        # origin
        r0, c0 = self._to_r_c(np.array([0, 0]))
        self.gmap[r0, c0] = 2 * self.bins

        # optional U overlay
        if getattr(self.env.unwrapped, "uobstacle", False):
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

    # ---------------------------
    # Tiny BFS for distances
    # ---------------------------

    def _bfs_distance(self, start_rc, goal_rc):
        """
        Return obstacle-aware L1 shortest-path distance between two cells
        using the current base mask in `self.gmap` (free == -1).
        """
        sr, sc = start_rc
        gr, gc = goal_rc
        if sr == gr and sc == gc:
            return 0.0

        # early exit if start or goal blocked
        if self.gmap[gr, gc] != -1:
            return float(2 * self.bins)
        if self.gmap[sr, sc] != -1:
            return float(2 * self.bins)

        visited = np.zeros_like(self.gmap, dtype=bool)
        q = Queue(maxsize=self.bins * self.bins)
        q.put((sr, sc, 0))
        visited[sr, sc] = True

        while not q.empty():
            r, c, d = q.get()
            # 4-neighbors
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if rr < 0 or rr >= self.bins or cc < 0 or cc >= self.bins:
                    continue
                if visited[rr, cc]:
                    continue
                if self.gmap[rr, cc] != -1:
                    continue  # obstacle
                if rr == gr and cc == gc:
                    return float(d + 1)
                visited[rr, cc] = True
                q.put((rr, cc, d + 1))

        return float(2 * self.bins)  # no path

    # ---------------------------
    # RM route helpers
    # ---------------------------

    def _collect_route_goals(self):
        """
        Collect an *ordered* list of route goals from the env's waypoints.
        Always convert to [x, y]. Fallback to single target if none found.
        """
        U = self.env.unwrapped
        try:
            # dict_values preserves insertion order (Py3.7+)
            goals_vals = getattr(U, 'waypoints', {}).values()
            goals = [np.array(g, dtype=float).reshape(-1)[:2] for g in goals_vals]
            # print(f"[RH] Collected {goals} route goals from waypoints.", flush=True)
            if goals:
                return goals
        except Exception as e:
            print(f"[RH] Warning: failed to collect waypoints ({e}); using single target fallback.", flush=True)

        # fallback: single target as [x, y]
        tgt_xy = np.array(U.target, dtype=float).reshape(-1)[:2]
        # print("[RH] Warning: no route goals found; using single target fallback.", flush=True)
        return [tgt_xy]


    def _stage_index_from_active_target(self):
        """Match env.unwrapped.target to the route goal index; fallback to nearest."""
        tgt = np.array(self.env.unwrapped.target, dtype=float)
        for k, g in enumerate(self._route_goals):
            if np.allclose(tgt, g, atol=1e-9):
                return k
        dists = [np.linalg.norm(tgt - g) for g in self._route_goals]
        return int(np.argmin(dists))

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed)

        # build base grid (free/obstacles), no goal fill
        self._build_base_mask()

        # cache route list (no maps)
        self._route_goals = self._collect_route_goals()
        # print(f"[RH] reset route with {len(self._route_goals)} goal(s).", flush=True)

        # initialize potential baseline (unchanged pattern)
        self.last_potential = 1.0 * (1.0 * self.abs_gamma**(self.bins * 2) - 1.0)
        return obs, info

    def _distance_to_completion(self, cell_rc):
        """
        Compute D_k(c) = d(c, G_k) + sum_{i=k}^{N-1} d(G_i, G_{i+1})
        entirely on the fly via BFS distances on `self.gmap`.
        """
        # current RM stage
        k = self._stage_index_from_active_target()

        # cell -> current goal
        gk = self._route_goals[k]
        tr_k, tc_k = self._to_r_c(gk)
        # print(f"[RH] Computing D_{k} from cell {cell_rc} to goal {gk.tolist()} at ({tr_k},{tc_k})", flush=True)
        Lkc = self._bfs_distance(cell_rc, (tr_k, tc_k))

        # tail over remaining consecutive goals
        tail = 0.0
        for i in range(k, len(self._route_goals) - 1):
            gi = self._route_goals[i]
            gi1 = self._route_goals[i + 1]
            ri, ci = self._to_r_c(gi)
            ri1, ci1 = self._to_r_c(gi1)
            tail += self._bfs_distance((ri, ci), (ri1, ci1))

        return (Lkc + tail), int(Lkc), int(k), k == len(self._route_goals) - 1

    def reward_heuristic(self):
        # fingertip -> discrete cell
        ftpos = self.env.unwrapped.data.body('fingertip').xpos
        ftr, ftc = self._to_r_c(ftpos)

        # route-aware abstract distance to completion
        Dk, dvec_local, k, is_final = self._distance_to_completion((ftr, ftc))
        # print(f"[RH] D_{k} at cell ({ftr},{ftc}) = {Dk} (local d={dvec_local})", flush=True)
        dvec = Dk

        # --------- UNCHANGED reward shaping logic ----------
        if not self.reward_shaping:
            # RHV: value function applied to 9x9 grid with +1 reward at goal
            rh = 1.0 * (1.0 * self.abs_gamma**(dvec - 1.0))  # <= 0   (same as your file)
        else:
            potential = 1.0 * (1.0 * self.abs_gamma**dvec - 1.0)
            rh = self.abs_gamma * potential - self.last_potential
            self.last_potential = potential
        # ---------------------------------------------------

        return float(rh), int(dvec_local), int(k), is_final

    def step(self, action):
        # standard env step
        observation, reward, term, trunc, info = super().step(action)

        # remove base env shapers
        reward -= info.get("reward_dist", 0.0)
        reward -= info.get("reward_ctrl", 0.0)

        # RH value + time penalty
        rh, dvec, k, is_final = self.reward_heuristic()
        # print(f"------------------------------------------------------------------------", flush=True)
        dt = float(getattr(self.env.unwrapped, "dt"))
        time_penalty = - self.time_beta * dt
        reward += time_penalty + rh

        # print(f"[RH] reward heuristic: {rh} (dvec={dvec}) reward dist: {info.get('reward_dist', 0.0)}", flush=True)
        # near-goal tweak
        if dvec <= 1 and is_final:
            reward += 0.5 + np.clip(info.get("reward_dist", 0.0), -0.5, 0.0)
            if dvec == 0:
                print(f"[RH] reached final goal at step!", flush=True)
                reward += 5

        # print(f"[RH] total reward: {reward} (time_penalty: {time_penalty})", flush=True)
        # print(f"------------------------------------------------------------------------", flush=True)

        # optional absorbing goal
        if self.absorb_goal and dvec == 0:
            term = True
            info = dict(info or {})
            info["done_reason"] = "goal_absorbing"

        # debug/info
        info = dict(info or {})
        info["rh_stage"] = k
        info["rh_goal"] = np.array(self.env.unwrapped.target, dtype=float).tolist()
        return observation, reward, term, trunc, info


def reacher_rh(**args):
    render_mode = args.get('render_mode', None)
    env = gym.make(args['envid'], render_mode=render_mode)
    env = RewardHeuristic(env,
                          reward_shaping=args.get('reward_shaping', False),
                          time_beta=args.get('time_beta', 1.0),
                          absorb_goal=args.get('absorb_goal', False))
    return env


def env_register(idreg, max_episode_steps=100, time_beta=1.0, absorb_goal=False):
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

for conf in ["FT", "FTO1", "FTO1b", "FTO2", "FTO2b", "FTO2c", "FTU", "FTO3", "FTO3b", "FTO4b"]:
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
