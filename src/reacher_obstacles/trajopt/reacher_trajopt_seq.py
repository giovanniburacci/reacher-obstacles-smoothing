from typing import List, Sequence, Tuple
import numpy as np
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.trajopt.reacher_trajopt import ReacherTrajopt

class ReacherTrajoptSequential:

    def __init__(
            self,
            xml_path: str,
            obstacles_pos: Sequence[np.ndarray],
            nsteps: int,
            expid: str = "1a",
    ):
        self.xml_path = xml_path
        self.obstacles_pos = obstacles_pos
        self.nsteps = nsteps
        self.expid = expid

    def solve_sequence(
            self,
            targets: Sequence[np.ndarray],
            q0: np.ndarray,
            expids_per_stage: Sequence[str] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
            targets: list/sequence of target positions (2D or 3D). If 2D, the RobotModel already pads z to 0.015 internally.
            q0:      initial configuration for the first stage.
            expids_per_stage: optional list of expid (weights/clearances) per stage; if None, reuse self.expid for all stages.

        Returns:
            X_all, A_all, U_all: concatenated lists across all stages
        """
        X_all: List[np.ndarray] = []
        A_all: List[np.ndarray] = []
        U_all: List[np.ndarray] = []

        current_q = np.copy(q0)

        # Choose expid per stage if provided
        if expids_per_stage is None:
            expids = [self.expid] * len(targets)
        else:
            assert len(expids_per_stage) == len(targets), "expids_per_stage length must match targets"
            expids = expids_per_stage
        test_steps = [80,80,80]
        for i, (tgt, expid_i) in enumerate(zip(targets, expids)):
            # 1) Build a RobotModel for this target
            robot = RobotModel(
                xml_path=self.xml_path,
                target_pos=np.array(tgt, dtype=float),
                obstacle_pos=list(self.obstacles_pos),
                q0=current_q,
            )

            # 2) Reuse single-goal trajectory optimizer
            reacher_task = ReacherTrajopt(robot, T=test_steps[i], expid=expid_i)

            # 3) Solve from the current configuration
            X, A, U = reacher_task.solve(robot.qpos)  # Same call-signature already used

            # 4) Stitch results:
            #    - For the first stage, take all.
            #    - For subsequent stages, drop the first state to avoid duplicating the join.
            if i == 0:
                X_all.extend(X)
                A_all.extend(A)
                U_all.extend(U)
            else:
                X_all.extend(X[1:])     # Skip duplicate join state
                A_all.extend(A)         # Per-step arrays; full append
                U_all.extend(U)

            # Advance q0 to last state's configuration for the next stage
            current_q = X[-1][:robot.nq]
            print(f"Completed stage {i+1}/{len(targets)} to target {tgt}, new q0: {current_q}")

        return X_all, A_all, U_all
