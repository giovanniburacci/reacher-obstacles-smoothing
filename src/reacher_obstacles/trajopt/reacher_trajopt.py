import numpy as np
from reacher_obstacles.trajopt.robot_model import RobotModel
import casadi
from pinocchio import casadi as cpin


class ReacherTrajopt():
    
    def __init__(self, robot_model: RobotModel, T: int = 100, expid: str = "1a"):
        self.robot_model = robot_model
        self.T = T
        self.initialized = False
        self.expid = expid
        
        if expid == "1a":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1 
            self.w_target_term = 5
            self.obs_distance_link1 = 0.05
            self.obs_distance_link2 = 0.05
            self.obs_distance_fingertip = 0.05
        elif expid == "2a":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1 
            self.w_target_term = 20
            self.obs_distance_link1 = 0.03
            self.obs_distance_link2 = 0.03
            self.obs_distance_fingertip = 0.03
        elif expid == "3a":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1 
            self.w_target_term = 50
            self.obs_distance_link1 = 0.03
            self.obs_distance_link2 = 0.03
            self.obs_distance_fingertip = 0.04
        elif expid == "4a":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1 
            self.w_target_term = 20
            self.obs_distance_link1 = 0.03
            self.obs_distance_link2 = 0.03
            self.obs_distance_fingertip = 0.03
        elif expid == "5a":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1 
            self.w_target_term = 12
            self.obs_distance_link1 = 0.05
            self.obs_distance_link2 = 0.05
            self.obs_distance_fingertip = 0.05
        elif expid == "7b":
            self.w_vel = 1e-5
            self.w_acc = 1e-5
            self.w_target = 1e-1
            self.w_target_term = 5
            self.obs_distance_link1 = 0.05
            self.obs_distance_link2 = 0.05
            self.obs_distance_fingertip = 0.05
        else:
            raise ValueError(f"Unknown experiment id: {expid}")

        # Casadi Pinocchio Model
        self.cpin_model = cpin.Model(robot_model.pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.nx = robot_model.nq + robot_model.nv
        
        cx = casadi.SX.sym("x", self.nx, 1)
        self.cq = cx[:robot_model.nq]
        caq = casadi.SX.sym("a", robot_model.nv, 1)
        tauq = casadi.SX.sym("tau", robot_model.nv, 1)
        cpin.forwardKinematics(self.cpin_model, self.cpin_data, cx[:robot_model.nq], cx[robot_model.nq:], caq)
        cpin.updateFramePlacements(self.cpin_model, self.cpin_data)
        cpin.aba(self.cpin_model, self.cpin_data, cx[:robot_model.nq], cx[robot_model.nq:], self.robot_model.gear_gains * tauq)
        
        # Some functions needed in the casadi graph
        self.caba = casadi.Function(
            "aba",
            [cx, tauq],
            [self.cpin_data.ddq],
        )

        self.target_error = casadi.Function(
            "end_effector_error",
            [cx],
            [self.cpin_data.oMf[robot_model.final_ee_frame].translation - robot_model.target_pos],
        )
        
        self.cnext = casadi.Function(
            "next_state",
            [cx, caq],
            [
                casadi.vertcat(
                    casadi.fmin(
                        casadi.fmax(
                            cpin.integrate(self.cpin_model, cx[:robot_model.nq], cx[robot_model.nq:] * robot_model.dt),
                            robot_model.qpos_inf_limit
                        ),
                        robot_model.qpos_sup_limit
                    ),
                    cx[robot_model.nq:] + caq * robot_model.dt
                )
            ]
        )
                
        # Optimization problem
        self.opti = casadi.Opti()
        self.X = [self.opti.variable(self.nx) for t in range(T + 1)]
        self.Aq = [self.opti.variable(robot_model.nv) for t in range(T)]
        self.U = [self.opti.variable(robot_model.nv) for t in range(T)]
        
    
    def solve(self, q0: np.ndarray):
        self._set_initial_constraint(q0)
        cost = self._set_cost()
        self._set_constraints()
        self.opti.minimize(cost)
        self.opti.solver("ipopt")
        p_opts, s_opts = {"ipopt.print_level": 4, "ipopt.tol": 1e-3, "ipopt.max_iter": 500_000, "expand": True}, {}
        self.opti.solver("ipopt", p_opts, s_opts)
        
        try:
            self.opti.solve()
            sol_X = [self.opti.value(var_x) for var_x in self.X]
            sol_Aq = [self.opti.value(var_a) for var_a in self.Aq]
            sol_U = [self.opti.value(var_u) for var_u in self.U]
            print("COST:", self.opti.value(cost))
        except:
            raise Exception("ERROR in convergence")
        
        return sol_X, sol_Aq, sol_U
        
        
    def _set_cost(self) -> float:
        cost = 0
        
        for t in range(self.T):
            cost += self.w_vel * self.robot_model.dt * casadi.sumsqr(self.X[t][self.robot_model.nq:])
            cost += self.w_acc * self.robot_model.dt * casadi.sumsqr(self.Aq[t])
            cost += self.w_target * casadi.sumsqr(self.target_error(self.X[t]))
                
        cost += self.w_target_term * casadi.sumsqr(self.target_error(self.X[self.T]))
            
        return cost
        
    def _set_constraints(self):
        for t in range(self.T):
            self.opti.subject_to(self.cnext(self.X[t], self.Aq[t]) == self.X[t + 1])
            self.opti.subject_to(self.caba(self.X[t], self.U[t]) == self.Aq[t])
            self.opti.subject_to(self.U[t] <= 1.0)
            self.opti.subject_to(self.U[t] >= -1.0)
            
            for obstacle_idx in range(len(self.robot_model.obstacle_pos)):
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("body1"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_link1)
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("body2"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_link2)
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("fingertip"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_fingertip)

    def _ee_obstacle_distance(self, ee_frame_idx: int, obstacle_idx: int):
        obstacle_pos = self.robot_model.obstacle_pos[obstacle_idx]
        
        return casadi.Function(
            "obstacle_distance",
            [self.cq],
            [casadi.norm_2(self.cpin_data.oMf[ee_frame_idx].translation[:2] - obstacle_pos[:2])],
        )
        
    def _set_initial_constraint(self, q0: np.ndarray):
        self.opti.subject_to(self.X[0][:self.robot_model.nq] == q0)
        self.opti.subject_to(self.X[0][self.robot_model.nq:] == 0)       # zero initial velocity
        self.opti.subject_to(self.X[self.T][self.robot_model.nq:] == 0)  # zero terminal velocity
        self.initialized = True
    