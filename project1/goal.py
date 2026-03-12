import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        stateVec = state["stateVec"]
        x_t, y_t, theta_t = stateVec[7], stateVec[8], stateVec[9]

        # EE pose in world frame (base is at [-0.4, -0.2, 0] in world frame)
        pos_base, quat = self.jac_solver.forward_kinematics(stateVec[0:7])
        x_ee, y_ee = pos_base[0] - 0.4, pos_base[1] - 0.2

        # Project EE local axes onto world XY using rotation matrix from quat [qx, qy, qz, qw]
        qx, qy, qz, qw = quat
        # Local Z (EE approach/facing direction) → XY
        v_face = np.array([2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx)])
        v_face /= np.linalg.norm(v_face)
        # Local Y (gripper opening direction) → XY
        v_open = np.array([2*(qx*qy - qw*qz), 1 - 2*(qx*qx + qz*qz)])
        v_open /= np.linalg.norm(v_open)

        delta = np.array([x_t - x_ee, y_t - y_ee])
        d1 = abs(np.dot(delta, v_face))   # dist to EE face plane along approach axis
        d2 = abs(np.dot(delta, v_open))   # lateral offset along gripper opening axis

        # γ: min angular distance to any valid grasp angle (mod π/2 handles 180° gripper
        # symmetry + 2 pairs of cube sides; result is in [0, π/4])
        phi_ee = np.arctan2(v_face[1], v_face[0])
        diff = (phi_ee - theta_t) % (np.pi / 2)
        gamma = min(diff, np.pi / 2 - diff)

        return d1 < 0.01 and d2 < 0.02 and gamma < 0.2
        ##########################
        