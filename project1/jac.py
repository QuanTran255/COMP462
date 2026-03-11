import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc


class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        J = np.zeros(shape=(6, 7))
        eps = 1e-6

        q = np.array(joint_values, dtype=float)
        pos0, quat0 = self.forward_kinematics(q)
        _, quat0_inv = self.bullet_client.invertTransform([0.0, 0.0, 0.0], quat0.tolist())

        for j in range(7):
            q_pert = q.copy()
            q_pert[j] += eps

            pos1, quat1 = self.forward_kinematics(q_pert)

            # Linear velocity component
            J[0:3, j] = (pos1 - pos0) / eps

            # Angular velocity component via relative quaternion
            _, q_rel = self.bullet_client.multiplyTransforms([0.0, 0.0, 0.0], quat1.tolist(),
                                                             [0.0, 0.0, 0.0], quat0_inv)
            q_rel = np.array(q_rel)
            q_rel /= np.linalg.norm(q_rel)
            if q_rel[3] < 0.0:
                q_rel = -q_rel

            vec_norm = np.linalg.norm(q_rel[:3])
            if vec_norm < 1e-12:
                rotvec = np.zeros(3)
            else:
                angle = 2.0 * np.arctan2(vec_norm, q_rel[3])
                rotvec = (q_rel[:3] / vec_norm) * angle
            J[3:6, j] = rotvec / eps

        ##########################
        return J