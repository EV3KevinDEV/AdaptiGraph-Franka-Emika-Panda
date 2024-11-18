import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import panda_py


class PandaRobot:
    def __init__(
        self,
        hostname="172.16.0.2",
        init_pose=None,
        init_joint_angles=None,
        speed_factor=0.2,  # Speed factor (0 to 1)
    ):
        self.hostname = hostname
        self.init_pose = init_pose
        self.init_joint_angles = init_joint_angles
        self.speed_factor = speed_factor
        self.alive = True
        self.panda = panda_py.Panda(hostname=self.hostname)
        self._robot_init()

    def _robot_init(self):
        # Reset any errors and set default behavior
        self.panda.recover()
        self.panda.set_default_behavior()

    def move_to_pose(self, pose, wait=True, ignore_error=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        # Convert pose from xArm format [X(mm), Y(mm), Z(mm), roll(deg), pitch(deg), yaw(deg)]
        # to Panda format: position (meters) and orientation (quaternion)
        position = np.array(pose[:3]) / 1000.0  # Convert mm to meters
        roll, pitch, yaw = np.deg2rad(pose[3:6])  # Convert degrees to radians
        rotation = R.from_euler('xyz', [roll, pitch, yaw])
        orientation = rotation.as_quat()  # Quaternion as [x, y, z, w]
        # The panda_py library expects orientation as [x, y, z, w] (scalar last)
        success = self.panda.move_to_pose(
            position=position,
            orientation=orientation,
            speed_factor=self.speed_factor,
        )
        if not ignore_error and not success:
            raise ValueError("move_to_pose Error")
        return True

    def get_current_pose(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        pose_matrix = self.panda.get_pose()
        # Extract position and rotation matrix
        position = pose_matrix[:3, 3]  # Position in meters
        rotation_matrix = pose_matrix[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
        # Convert position to millimeters for consistency with xArm format
        pose = np.hstack((position * 1000.0, [roll, pitch, yaw]))
        return pose

    def get_current_joint(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        state = self.panda.get_state()
        joint_angles = state.q  # Joint positions in radians
        joint_angles_deg = np.rad2deg(joint_angles)  # Convert to degrees
        return joint_angles_deg

    def reset(self, wait=True):
        if self.init_joint_angles is not None:
            # Move to initial joint angles
            init_angles_rad = np.deg2rad(self.init_joint_angles)
            success = self.panda.move_to_joint_position(
                positions=init_angles_rad,
                speed_factor=self.speed_factor,
            )
            if not success:
                raise ValueError("reset Error")
        elif self.init_pose is not None:
            # Move to initial pose
            self.move_to_pose(self.init_pose, wait=wait)
        else:
            # Move to default start position
            success = self.panda.move_to_start(speed_factor=self.speed_factor)
            if not success:
                raise ValueError("reset Error")

    def open_gripper(self, wait=True):
        raise NotImplementedError("Gripper control is not implemented in this version.")

    def close_gripper(self, wait=True):
        raise NotImplementedError("Gripper control is not implemented in this version.")

    def get_gripper_state(self):
        raise NotImplementedError("Gripper control is not implemented in this version.")

    @property
    def is_alive(self):
        try:
            self.panda.raise_error()
            return True
        except RuntimeError:
            return False
