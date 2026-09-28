import pybullet as p
import pybullet_data
import numpy as np
import config

class WalkerEnvironment:
    def __init__(self, render=config.RENDER_SIMULATION):
        mode = p.GUI if render else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadMJCF("humanoid/humanoid.mjcf")[0]
        self.joint_indices = [p.getJointInfo(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE]

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 1.3], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        for joint in self.joint_indices:
            p.resetJointState(self.robot_id, joint, 0.0, 0.0)
        return self.get_observation()

    def get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        obs = [pos[2], ori[0], ori[1], ori[2], linear_vel[0], linear_vel[1], linear_vel[2]]
        for i in range(min(5, len(self.joint_indices))):
            joint_pos, _, _, _ = p.getJointState(self.robot_id, self.joint_indices[i])
            obs.append(joint_pos)
        while len(obs) < config.INPUT_SIZE:
            obs.append(0.0)
        return np.array(obs[:config.INPUT_SIZE])

    def apply_action(self, actions):
        forces = actions * 50.0
        for i, joint in enumerate(self.joint_indices):
            if i < len(forces):
                p.setJointMotorControl2(self.robot_id, joint, p.TORQUE_CONTROL, force=forces[i])

    def step(self):
        p.stepSimulation()

    def get_fitness(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        if pos[2] < 0.4:
            return pos[0] - 2.0
        return pos[0]

    def close(self):
        p.disconnect()
