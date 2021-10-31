
from abc import ABC, abstractmethod
import numpy as np
from state.state import StateHypothesis


class MotionModel(ABC):
    @abstractmethod
    def propagate(self, state):
        raise NotImplementedError


class VelocityMotionModel(MotionModel):
    def __init__(self):
        pass

    def propagate(self, state, control_input):
        dt = state.timestep
        dw = 1e-11/dt
        x = state.state[0]
        y = state.state[1]
        fi = state.state[2]
        v = control_input[0]
        w = control_input[1] + dw  # avoid divison by 0

        x_next = x - v/w*np.sin(fi) + v/w*np.sin(fi + w*dt)
        y_next = y + v/w*np.cos(fi) - v/w*np.cos(fi + w*dt)
        fi_next = fi + w*dt

        return np.array([x_next, y_next, fi_next])

    def calcJacobian(self, state, control_input):
        dt = state.timestep
        dw = 1e-11/dt
        x = state.state[0]
        y = state.state[1]
        fi = state.state[2]
        v = control_input[0]
        w = control_input[1] + dw  # avoid divison by 0

        df1_dfi = - v/w*np.cos(fi) + v/w*np.cos(fi + w*dt)
        df2_dfi = - v/w*np.sin(fi) + v/w*np.sin(fi + w*dt)

        jacob_x = np.array([[1, 0, df1_dfi], [0, 1, df2_dfi], [0, 0, 1]])

        df1_dv = -1/w*np.sin(fi) + 1/w*np.sin(fi+w*dt)
        df1_dw = v*w**(-2)*np.sin(fi) + v/w*dt * \
            np.cos(fi + w*dt) - v*w**(-2)*np.cos(fi + w*dt)

        df2_dv = 1/w*np.cos(fi) - 1/w*np.cos(fi+w*dt)
        df2_dw = -v*w**(-2)*np.cos(fi) + v/w*dt * \
            np.sin(fi + w*dt) + v*w**(-2)*np.cos(fi + w*dt)


class OdometryMotionModel(MotionModel):
    def __init__(self, alfas):
        self.alfa_1 = alfas[0]
        self.alfa_2 = alfas[1]
        self.alfa_3 = alfas[2]
        self.alfa_4 = alfas[3]
        pass

    def propagate(self, state: StateHypothesis, control_input, noise=False) -> StateHypothesis:

        state_odom_prev = control_input[0]
        state_odom_now = control_input[1]
        fi = state.pose[2, 0]

        if (np.linalg.norm(state_odom_now[1] - state_odom_prev[1]) < 0.01):
            delta_rot_1 = 0
        else:
            delta_rot_1 = self.calc_angle_diff(np.arctan2(
                state_odom_now[1] - state_odom_prev[1],
                state_odom_now[0] - state_odom_prev[0]), state_odom_prev[2])

        delta_trans = np.sqrt(
            (state_odom_now[0] - state_odom_prev[0])**2
            + (state_odom_now[1] - state_odom_prev[1])**2)

        delta_rot_2 = self.calc_angle_diff(
            state_odom_now[2] - state_odom_prev[2], delta_rot_1)

        # We want to treat backward and forward motion symmetrically for the
        # noise model to be applied below.  The standard model seems to assume
        # forward motion.
        # From https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/sensors/amcl_odom.cpp
        delta_rot_1_corr = min(abs(self.calc_angle_diff(delta_rot_1, 0)), abs(
            self.calc_angle_diff(delta_rot_1, np.pi)))
        delta_rot_2_corr = min(abs(self.calc_angle_diff(delta_rot_2, 0)), abs(
            self.calc_angle_diff(delta_rot_2, np.pi)))

        control_covar = self.calcControlNoiseCovar(
            delta_rot_1_corr, delta_rot_2_corr, delta_trans)

        if noise:
            delta_rot_1 = self.calc_angle_diff(
                delta_rot_1, np.sqrt(control_covar[0, 0]) * np.random.randn())
            delta_trans = self.calc_angle_diff(
                delta_trans, np.sqrt(control_covar[1, 1]) * np.random.randn())
            delta_rot_2 = self.calc_angle_diff(
                delta_rot_2, np.sqrt(control_covar[2, 2]) * np.random.randn())

        prop_pose = state.pose + np.array([[delta_trans*np.cos(fi + delta_rot_1),
                                          delta_trans *
                                          np.sin(fi + delta_rot_1),
                                          delta_rot_1 + delta_rot_2]]).T
        # if prop_pose[2] > np.pi:
        #     prop_pose[2] -= 2*np.pi
        # elif prop_pose[2] < -np.pi:
        #     prop_pose[2] += 2*np.pi

        if len(state.covar):
            jacobi_state, jacobi_input = self.calcJacobians(
                delta_rot_1, delta_trans, fi)
            prop_covar = jacobi_state @ state.covar @ jacobi_state.T
            + jacobi_input @ control_covar @ jacobi_input.T
        else:
            prop_covar = None

        prop_state = StateHypothesis()
        prop_state.pose = prop_pose
        prop_state.covar = prop_covar
        return prop_state

    def propagate_particles(self, particle_poses, control_input):
        def propagate(pose):
            return self.propagate(StateHypothesis(pose), control_input, noise=True).pose

        particle_poses_next = np.array(
            list(map(propagate, particle_poses))).squeeze()

        return particle_poses_next

    def calcJacobians(self, delta_rot_1, delta_trans, fi):

        J_state_13 = -delta_trans*np.sin(fi + delta_rot_1)
        J_state_23 = delta_trans*np.cos(fi + delta_rot_1)
        J_state = np.array(
            [[1, 0, J_state_13], [0, 1, J_state_23], [0, 0, 1]])

        J_input_11 = -delta_trans*np.sin(fi + delta_rot_1)
        J_input_21 = delta_trans*np.cos(fi + delta_rot_1)
        J_input_12 = np.cos(fi + delta_rot_1)
        J_input_22 = np.sin(fi + delta_rot_1)

        J_input = np.array([[J_input_11, J_input_12, 0],
                            [J_input_21, J_input_22, 0],
                            [1, 0, 1]])

        return [J_state, J_input]

    def calcControlNoiseCovar(self, delta_rot_1, delta_rot_2, delta_trans):

        control_covar_11 = self.alfa_1 * \
            delta_rot_1**2 + self.alfa_2 * delta_trans**2
        control_covar_22 = self.alfa_3 * delta_trans**2 + \
            self.alfa_4*(abs(delta_rot_1)**2 + abs(delta_rot_2)**2)
        control_covar_33 = self.alfa_1 * \
            abs(delta_rot_2)**2 + self.alfa_2 * delta_trans**2

        return np.diag([control_covar_11, control_covar_22, control_covar_33])

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def calc_angle_diff(self, a, b):
        a = self.normalize_angle(a)
        b = self.normalize_angle(b)
        d1 = a-b
        d2 = 2*np.pi - abs(d1)
        if(d1 > 0):
            d2 *= -1.0
        if(abs(d1) < abs(d2)):
            return d1
        else:
            return d2
