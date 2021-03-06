import numpy as np
import scipy
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad, euler2quat


# quat2euler,

class controller():
    """Controller Class"""

    def __init__(self, L, mass):
        self.n_qd = 2
        self.kp_roll = 40  # 60; 10 ;15
        self.kp_pitch = 40
        self.kp_yaw = 40  # 70

        self.kd_roll = 18  # 50; 14.3; 21(5s)
        self.kd_pitch = 18  # 50
        self.kd_yaw = 9  # 13
        self.ff_yaw = 0.0

        self.kp_x = 2.5  # 19   # 0.3 # 0.1;0.2;0.3
        self.kp_y = 2.5  # 19    0.3
        self.kp_z = 20  # 20  # 10;

        self.kd_x = 3  # 1.87 # 0.9 # 0.4;0.6(12.5s) 0.7();0.9(10)
        self.kd_y = 3  # 1.87 0.9
        self.kd_z = 7  # 18  # 6;

        self.kp_vx = 0  # 1.87 # 0.9 # 0.4;0.6(12.5s) 0.7();0.9(10)
        self.kp_vy = 0  # 1.87 0.9

        self.kd_vx = 0.0  # 1.87 # 0.9 # 0.4;0.6(12.5s) 0.7();0.9(10)
        self.kd_vy = 0.0  # 1.87 0.9

        self.kp_vz = 0
        self.kd_vz = 0

        # Structure Combined controller gain k
        self.k_sv = 0.1
        self.k_sa = 0.1

        # For Joystick RC Controller
        self.kp_yaw_rate = 45
        self.kd_yaw_rate = 0.1
        #

        self.Kf = 0.8
        self.Km = 0.1

        self.g = 9.81
        self.mass = mass
        self.Inertia = np.array([[0.00025, 0, 0],
                                 [0, 0.000232, 0],
                                 [0, 0, 0.0003738]])

        # Control allocation matrix:
        self.allocation_matrix = np.linalg.inv(np.array([[self.Kf, self.Kf, self.Kf, self.Kf],
                                                         [0, self.Kf * L, 0, -self.Kf * L],
                                                         [-self.Kf * L, 0, self.Kf * L, 0],
                                                         [self.Km, -self.Km, self.Km, -self.Km]]))

    def attitude_controller(self, state_des, state_now):
        """
        Attitude Controller
        :param state_des: Desired State[13]
        :param state_now: Current State[13]
        :return: M: output moment[3]
        """
        attitude_des = rot2euler(quat2rot(state_des[6:10]))
        attitude_now = rot2euler(quat2rot(state_now[6:10]))
        # attitude_des = quat2euler(state_des[6:10])
        # attitude_now = quat2euler(state_now[6:10])
        att_rate_des = state_des[10:]
        att_rate_now = state_now[10:]

        e_angle = attitude_des - attitude_now
        e_angular_rate = att_rate_des - att_rate_now

        M = self.Inertia @ np.array([(self.kp_roll * e_angle[0] + self.kd_roll * e_angular_rate[0]),
                                     (self.kp_pitch * e_angle[1] + self.kd_pitch * e_angular_rate[1]),
                                     (self.ff_yaw + self.kp_yaw * e_angle[2] + self.kd_yaw * e_angular_rate[2])])
        # M = k @ e[:, 0] + k @ e[:, 1] + k @ e[:, 2]
        # print(M)
        return M

    def hover_controller(self, state_des, state_now):
        acc_des = np.zeros(3)
        e_pos = state_des[0:3] - state_now[0:3]
        e_vel = state_des[3:6] - state_now[3:6]
        acc_des[0] = self.kp_x * e_pos[0] + self.kd_x * e_vel[0]
        acc_des[1] = self.kp_y * e_pos[1] + self.kd_y * e_vel[1]
        acc_des[2] = self.kp_z * e_pos[2] + self.kd_z * e_vel[2]

        F = self.mass * self.g + self.mass * acc_des[2]

        att_des = rot2euler(quat2rot(state_des[6:10]))
        # att_des = quat2euler(state_des[6:10])
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        roll_rate_des = 0
        pitch_rate_des = 0

        att_des[0] = phi_des
        att_des[1] = theta_des
        att_des[2] = psi_des

        # state_des[6:10] = euler2quat(att_des)
        state_des[6:10] = rot2quat(euler2rot(att_des))
        state_des[10] = roll_rate_des
        state_des[11] = pitch_rate_des

        return F, state_des

    def vel_controller(self, state_des, state_now, state_last):
        acc_des = np.zeros(3)
        # e_pos = state_des[0:3] - state_now[0:3]
        e_vel = state_des[3:6] - state_now[3:6]
        e_dv = state_now[3:6] - state_last[3:6]

        acc_des[0] = self.kp_vx * e_vel[0] + self.kd_vx * e_dv[0]
        acc_des[1] = self.kp_vy * e_vel[1] + self.kd_vy * e_dv[1]
        acc_des[2] = self.kp_vz * e_vel[2] + self.kd_vz * e_dv[2]

        F = self.mass * self.g + self.mass * acc_des[2]

        att_des = rot2euler(quat2rot(state_des[6:10]))
        # att_des = quat2euler(state_des[6:10])
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        roll_rate_des = 0
        pitch_rate_des = 0

        att_des[0] = phi_des
        att_des[1] = theta_des
        att_des[2] = psi_des

        # state_des[6:10] = euler2quat(att_des)
        state_des[6:10] = rot2quat(euler2rot(att_des))
        state_des[10] = roll_rate_des
        state_des[11] = pitch_rate_des

        M = self.attitude_controller(state_des, state_now)
        output = np.zeros(4)
        output[0] = F
        output[1:] = M

        return output

    def pos_vel_controller(self, state_des, state_now, state_last):
        acc_des = np.zeros(3)
        e_pos = state_des[0:3] - state_now[0:3]

        e_vel = state_des[3:6] - state_now[3:6]
        e_dv = state_now[3:6] - state_last[3:6]

        acc_des[0] = self.kp_vx * e_vel[0] + self.kd_vx * e_dv[0]
        acc_des[1] = self.kp_vy * e_vel[1] + self.kd_vy * e_dv[1]
        acc_des[2] = self.kp_vz * e_vel[2] + self.kd_vz * e_dv[2]

        F = self.mass * self.g + self.mass * acc_des[2]

        att_des = rot2euler(quat2rot(state_des[6:10]))
        # att_des = quat2euler(state_des[6:10])
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        roll_rate_des = 0
        pitch_rate_des = 0

        att_des[0] = phi_des
        att_des[1] = theta_des
        att_des[2] = psi_des

        state_des[6:10] = euler2quat(att_des)
        state_des[10] = roll_rate_des
        state_des[11] = pitch_rate_des

        M = self.attitude_controller(state_des, state_now)
        output = np.zeros(4)
        output[0] = F
        output[1:] = M

    def PID(self, state_des, state_now):
        F, state_des_c = self.hover_controller(state_des, state_now)
        M = self.attitude_controller(state_des_c, state_now)
        output = np.zeros(4)
        output[0] = F
        output[1:] = M
        return output

    def rc_controller(self, state_des, state_now, state_last):
        # acc_des = np.zeros(3)
        # e_z = state_des[2] - state_now[2]
        # altitude velocity  controller
        e_vz = state_des[5] - state_now[5]
        e_dvz = state_now[5] - state_last[5]
        acc_z_des = self.kp_vz * e_vz + self.kd_vz * e_dvz  # self.kp_z * e_z + self.kd_z * e_vz
        F = self.mass * self.g + self.mass * acc_z_des
        # M = self.attitude_controller(state_des, state_now)
        # roll & pitch controller
        attitude_des = quat2euler(state_des[6:10])
        attitude_now = quat2euler(state_now[6:10])
        att_rate_des = state_des[10:]
        att_rate_now = state_now[10:]

        e_angle = attitude_des - attitude_now
        e_angular_rate = att_rate_des - att_rate_now
        e_dangular_rate = state_now[10:12] - state_last[10:12]
        e_dyaw_rate = state_now[12] - state_last[12]

        M = np.array([(self.kp_roll * e_angle[0] + self.kd_roll * e_angular_rate[0]),
                      (self.kp_pitch * e_angle[1] + self.kd_pitch * e_angular_rate[1]),
                      (self.kp_yaw_rate * e_angular_rate[2] + self.kd_yaw_rate * e_dyaw_rate)])

        output = np.zeros(4)
        output[0] = F
        output[1:] = M
        return output

    def att_alt_controller(self, state_des, state_now):
        # acc_des = np.zeros(3)
        e_z = state_des[2] - state_now[2]
        e_vz = state_des[5] - state_now[5]
        acc_z_des = self.kp_z * e_z + self.kd_z * e_vz  # self.kp_z * e_z + self.kd_z * e_vz
        F = self.mass * self.g + self.mass * acc_z_des
        M = self.attitude_controller(state_des, state_now)

        output = np.zeros(4)
        output[0] = F
        output[1:] = M
        return output

    def structure_hover_controller(self, state_des, state_now):
        """

        :param state_des:
        :param state_now:
        :return: F, attitude_des
        """
        vel_des = self.k_sv * (state_des[0:3] - state_now[0:3])  # state_des[3:6]
        vel_now = state_now[3:6]
        acc_des = self.k_sa * (vel_des - vel_now)
        F = self.mass * (self.n_qd * self.g + acc_des[2])

        att_des = rot2euler(quat2rot(state_des[6:10]))
        # att_des = quat2euler(state_des[6:10])
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        attitude_des = np.array([phi_des, theta_des, psi_des])

        return F, attitude_des

    def structure_vel_controller(self, state_des, state_now):
        """

        :param state_des:
        :param state_now:
        :param n_qd: number of docked quadrotors
        :return: F, attitude_des
        """
        vel_des = state_des[3:6]  # state_des[3:6]
        vel_now = state_now[3:6]
        acc_des = self.k_sa * (vel_des - vel_now)
        F = self.mass * (self.n_qd * self.g + acc_des[2])

        att_des = rot2euler(quat2rot(state_des[6:10]))
        # att_des = quat2euler(state_des[6:10])
        psi_des = att_des[2]

        phi_des = (acc_des[0] * np.sin(psi_des) - acc_des[1] * np.cos(psi_des)) / self.g
        theta_des = (acc_des[0] * np.cos(psi_des) + acc_des[1] * np.sin(psi_des)) / self.g

        attitude_des = np.array([phi_des, theta_des, psi_des])

        return F, attitude_des

    def module_controller(self, state_des, state_now, F, attitude_des):
        # F, attitude_des = self.structure_hover_controller(state_des=state_des, state_now=state_now)
        attitude_now = rot2euler(quat2rot(state_now[6:10]))
        att_rate_des = state_des[10:]
        att_rate_now = state_now[10:]

        e_angle = attitude_des - attitude_now
        e_angular_rate = att_rate_des - att_rate_now

        angular_rate_des = np.array([(self.kp_roll * e_angle[0] + self.kd_roll * e_angular_rate[0]),
                                     (self.kp_pitch * e_angle[1] + self.kd_pitch * e_angular_rate[1]),
                                     (self.ff_yaw + self.kp_yaw * e_angle[2] + self.kd_yaw * e_angular_rate[2])])
        return F, angular_rate_des

        # Force and Moment to 4x Thrust
        # fm = np.zeros(3)
        # fm[0] = F
        # fm[1:] = self.Inertia @ angular_rate_des
        # Thrust = P_i @ E @ fm
        # return Thrust

    def module_thrust_allocation(self, P_i, E, F, angular_rate_des):
        fm = np.zeros(4)
        fm[0] = F
        fm[1:] = self.Inertia @ angular_rate_des
        Thrust = P_i @ E @ fm
        return Thrust

    def get_motor_output(self, u):
        """Calculate motor control command as angular velocity"""
        F = u[0]
        M = u[1:]
        rotor_omega = self.allocation_matrix @ u
        return rotor_omega
