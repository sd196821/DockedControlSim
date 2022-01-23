from dynamics.quadrotor import Drone
from controller.PIDController import controller
from utils.transform import quat2rot, rot2euler, euler2rot, rot2quat, rad2deg, deg2rad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ini_pos = np.array([0, 0, 1])
ini_att = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), 0])))
ini_angular_rate = np.array([0, deg2rad(0), 0])

# Quadrotor 1 Initial States
ini_state_qd1 = np.zeros(13)
ini_state_qd1[0:3] = np.array([1.0, 0.0, 1.0])
ini_state_qd1[6:10] = ini_att
ini_state_qd1[10:] = ini_angular_rate

# Quadrotor 2 Initial States
ini_state_qd2 = np.zeros(13)
ini_state_qd2[0:3] = np.array([1.2, 0.0, 1.0])
ini_state_qd2[6:10] = ini_att
ini_state_qd2[10:] = ini_angular_rate
qd2_dock_port = np.array([-0.1, 0, 0])

# att_des = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))
# pos_des = np.array([0.0, 0, 1.0])  # [x, y, z]
# state_des = np.zeros(13)
# state_des[0:3] = pos_des
# state_des[6:10] = att_des

# Initializing two drones and set initial state
quad1 = Drone()
quad1.reset(ini_state_qd1)

quad2 = Drone()
quad2.reset(reset_state=ini_state_qd2, dock_port=qd2_dock_port)

# Structure parameter
d = 0.086  # same as drone.armlength
dock_port_length = 0.1
x1 = -dock_port_length  # modular drone coordination in Structure Coordination System
y1 = 0
x2 = dock_port_length
y2 = 0
n_qd = 2  # number of quadrotors
motor_lambda = quad1.motor_lambda
P1_matrix = np.array([[1, np.sign(y1 - d), np.sign(x1 + d), 1],
                      [1, np.sign(y1 - d), np.sign(x1 - d), -1],
                      [1, np.sign(y1 + d), np.sign(x1 - d), 1],
                      [1, np.sign(y1 + d), np.sign(x1 + d), -1]])
P2_matrix = np.array([[1, np.sign(y2 - d), np.sign(x2 + d), 1],
                      [1, np.sign(y2 - d), np.sign(x2 - d), -1],
                      [1, np.sign(y2 + d), np.sign(x2 - d), 1],
                      [1, np.sign(y2 + d), np.sign(x2 + d), -1]])
E_matrix = np.diag([0.25 / n_qd, 4 * d, 4 * d, 4])

structure_allocation_qd1 = np.array([[1, 1, 1, 1],
                                     [y1, y1 + d, y1, y1 - d],
                                     [-(x1 + d), -x1, -(x1 - d), -x1],
                                     [motor_lambda, -motor_lambda, motor_lambda, -motor_lambda]])
structure_allocation_qd2 = np.array([[1, 1, 1, 1],
                                     [y2, y2 + d, y2, y2 - d],
                                     [-(x2 + d), -x2, -(x2 - d), -x2],
                                     [motor_lambda, -motor_lambda, motor_lambda, -motor_lambda]])
# Docked Structure, Initialize a virtual drone
quads = Drone()
ini_state_qds = np.zeros(13)
ini_state_qds[0:6] = (quad1.state[0:6] + quad2.state[0:6]) / n_qd
ini_state_qds[6:10] = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))
ini_state_qds[10:] = ini_angular_rate

quads_Ixx = np.square(x1) + np.square(x2)
quads_Iyy = np.square(y1) + np.square(y2)
quads_Izz = np.square(x1) + np.square(x2) + np.square(y1) + np.square(y2)
quads_inertia = quad1.get_inertia() + \
                quad2.get_inertia() + \
                quad1.get_mass() * np.diag([quads_Ixx, quads_Iyy, quads_Izz])
quads.set_inertia(quads_inertia)
quads.set_mass(quad1.get_mass() + quad2.get_mass())
quads.reset(reset_state=ini_state_qds)

att_des_qds = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))
pos_des_qds = np.array([1.1, 0, 1.0])  # [x, y, z]
state_des_qds = np.zeros(13)
state_des_qds[0:3] = np.array([0.0, 0, 1.0])
state_des_qds[6:10] = rot2quat(euler2rot(np.array([deg2rad(0), deg2rad(0), deg2rad(0)])))

# Controller Initialization
control_qd1 = controller(quad1.get_arm_length(), quad1.get_mass())
control_qd2 = controller(quad2.get_arm_length(), quad2.get_mass())

# Control Command
u_qd1 = np.zeros(quad1.dim_u)
u_qd2 = np.zeros(quad2.dim_u)
u_qds = np.zeros(4)

total_step = 500  # 10s
# total_step = 100 # 2s
state_qd1 = np.zeros((total_step, 13))
state_des_all_qd1 = np.zeros((total_step, 13))
rpy_qd1 = np.zeros((total_step, 3))
u_all_qd1 = np.zeros((total_step, 4))

state_qd2 = np.zeros((total_step, 13))
state_des_all_qd2 = np.zeros((total_step, 13))
rpy_qd2 = np.zeros((total_step, 3))
u_all_qd2 = np.zeros((total_step, 4))

state_qds = np.zeros((total_step, 13))
state_des_all_qds = np.zeros((total_step, 13))
rpy_qds = np.zeros((total_step, 3))
u_all_qds = np.zeros((total_step, 4))

time = np.zeros(total_step)

# Run simulation
for t in range(total_step):
    state_now_qd1 = quad1.get_state()
    state_now_qd2 = quad2.get_state()
    state_now_qds = quads.get_state()
    state_now_qds[3:6] = (state_now_qd1[3:6] + state_now_qd2[3:6]) / n_qd

    # u = control.att_alt_controller(state_des, state_now)
    # Paper Fig.4 Eq.(15)-Eq.(14)
    F_qds, att_des_qds = control_qd1.structure_hover_controller(state_des_qds, state_now_qds)
    # F_qd2, ang_acc_des_qd2 = control_qd2.structure_module_controller(state_des_qds, state_now_qds)

    # Paper Fig.5 Eq.(11)-Eq.(13)
    F_qd1, ang_acc_des_qd1 = control_qd1.module_controller(state_des_qds, state_now_qd1, F_qds, att_des_qds)
    F_qd2, ang_acc_des_qd2 = control_qd2.module_controller(state_des_qds, state_now_qd2, F_qds, att_des_qds)

    Thrust_qd1 = control_qd1.module_thrust_allocation(P1_matrix, E_matrix, F_qd1, ang_acc_des_qd1)
    Thrust_qd2 = control_qd1.module_thrust_allocation(P2_matrix, E_matrix, F_qd2, ang_acc_des_qd2)

    u_qd1 = quad1.thrust_to_force(Thrust_qd1)
    u_qd2 = quad1.thrust_to_force(Thrust_qd2)

    u_qd1_in_qds = structure_allocation_qd1 @ Thrust_qd1
    u_qd2_in_qds = structure_allocation_qd2 @ Thrust_qd2
    u_qds = u_qd1_in_qds + u_qd2_in_qds

    # u[1:] = quad1.Inertia @
    # u[1:] = control.attitude_controller(state_des, state_now)
    u_all_qd1[t, :] = u_qd1
    u_all_qd2[t, :] = u_qd2
    u_all_qds[t, :] = u_qds

    state_qd1[t, :] = state_now_qd1
    state_qd2[t, :] = state_now_qd2
    state_qds[t, :] = state_now_qds

    rpy_qd1[t, :] = rot2euler(quat2rot(state_now_qd1[6:10]))
    rpy_qd2[t, :] = rot2euler(quat2rot(state_now_qd2[6:10]))
    rpy_qds[t, :] = rot2euler(quat2rot(state_now_qds[6:10]))

    # rpy[t, :] = quat2euler(state_now[6:10])
    time[t] = quad1.get_time()
    # print("time : ", time[t])
    quad1.step(u_qd1)
    quad2.step(u_qd2)
    quads.step(u_qds)

# ******** Plot Results
fig = plt.figure()
fig.canvas.set_window_title("Structure Profile")
plt.subplot(2, 3, 1)
plt.plot(time, state_qds[:, 0:3])
plt.legend(['x', 'y', 'z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

# plt.figure()
plt.subplot(2, 3, 2)
plt.plot(time, state_qds[:, 3:6])
plt.legend(['vx', 'vy', 'vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

# plt.figure()
plt.subplot(2, 3, 3)
plt.plot(time, rad2deg(rpy_qds))
plt.legend(['roll', 'pitch', 'yaw'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

# plt.figure()
plt.subplot(2, 3, 4)
plt.plot(time, rad2deg(state_qds[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

# plt.figure()
plt.subplot(2, 3, 5)
plt.plot(time, u_all_qds[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

# plt.figure()
plt.subplot(2, 3, 6)
plt.plot(time, u_all_qds[:, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

trajectory_fig = plt.figure()
trajectory_fig.canvas.set_window_title("Structure Trajectory")
ax = Axes3D(trajectory_fig)
ax.plot3D(state_qds[:, 0], state_qds[:, 1], state_qds[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# plt.show()

# ******* Quadrotor 1
fig = plt.figure()
fig.canvas.set_window_title("Quadrotor 1 Profile")
plt.subplot(2, 3, 1)
plt.plot(time, state_qd1[:, 0:3])
plt.legend(['x', 'y', 'z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

# plt.figure()
plt.subplot(2, 3, 2)
plt.plot(time, state_qd1[:, 3:6])
plt.legend(['vx', 'vy', 'vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

# plt.figure()
plt.subplot(2, 3, 3)
plt.plot(time, rad2deg(rpy_qd1))
plt.legend(['roll', 'pitch', 'yaw'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

# plt.figure()
plt.subplot(2, 3, 4)
plt.plot(time, rad2deg(state_qd1[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

# plt.figure()
plt.subplot(2, 3, 5)
plt.plot(time, u_all_qd1[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

# plt.figure()
plt.subplot(2, 3, 6)
plt.plot(time, u_all_qd1[:, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

trajectory_fig = plt.figure()
trajectory_fig.canvas.set_window_title("Quadrotor 1 Trajectory")
ax = Axes3D(trajectory_fig)
ax.plot3D(state_qd1[:, 0], state_qd1[:, 1], state_qd1[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# plt.show()

# ******** Quadrotor 2
fig = plt.figure()
fig.canvas.set_window_title("Quadrotor 2 Profile")
plt.subplot(2, 3, 1)
plt.plot(time, state_qd2[:, 0:3])
plt.legend(['x', 'y', 'z'])
plt.xlabel("Time/s")
plt.ylabel("Position/m")
plt.title("Position")

# plt.figure()
plt.subplot(2, 3, 2)
plt.plot(time, state_qd2[:, 3:6])
plt.legend(['vx', 'vy', 'vz'])
plt.xlabel("Time/s")
plt.ylabel("Velocity/m*s^-1")
plt.title("Velocity")

# plt.figure()
plt.subplot(2, 3, 3)
plt.plot(time, rad2deg(rpy_qd2))
plt.legend(['roll', 'pitch', 'yaw'])
plt.xlabel("Time/s")
plt.ylabel("Angle/deg")
plt.title("Attitude")

# plt.figure()
plt.subplot(2, 3, 4)
plt.plot(time, rad2deg(state_qd2[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.xlabel("Time/s")
plt.ylabel("Angular rate/deg*s^-1")
plt.title("Angular Rates")

# plt.figure()
plt.subplot(2, 3, 5)
plt.plot(time, u_all_qd2[:, 1:])
plt.legend(['Mx', 'My', 'Mz'])
plt.xlabel("Time/s")
plt.ylabel("Moment/Nm")
plt.title("Control Moment")

# plt.figure()
plt.subplot(2, 3, 6)
plt.plot(time, u_all_qd2[:, 0])
plt.xlabel("Time/s")
plt.ylabel("Force/N")
plt.title("Total Thrust")

trajectory_fig = plt.figure()
trajectory_fig.canvas.set_window_title("Quadrotor 2 Trajectory")
ax = Axes3D(trajectory_fig)
ax.plot3D(state_qd2[:, 0], state_qd2[:, 1], state_qd2[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
