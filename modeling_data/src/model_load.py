from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import struct
import socket
import threading


l1 = np.array([-0.265, 0, 0])
l2 = np.array([-0.26, 0, 0])

model = load_model_from_path("./assets/robot_arm_2d.xml")

sim = MjSim(model)

viewer = MjViewer(sim)

# client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# ip_port=('127.0.0.1',8000)

# client.connect(ip_port)
data = np.zeros(254)

pos_offset = [0, 0.10921554,  0.471059]
hand_offset = [-0.3, 0.3, 0.8]


def rotation(theta_y, theta_x, theta_z):
    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_y.dot(rot_x).dot(rot_z)
    return R

# def socket_data():
#     global data
#     while 1:
#         a=client.recv(1024)
#         if len(a) == 1016:
#             length = len(a)/4
#             data = struct.unpack('<' + str(int(length)) + 'f', a)
#         else:
#             pass
def mujoco_sim():
    # target_id = sim.model.site_name2id("target")
    # grasp_id = sim.model.site_name2id("grasp_center")

    while 1:
        # print(np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id]))
        # if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id])< 0.05:
        #     # print("done")
        #     sim.data.site_xpos[target_id] = [0.1, 0.1 ,0.1]
        #     sim.step()
        #     viewer.render()
        # # base_matrix = np.array([[1, 0.0, 0], [0, 0, -1], [0, 1, 0]])
        # matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        # matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)
        # # matrix3 = rotation(data[16 * 6 + 3] / 180 * 3.14, data[16 * 6 + 4] / 180 * 3.14, data[16 * 6 + 5] / 180 * 3.14)
        # # print(base_matrix)
        # # print(matrix1)
        # pos1 = matrix1.dot(l1)
        # pos2 = matrix1.dot(matrix2).dot(l2)
        # pos = pos1 + pos2
        # pos = [pos[0], -pos[2] + 0.3, 0.1]
        # # print(pos)
        # sim.data.set_mocap_pos("mocap", [0.1, 0, 0.1])
        # sim.data.set_mocap_pos("mocap", pos)
        pos = (np.random.rand(2) - 0.5) * 0.4
        # sim.data.site_xpos[target_id] = pos
        # sim.model.site_pos[target_id] = pos
        sim.data.qpos[0:2] = pos

        sim.step()
        viewer.render()

# t1 = threading.Thread(target = socket_data)


# t1.start()



mujoco_sim()