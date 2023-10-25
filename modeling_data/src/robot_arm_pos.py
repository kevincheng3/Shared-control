from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import struct
import socket
import threading
import os
import time

l1 = np.array([-0.265, 0, 0])
l2 = np.array([-0.26, 0, 0])

model = load_model_from_path("./assets/robot_arm_2d.xml")

sim = MjSim(model)

# viewer = MjViewer(sim)
body_id = sim.model.body_name2id("target")

lookat = sim.data.body_xpos[body_id]
viewer = MjViewer(sim)
# viewer2 = MjViewer(sim)
for idx, value in enumerate(lookat):
    viewer.cam.lookat[idx] = value
viewer.cam.distance = 2.2
viewer.cam.azimuth = -90.
viewer.cam.elevation = -80.
viewer._run_speed = 0.02
target_id = sim.model.body_name2id("target")
grasp_id = sim.model.site_name2id("grasp_center")
obs_id = sim.model.body_name2id("obs")

client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

ip_port=('127.0.0.1',8000)

client.connect(ip_port)
data = np.zeros(254)

pos_offset = [0, 0.10921554,  0.471059]
hand_offset = [-0.3, 0.3, 0.8]
pos = [0, 0, 0]

def rotation(theta_y, theta_x, theta_z):
    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_y.dot(rot_x).dot(rot_z)
    return R

def socket_data():
    global data
    while 1:
        # time.sleep(0.01)
        a=client.recv(1024)
        if len(a) == 1016:
            length = len(a)/4
            data = struct.unpack('<' + str(int(length)) + 'f', a)
            # time.sleep(0.1)

        else:
            pass

def mujoco_human():
    path1 = "./modeling_ur/data/human/current.txt"
    path2 = "./modeling_ur/data/human/target.txt"
    path3 = "./modeling_ur/data/human/observation.txt"

    if os.path.exists(path1):
        os.remove(path1)

    if os.path.exists(path2):
        os.remove(path2)

    if os.path.exists(path3):
        os.remove(path3)


    while 1:
        # print(np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id]))
        with open(path1,"a") as file:
            np.savetxt(file, sim.data.site_xpos[grasp_id].reshape(1,3))

        with open(path2,"a") as file:
            np.savetxt(file, sim.data.body_xpos[target_id].reshape(1,3))
        if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[target_id])< 0.005:
            # print("done")
            # pos = np.random.rand(2)
        # sim.data.site_xpos[target_id] = pos
        # sim.model.site_pos[target_id] = pos

            random_pos = (np.random.rand(2) - 0.5) * 0.4
            sim.data.qpos[0:2] = random_pos
            sim.data.qpos[2:4] = random_pos + (np.random.rand(2) - 0.5) * 0.2


            # sim.data.site_xpos[target_id] = [0.1, 0.1 ,0.1]
            sim.step()
            viewer.render()
        matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)

        pos1 = matrix1.dot(l1)
        pos2 = matrix1.dot(matrix2).dot(l2)
        pos = pos1 + pos2
        pos = [pos[0], -pos[2] + 0.3, 0.1]

        sim.data.set_mocap_pos("mocap", pos)

        sim.step()
        viewer.render()

def mujoco_auto():
    sim.data.set_mocap_pos("mocap", [0.0,0,0])
    for i in range(300):
        sim.step()

    path1 = "./modeling_ur/data/auto/current1.txt"
    path2 = "./modeling_ur/data/auto/target1.txt"
    path3 = "./modeling_ur/data/auto/observation1.txt"

    if os.path.exists(path1):
        os.remove(path1)

    if os.path.exists(path2):
        os.remove(path2)

    if os.path.exists(path3):
        os.remove(path3)
    while 1:
        with open(path1,"a") as file:
            np.savetxt(file, sim.data.site_xpos[grasp_id].reshape(1,3))

        with open(path2,"a") as file:
            np.savetxt(file, sim.data.body_xpos[target_id].reshape(1,3))

        with open(path3,"a") as file:
            np.savetxt(file, sim.data.body_xpos[obs_id].reshape(1,3))        
        # print(np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id]))
        if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[obs_id])< 0.005:

            random_pos = (np.random.rand(2) - 0.5) * 0.4
            sim.data.qpos[0:2] = random_pos
            sim.data.qpos[2:4] = random_pos + (np.random.rand(2) - 0.5) * 0.2

            # sim.step()
            # viewer.render()

        delta_pos = -(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[obs_id]) * 0.009

        pos = [sim.data.mocap_pos[0][0] + delta_pos[0], sim.data.mocap_pos[0][1] + delta_pos[1], 0.1]
        sim.data.set_mocap_pos("mocap", pos)
        sim.data.set_mocap_quat("mocap", [0.49989816, 0.49999999, 0.49999999, 0.50010184])
        

        sim.step()
        viewer.render()

def mujoco_shared():
    auto_mode = True
    pos = np.array([0,0,0])
    sim.data.set_mocap_pos("mocap", [0.0,0,0])

    path1 = "./modeling_ur/data/shared/current.txt"
    path2 = "./modeling_ur/data/shared/target.txt"
    path3 = "./modeling_ur/data/shared/observation.txt"

    if os.path.exists(path1):
        os.remove(path1)

    if os.path.exists(path2):
        os.remove(path2)

    if os.path.exists(path3):
        os.remove(path3)
    for i in range(300):
        sim.step()
    while 1:
        # print(np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id]))
        with open(path1,"a") as file:
            np.savetxt(file, sim.data.site_xpos[grasp_id].reshape(1,3))

        with open(path2,"a") as file:
            np.savetxt(file, sim.data.body_xpos[target_id].reshape(1,3))
        
        last_pos = pos.copy()
        matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)

        pos1 = matrix1.dot(l1)
        pos2 = matrix1.dot(matrix2).dot(l2)
        pos = pos1 + pos2
        pos = np.array([pos[0], -pos[2] + 0.3, 0.1])

        if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[target_id])< 0.005:
            random_pos = (np.random.rand(2) - 0.5) * 0.4
            sim.data.qpos[0:2] = random_pos
            sim.data.qpos[2:4] = random_pos + (np.random.rand(2) - 0.5) * 0.2
            auto_mode = True

            # sim.data.site_xpos[target_id] = [0.1, 0.1 ,0.1]
            sim.step()
            viewer.render()

        if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[obs_id]) > 0.01 and auto_mode == True:

            delta_pos = -(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[obs_id]) * 0.01
            # delta_pos = [pos[0], pos[1], 0]
            # print(delta_pos, sim.data.mocap_pos)
            pos1 = [sim.data.mocap_pos[0][0] + delta_pos[0], sim.data.mocap_pos[0][1] + delta_pos[1], 0.1]
            sim.data.set_mocap_pos("mocap", pos1)
            sim.data.set_mocap_quat("mocap", [0.49989816, 0.49999999, 0.49999999, 0.50010184])
        else: 
            # print("done")
            auto_mode = False
            new_pos = pos -last_pos
            # print(new_pos)
            mocap_pos = [sim.data.mocap_pos[0][0] + new_pos[0], sim.data.mocap_pos[0][1] + new_pos[1], 0.1]
            sim.data.set_mocap_pos("mocap", mocap_pos)


        # for i in range(10):
        #     sim.step()
        #     viewer.render()

        sim.step()
        viewer.render()

def mujoco_lantency():
    path1 = "./modeling_ur/data/human/current.txt"
    path2 = "./modeling_ur/data/human/target.txt"
    path3 = "./modeling_ur/data/human/observation.txt"

    if os.path.exists(path1):
        os.remove(path1)

    if os.path.exists(path2):
        os.remove(path2)

    if os.path.exists(path3):
        os.remove(path3)


    while 1:
        # print(np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.site_xpos[target_id]))
        with open(path1,"a") as file:
            np.savetxt(file, sim.data.site_xpos[grasp_id].reshape(1,3))

        with open(path2,"a") as file:
            np.savetxt(file, sim.data.body_xpos[target_id].reshape(1,3))
        
        if np.linalg.norm(sim.data.site_xpos[grasp_id] - sim.data.body_xpos[target_id])< 0.005:
            # print("done")
            # pos = np.random.rand(2)
        # sim.data.site_xpos[target_id] = pos
        # sim.model.site_pos[target_id] = pos

            random_pos = (np.random.rand(2) - 0.5) * 0.4
            sim.data.qpos[0:2] = random_pos
            sim.data.qpos[2:4] = random_pos + (np.random.rand(2) - 0.5) * 0.2
            # sim.data.site_xpos[target_id] = [0.1, 0.1 ,0.1]
            sim.step()
            viewer.render()
        matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)

        pos1 = matrix1.dot(l1)
        pos2 = matrix1.dot(matrix2).dot(l2)
        pos = pos1 + pos2
        pos = [pos[0], -pos[2] + 0.3, 0.1]

        sim.data.set_mocap_pos("mocap", pos)

        sim.step()
        viewer.render()



t1 = threading.Thread(target = socket_data)


t1.start()


# mujoco_lantency()
mujoco_human()
# mujoco_auto()
# mujoco_shared()