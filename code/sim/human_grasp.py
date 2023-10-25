import socket
import struct
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from mujoco_py import load_model_from_path, MjSim, MjViewer
import threading
import os


client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

ip_port=('127.0.0.1',8000)

client.connect(ip_port)
data = np.zeros(254)


import serial
from mujoco_py import load_model_from_path, MjSim, MjViewer

serialPort = "COM9"
baudRate = 115200
ser = serial.Serial(serialPort, baudRate, timeout = 1)

Angle = [0.0]*20


def rotation(theta_y, theta_x, theta_z):
    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_y.dot(rot_x).dot(rot_z)
    return R


def apply_torque(current_pos, current_vel, desired_pos, desired_vel, fd_torque = 0, kp = 100, kv = 0.01 ):
    "pd controller"
    pos_torque = - kp * (current_pos - desired_pos)
    vel_torque = - kv * (current_vel - desired_vel)
    torque = pos_torque + vel_torque + fd_torque

    return torque

def serial_data():
    global Angle
    while True:
        if int(ser.read().hex(),16) == 253:
            if int(ser.read().hex(),16) == 255:
                for i in range(20):
                    Data = int(ser.read(2).hex(),16)
                    Angle[i] = (-Data+512)*6.28/1024

                    if i == 0 :
                        Angle[i] = -Angle[i]
                    if i == 18 or i == 19:
                        Angle[i] = -Angle[i]
                    if Angle[i]<0:
                        Angle[i]=0


def socket_data():
    global data
    while 1:
        a=client.recv(1024)
        if len(a) == 1016:
            length = len(a)/4
            data = struct.unpack('<' + str(int(length)) + 'f', a)
        else:
            pass


def mujoco_sim():
    global data
    global Angle
    model= load_model_from_path("./assets/intelligence_human.xml")
    sim = MjSim(model)
    body_id = sim.model.body_name2id('table')
    # palm_id = sim.model.body_name2id('robot0:hand mount')
    # path = "./robot_intelligence/traject_human1.txt"
    # if os.path.exists(path):
    #     os.remove(path)

    l1 = np.array([-0.265, 0, 0])
    l2 = np.array([-0.26, 0, 0])
    lookat = sim.data.body_xpos[body_id]
    viewer = MjViewer(sim)
    viewer2 = MjViewer(sim)
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2
    viewer.cam.azimuth = 0.
    viewer.cam.elevation = -20.
    # viewer._run_speed = 1

    for idx, value in enumerate(lookat):
        viewer2.cam.lookat[idx] = value
    viewer2.cam.distance = 2
    viewer2.cam.azimuth = -90.
    viewer2.cam.elevation = -20.
    viewer2._run_speed = 1
    qpos = sim.data.qpos
    qvel = sim.data.qvel
    ctrl = sim.data.ctrl
    while 1:
        base_matrix = np.array([[1, 0.0, 0], [0, 0, -1], [0, 1, 0]])
        matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)
        matrix3 = rotation(data[16 * 6 + 3] / 180 * 3.14, data[16 * 6 + 4] / 180 * 3.14, data[16 * 6 + 5] / 180 * 3.14)

        pos1 = matrix1.dot(l1)
        pos2 = matrix1.dot(matrix2).dot(l2)
        pos = pos1 + pos2
        # print(pos)
        # time.sleep(0.5)
        # pos = [pos[0], (-pos[2] + 0.2) * 2, pos[1] + 0.4]
        matrix = matrix1.dot(matrix2).dot(matrix3)
        # matrix = base_matrix.dot(matrix)
        # matrix = matrix1.dot(matrix2).dot(matrix3)
        r = R.from_matrix(matrix)
        euler = r.as_euler('YXZ')
        # print(euler[0])
        # quat = r.as_quat()
        # quat = np.array([quat[3], quat[0], quat[1], quat[2]])


        # sim.data.set_mocap_pos("mocap", pos)

        # sim.data.set_mocap_quat("mocap", quat)

        ctrl[0] = apply_torque(qpos[0], qvel[0], -pos[1], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        ctrl[1] = apply_torque(qpos[1], qvel[1], pos[2] - 0.3, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        ctrl[2] = apply_torque(qpos[2], qvel[2], -pos[0] + 0.2, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        
                
        ctrl[3] = apply_torque(qpos[3], qvel[3], euler[0], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        ctrl[4] = apply_torque(qpos[4], qvel[4], euler[1], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        # ctrl[5] = apply_torque(qpos[5], qvel[5], euler[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
        ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

        ctrl[6] = apply_torque(qpos[6], qvel[6], Angle[6], 0, fd_torque = 0, kp = 100, kv = 0.1 )
        ctrl[7] = apply_torque(qpos[9], qvel[9], Angle[10], 0, fd_torque = 0, kp = 100, kv = 0.1 )
        ctrl[8] = apply_torque(qpos[12], qvel[12], Angle[14], 0, fd_torque = 0, kp = 100, kv = 0.1 )
        ctrl[9] = apply_torque(qpos[15], qvel[15], Angle[18], 0, fd_torque = 0, kp = 100, kv = 0.1 )
        ctrl[10] = apply_torque(qpos[19], qvel[19], Angle[3] * 2, 0, fd_torque = 0, kp = 100, kv = 0.1 )
        ctrl[11] = apply_torque(qpos[18], qvel[18], (Angle[0] -0.3)* 3, 0, fd_torque = 0, kp = 10, kv = 0.1 )
    
        # print(Angle[6], Angle[10], Angle[14], Angle[18], Angle[2] * 2, (Angle[0] -0.25)* 3)

        # r = R.from_matrix(sim.data.body_xmat[palm_id].reshape(3,3))
        # euler = r.as_euler('YXZ')
        # trajectory = np.array([sim.data.body_xpos[palm_id], euler]).reshape(1,6)
        trajectory = np.array([qpos[0:6]]).reshape(1,6)

        # print(trajectory)
        # with open(path,"a") as file:
        #     np.savetxt(file, trajectory)    
        sim.step()          
        viewer.render()
        viewer2.render()



t1 = threading.Thread(target = socket_data)
t2 = threading.Thread(target = serial_data)

t1.start()
t2.start()


mujoco_sim()