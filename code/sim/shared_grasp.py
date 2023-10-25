import socket
import struct
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import threading
import os
import cv2
import copy
import math

client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

ip_port=('127.0.0.1',8000)

client.connect(ip_port)
data = np.zeros(254)

import serial
from mujoco_py import load_model_from_path, MjSim, MjViewer

serialPort = "COM3"
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
    model= load_model_from_path("./assets/intelligence_shared_new.xml")
    sim = MjSim(model)
    body_id = sim.model.body_name2id('table')
    palm_id = sim.model.body_name2id('robot0:hand mount')
    grasp_id = sim.model.site_name2id('grasp')

    path = "./robot_intelligence/traject_sharednew.txt"
    if os.path.exists(path):
        os.remove(path)

    data1 = np.zeros(7)
    data2= np.zeros(7)

    record1 = False
    record2 = False

    l1 = np.array([-0.265, 0, 0])
    l2 = np.array([-0.26, 0, 0])
    lookat = sim.data.body_xpos[body_id]

    offscreen = MjRenderContextOffscreen(sim, 0)

    viewer = MjViewer(sim)
    viewer2 = MjViewer(sim)
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 0.
    viewer.cam.elevation = -20.
    viewer._run_speed = 1 

    for idx, value in enumerate(lookat):
        viewer2.cam.lookat[idx] = value
    viewer2.cam.distance = 1.8
    viewer2.cam.azimuth = -90.
    viewer2.cam.elevation = -20.
    viewer2._run_speed = 1
    qpos = sim.data.qpos
    qvel = sim.data.qvel
    ctrl = sim.data.ctrl

    auto = False
    px = 0
    py = 0
    x = 0
    y = 0
    pos = np.array([0, 0, 0])
    fovy = 60
    f = 0.5 * 1080 / math.tan(fovy * math.pi / 360)

    while 1:
        offscreen.render(1920, 1080, 0)
        image = offscreen.read_pixels(1920, 1080)[0]
        # image = copy.deepcopy(sim.render(width = 1920, height = 1080, camera_name = 'realsense'))
        # cv2.imshow("image", image)
        ret, thresh = cv2.threshold(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) , 100, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        last_pos = pos
        matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
        matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)
        matrix3 = rotation(data[16 * 6 + 3] / 180 * 3.14, data[16 * 6 + 4] / 180 * 3.14, data[16 * 6 + 5] / 180 * 3.14)

        pos1 = matrix1.dot(l1)
        pos2 = matrix1.dot(matrix2).dot(l2)
        pos = pos1 + pos2

        matrix = matrix1.dot(matrix2).dot(matrix3)
        # print(last_pos, pos)
        r = R.from_matrix(matrix)
        euler = r.as_euler('YXZ')
        # print(len(contours))
        if not auto:
            if (len(contours)) == 1 and sim.data.site_xpos[grasp_id][2]<0.355:
                if (not record1):
                    # cv2.imshow("image", thresh)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    record1 = True
                    print('record1', 1)
                    data1[0:3] = sim.data.site_xpos[grasp_id] 
                    for c in contours:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        # print(box)
                        data1[3:5] = box[0]
                        data1[5:7] = box[2]
                    while sim.data.site_xpos[grasp_id][2]>0.305:
                        ctrl[0] = apply_torque(qpos[0], qvel[0], qpos[0]+ 0.1, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                        trajectory = np.array([qpos[0:6]]).reshape(1,6)
                        with open(path,"a") as file:
                            np.savetxt(file, trajectory)
                        sim.step()
                        viewer.render()
                        viewer2.render()

                elif (not record2):

                    # cv2.imshow("image", thresh)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    data2[0:3] = sim.data.site_xpos[grasp_id]
                    record2 = True   
                    print('record2', record2)
                    auto = True
                    ini_pos = pos
                    print("ini_pos", ini_pos)
                    for c in contours:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        # print(box)
                        data2[3:5] = box[0]
                        data2[5:7] = box[2]

                    if record1 and record2:
                        x1 = data1[5] - data1[3]
                        x2 = data2[5] - data2[3]
                        delta_z = data1[2] - data2[2]
                        x = x1 * x2 * (delta_z) /(f * (x2-x1))
                        z1 = f * x / x1
                        z2 = f * x / x2

                        px = ((data2[3] + data2[5]) / 2 -960) * z2 / f + sim.data.site_xpos[grasp_id][0]
                        py = ((data2[4] + data2[6]) / 2 -540) * z2 / f + sim.data.site_xpos[grasp_id][1]   
                        # print((data30[3] + data30[5]) / 2 -960, ((data30[3] + data30[5]) / 2 -960) * z2 / f, sim.data.site_xpos[grasp_id][0])

                        print(px, py, x)
                else:
                    control = px + x / 2 - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[0]
                                # thumb_pos = sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396])

                    controly = py - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[1]            
                    # print(px, py, x)
                    ctrl[1] = apply_torque(qpos[1], qvel[1], -controly, 0, fd_torque = 0, kp = 500, kv = 3)
                    ctrl[2] = apply_torque(qpos[2], qvel[2], -control, 0, fd_torque = 0, kp = 500, kv = 3)
                    trajectory = np.array([qpos[0:6]]).reshape(1,6)
                    
                    with open(path,"a") as file:
                        np.savetxt(file, trajectory)
                    sim.step()
                    viewer.render()
                    viewer2.render()
            
            else:    
                for i in range(10):
                    ctrl[0] = apply_torque(qpos[0], qvel[0], -pos[1], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[1] = apply_torque(qpos[1], qvel[1], pos[2] - 0.4, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[2] = apply_torque(qpos[2], qvel[2], -pos[0] + 0.2, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    
                    # ctrl[0] = apply_torque(qpos[0], qvel[0], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    # ctrl[1] = apply_torque(qpos[1], qvel[1], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    # ctrl[2] = apply_torque(qpos[2], qvel[2], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[3] = apply_torque(qpos[3], qvel[3], 1.571, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[4] = apply_torque(qpos[4], qvel[4], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                    # ctrl[3] = apply_torque(qpos[3], qvel[3], euler[0], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    # ctrl[4] = apply_torque(qpos[4], qvel[4], euler[1], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    # # ctrl[5] = apply_torque(qpos[5], qvel[5], euler[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                    ctrl[6] = apply_torque(qpos[6], qvel[6], Angle[6], 0, fd_torque = 0, kp = 100, kv = 0.1 )
                    ctrl[7] = apply_torque(qpos[9], qvel[9], Angle[10], 0, fd_torque = 0, kp = 100, kv = 0.1 )
                    ctrl[8] = apply_torque(qpos[12], qvel[12], Angle[14], 0, fd_torque = 0, kp = 100, kv = 0.1 )
                    ctrl[9] = apply_torque(qpos[15], qvel[15], Angle[18], 0, fd_torque = 0, kp = 100, kv = 0.1 )
                    ctrl[10] = apply_torque(qpos[19], qvel[19], Angle[3] * 2, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                    ctrl[11] = apply_torque(qpos[18], qvel[18], (Angle[0] -0.25)* 3, 0, fd_torque = 0, kp = 10, kv = 0.1 )
                    trajectory = np.array([qpos[0:6]]).reshape(1,6)
                    with open(path,"a") as file:
                        np.savetxt(file, trajectory)
                    sim.step()          
                    viewer.render()
                    viewer2.render()


            # print(Angle[6], Angle[10], Angle[14], Angle[18], Angle[2] * 2, (Angle[0] -0.25)* 3)

            # r = R.from_matrix(sim.data.body_xmat[palm_id].reshape(3,3))
            # euler = r.as_euler('YXZ')
            # trajectory = np.array([sim.data.body_xpos[palm_id], euler]).reshape(1,6)
            trajectory = np.array([qpos[0:6]]).reshape(1,6)

            # print(trajectory)
            with open(path,"a") as file:
                np.savetxt(file, trajectory)    
            sim.step()          
            viewer.render()
            viewer2.render()

        else:
            py = 0.1
            px = -0.00
            x = 0.03
            print("ini_pos", ini_pos)

            # controly = py + x / 2 - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[1] + 0.2
            # control = px - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[0] -0.4         
            control2 = px - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[0] + (sim.data.body_xpos[palm_id][0] - 0.4)
            control1 = py + x / 2 - (sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))[1]+ 0.02 + sim.data.body_xpos[palm_id][1]           
            # print(control1)
            # print(px,py,x)
            for i in range(100):
                last_pos = pos
                matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
                matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)
                matrix3 = rotation(data[16 * 6 + 3] / 180 * 3.14, data[16 * 6 + 4] / 180 * 3.14, data[16 * 6 + 5] / 180 * 3.14)

                pos1 = matrix1.dot(l1)
                pos2 = matrix1.dot(matrix2).dot(l2)
                pos = pos1 + pos2
                delta_pos = pos - last_pos
                # print(pos, last_pos)
                control2 = control2 + delta_pos[0]
                # print(delta_pos[0])
                for j in range(10):
                    # ctrl[0] = apply_torque(qpos[0], qvel[0], 0.36 * i /99, 0, fd_torque = 0, kp = 500, kv = 3)
                    ctrl[0] = apply_torque(qpos[0], qvel[0], 0.36, 0, fd_torque = 0, kp = 500, kv = 3)

                    ctrl[1] = apply_torque(qpos[1], qvel[1], -control1, 0, fd_torque = 0, kp = 500, kv = 3)
                    ctrl[2] = apply_torque(qpos[2], qvel[2], -control2 , 0, fd_torque = 0, kp = 500, kv = 3)
                    ctrl[3] = apply_torque(qpos[3], qvel[3], 1.571, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[4] = apply_torque(qpos[4], qvel[4], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    # # ctrl[5] = apply_torque(qpos[5], qvel[5], euler[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                    ctrl[11] = apply_torque(qpos[18], qvel[18], 1.6, 0, fd_torque = 0, kp = 100, kv = 0.1 )

                    trajectory = np.array([qpos[0:6]]).reshape(1,6)

                    with open(path,"a") as file:
                        np.savetxt(file, trajectory)    
                    sim.step()
                    viewer.render()
                    viewer2.render()
            # print(sim.data.body_xpos[palm_id] - np.array([-0.01882551, 0.0941442, 0.13251396]))
            for i in range(500):
                # print("grasp", sim.data.site_xpos[grasp_id])
                # print("obj", sim.data.body_xpos[obj_id])

                # sim.data.ctrl[0] = 0.36
                ctrl[0] = apply_torque(qpos[0], qvel[0], 0.36, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[1] = apply_torque(qpos[1], qvel[1], -control1 + 0.05, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[2] = apply_torque(qpos[2], qvel[2], -control2, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                
                ctrl[3] = apply_torque(qpos[3], qvel[3], 1.571, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[4] = apply_torque(qpos[4], qvel[4], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                # # ctrl[5] = apply_torque(qpos[5], qvel[5], euler[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[11] = apply_torque(qpos[18], qvel[18], 1.6, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                ctrl[6] = apply_torque(qpos[6], qvel[6], 1, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                ctrl[7] = apply_torque(qpos[9], qvel[9], 1, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                ctrl[8] = apply_torque(qpos[12], qvel[12], 1, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                ctrl[9] = apply_torque(qpos[15], qvel[15], 1, 0, fd_torque = 0, kp = 100, kv = 0.1 )        # r = R.from_matrix(sim.data.body_xmat[palm_id].reshape(3,3))
                # euler = r.as_euler('YXZ')
                # trajectory = np.array([sim.data.body_xpos[palm_id], euler]).reshape(1,6)
                trajectory = np.array([qpos[0:6]]).reshape(1,6)
                # print(trajectory)
                with open(path,"a") as file:
                    np.savetxt(file, trajectory)    
                # sim.data.ctrl[2] = -control + i * 0.05 / 999 
                # sim.data.ctrl[2] = 0.15
                # sim.data.ctrl[5] = 0.35
                # sim.data.ctrl[11] = 1.6
                sim.step()
                viewer.render()
                viewer2.render()


            for i in range(1000):
                ctrl[0] = apply_torque(qpos[0], qvel[0], 0.36, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                ctrl[1] = apply_torque(qpos[1], qvel[1], qpos[1], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[2] = apply_torque(qpos[2], qvel[2], qpos[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[3] = apply_torque(qpos[3], qvel[3], 1.571, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[4] = apply_torque(qpos[4], qvel[4], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                ctrl[6] = apply_torque(qpos[6], qvel[6], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[7] = apply_torque(qpos[9], qvel[9], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[8] = apply_torque(qpos[12], qvel[12], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[9] = apply_torque(qpos[15], qvel[15], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[10] = apply_torque(qpos[19], qvel[19], 0.4, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[11] = apply_torque(qpos[18], qvel[18], 1.6, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                # r = R.from_matrix(sim.data.body_xmat[palm_id].reshape(3,3))
                # euler = r.as_euler('YXZ')
                # trajectory = np.array([sim.data.body_xpos[palm_id], euler]).reshape(1,6)
                # print(trajectory)
                trajectory = np.array([qpos[0:6]]).reshape(1,6)
                with open(path,"a") as file:
                    np.savetxt(file, trajectory)    
                # ctrl[2] = apply_torque(qpos[2], qvel[2], qpos[2], 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                sim.step()          
                viewer.render()
                viewer2.render()

            # for i in range(1000):
            #     # print("grasp", sim.data.site_xpos[grasp_id])
            #     # print("obj", sim.data.body_xpos[obj_id])

            #     sim.step()          
            #     viewer.render()
            for i in range(2000):
                # print(sim.data.site_xpos[grasp_id])
                ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[3] = apply_torque(qpos[3], qvel[3], 1.571, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[4] = apply_torque(qpos[4], qvel[4], 0, 0, fd_torque = 0, kp = 1000, kv = 0.1 )

                ctrl[0] = apply_torque(qpos[0], qvel[0], qpos[0] - 0.03, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                ctrl[6] = apply_torque(qpos[6], qvel[6], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[7] = apply_torque(qpos[9], qvel[9], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[8] = apply_torque(qpos[12], qvel[12], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[9] = apply_torque(qpos[15], qvel[15], 1.57, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[10] = apply_torque(qpos[18], qvel[18], 0.3, 0, fd_torque = 0.1, kp = 100, kv = 0.1 )
                ctrl[11] = apply_torque(qpos[18], qvel[18], 1.6, 0, fd_torque = 0, kp = 100, kv = 0.1 )
                # r = R.from_matrix(sim.data.body_xmat[palm_id].reshape(3,3))
                # euler = r.as_euler('YXZ')
                # trajectory = np.array([sim.data.body_xpos[palm_id], euler]).reshape(1,6)
                # print(trajectory)
                # with open(path,"a") as file:
                #     np.savetxt(file, trajectory)    
                trajectory = np.array([qpos[0:6]]).reshape(1,6)
                # ctrl[5] = apply_torque(qpos[5], qvel[5], 0.35, 0, fd_torque = 0, kp = 1000, kv = 0.1 )
                with open(path,"a") as file:
                    np.savetxt(file, trajectory)
                # sim.data.ctrl[0] -= .0001
                sim.step()          
                viewer.render()           
                viewer2.render()

t1 = threading.Thread(target = socket_data)
t2 = threading.Thread(target = serial_data)

t1.start()
t2.start()


mujoco_sim()