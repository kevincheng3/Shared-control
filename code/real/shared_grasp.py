import socket
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
import sys

import serial
import time
import serial.tools.list_ports

import pyrealsense2 as rs
import cv2



client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

ip_port=('127.0.0.1',8000)
ur_ip = "192.168.137.123"

client.connect(ip_port)
data = np.zeros(254)
pos = np.zeros(3) 
euler = np.array([2, 0, 0])
axis_angle = np.array([-0.6739679061437915, -1.4925770675085284, 0.724235477027583])

Angle = [0.0]*20
Last_Angle = [0.0]*20
Hex_str0 = bytes.fromhex('c1 c4')
Hex_str7 = bytes.fromhex('c3 c6')
# ser_hand=serial.Serial("COM7", 115200) 
# ser_glove = serial.Serial("COM5", 115200, timeout = 1)

# noitom_data = np.zeros((1000,4))
# ur_data = np.zeros((1000,4))

def rotation(theta_y, theta_x, theta_z):
    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_y.dot(rot_x).dot(rot_z)
    return R

def socket_data():
    """
    pass the end effector to the robot arm
    output: position command or velocity command
    """

    global data
    global pos
    global axis_angle
    global command
    global centroid
    global aligned
    global grasped
    # global euler
    l1 = np.array([-0.265, 0, 0])
    l2 = np.array([-0.26, 0, 0])

    aligned = False
    grasped = False
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    centroid = np.array([320, 240])


    boundary = np.array([0.295, 0.305])
    f = 613.243
    node = 0.3
    obj_infor = []
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        blurred = cv2.GaussianBlur(color_image, (3, 3), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # images = np.hstack((binary, gray))
        # print(len(contours))
        # print(command[2])
        # if len(contours) == 1 and boundary[0]<command[2]<boundary[1]:
        if len(contours) == 1 and not grasped:
            # print("done")
            # boundary = boundary - 0.05
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # print(box[0], box[2], command[2])
                # print((box[0] + box[2])/2)
                centroid = (box[0] + box[2])/2
                if abs(centroid[0]-320) > 3 and abs(centroid[0]-240) > 3 and not aligned:
                    x = -(centroid[0] -320) / abs((centroid[0] -320)) * 0.01 if abs((centroid[0] -320))>1e-10 else 0
                    y = (centroid[1] -240) / abs((centroid[1] -240)) * 0.01 if abs((centroid[1] -240))>1e-10 else 0
                    # z方向到达指定位置
                    # rtde_c.speedL([x, y, 0, 0, 0, 0], 0.1, time=1.0/20)
                    vel = [x, y, 0, 0, 0, 0]
                else :
                    aligned = True
                cv2.drawContours(color_image, [box], 0, (0, 255, 0), 3)
        # print(command[2])
            if command[2] > 0.15 and not grasped:
                if boundary[0] < command[2] < boundary[1]  and aligned:
                    vel = [x, y, 0, 0, 0, 0]

                    # print("done")
                    boundary = boundary - 0.05
                    obj_infor.append([box[0], box[2], command[2]])
                    # print(obj_infor)
                    if len(obj_infor) >1:
                        delta_z = obj_infor[-2][2] - obj_infor[-1][2]
                        x1 = abs(obj_infor[-2][0][0] - obj_infor[-2][1][0])
                        x2 = abs(obj_infor[-1][0][0] - obj_infor[-1][1][0])
                        x = x1 * x2 * (delta_z) /(f*(x2-x1))
                        z = f * x / x2
                        print(x, z, obj_infor[-1][2])
            else:
                pass
                """
                reach the default grasp point
                """




        # cv2.imshow('RealSense2', color_image)
        # # cv2.imshow('gray', images)
        # cv2.imshow('gray', binary)
        # # cv2.imshow('thresh', thresh)
        # cv2.waitKey(1)


        else:
            a=client.recv(1024)
        # data 数据的校验 不能全为0
            if len(a) == 1016:
                length = len(a)/4
                data = struct.unpack('<' + str(int(length)) + 'f', a)

                matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
                matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)
                matrix3 = rotation(data[16 * 6 + 3] / 180 * 3.14, data[16 * 6 + 4] / 180 * 3.14, data[16 * 6 + 5] / 180 * 3.14)
                matrix = matrix1.dot(matrix2).dot(matrix3)

                pos1 = matrix1.dot(l1)
                pos2 = matrix1.dot(matrix2).dot(l2)
                pos = pos1 + pos2
                base_matrix = np.array([[0.0, 0.0, -1], [-1, 0, 0], [0, 1, 0]])
                pos = base_matrix.dot(pos) + np.array([0.3, 0, 0])

                # matrix1 = rotation(data[14 * 6 + 3] / 180 * 3.14, data[14 * 6 + 4] / 180 * 3.14, data[14 * 6 + 5] / 180 * 3.14)
                # matrix2 = rotation(data[15 * 6 + 3] / 180 * 3.14, data[15 * 6 + 4] / 180 * 3.14, data[15 * 6 + 5] / 180 * 3.14)

                r = R.from_matrix(matrix)

                euler = r.as_euler("YXZ")

                if sum(euler) < 1e-2:
                    pass
                else:
                    euler[2] = np.clip(euler[2], 0.1, 0.2)

                    r = R.from_euler("YXZ", euler)
                    matrix = r.as_matrix()
                    x0 = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]) # 人手本地坐标系变换到传感器的坐标系

                    x1 = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]) # 运动捕获坐标系变换到机械臂坐标系

                    x2_i = np.array([[0.7071067811865476, -0.7071067811865476, 0], 
                    [0.7071067811865476, 0.7071067811865476, 0], 
                    [0, 0, 1]])  #机械手本地坐标系变换到机械臂坐标系的逆

                    final_matrix = x0.dot(matrix).dot(x1).dot(x2_i)
                    r = R.from_matrix(final_matrix)
                    axis_angle = r.as_rotvec()

            else:
                pass
            
            """
                return endefforctor pose
            """



def hand_control():
    # ser_hand=serial.Serial("COM7", 115200) #使用USB连接串行口
    global Angle
    while True:
        little_pos = int(Angle[17] / 1.571 * 0x0cd0)
        ring_pos = int(Angle[13] / 1.571 * 0x0df0)
        middle_pos = int(Angle[9] / 1.571 * 0x1000)
        index_pos = int(Angle[5] / 1.571 * 0x0fc0)
        # little_pos = int(0 / 1.571 * 0x0cd0)
        # ring_pos = int(0 / 1.571 * 0x0df0)
        # middle_pos = int(0 / 1.571 * 0x1000)
        # index_pos = int(0 / 1.571 * 0x0fc0)   
        thumb1_pos =int(Angle[3] / 1.571 * 0xf00) #2500   
        # thumb1_pos =int(0 / 1.571 * 0xf00) #2500   
        # thumb2_pos =int((Angle[0]-0.3) * 3 / 1.571 * 0x200)
        thumb2_pos =int(max((Angle[0]-0.35) * 4,0)/ 1.571 * 0x200)

        control = struct.pack('>BBHHHHHHB', 0xc2, 0xc5, little_pos, ring_pos, middle_pos, index_pos, thumb1_pos, thumb2_pos, 0x00)
        ser_hand.write(control)
        time.sleep(0.01)

def glove_data():
    # ser_glove = serial.Serial("COM5", 115200, timeout = 1)
    global Angle
    while True:
        Last_Angle = Angle.copy()
        if int(ser_glove.read().hex(),16) == 253:
            if int(ser_glove.read().hex(),16) == 255:
                for i in range(20):
                    Data = int(ser_glove.read(2).hex(),16)
                    Angle[i] = (-Data+512) * 6.28 / 1024
                    if i == 0 :
                        Angle[i] = -Angle[i]
                    if i == 18 or i == 19:
                        Angle[i] = -Angle[i]
                    if Angle[i]<0:
                        Angle[i]=0
                    if abs(Angle[i] - Last_Angle[i]) < 0.1:
                        Angle[i] = Last_Angle[i]

def ur_move():
    global axis_angle
    rtde_c = RTDEControl(ur_ip)
    rtde_r = RTDEReceive(ur_ip)
    dt = 1/500  # 2ms
    lookahead_time = 0.03
    gain = 800

    while 1:

        command = np.array([-0.55, -0.0, 0.3, 0, 0 ,0 ])

        command[3:6] = axis_angle
        command[0] += pos[0] / 2
        command[1] += pos[1] / 2
        command[2] += pos[2] / 2

        command[0] = np.clip(command[0], -0.65, -0.3)
        command[1] = np.clip(command[1], -0.25, 0.25)
        command[2] = np.clip(command[2], 0.12, 0.45)

        # command[3:6] = axis_angle
        start = time.time()    


        rtde_c.moveL(command, 0.1, 0.1)
        # start = time.time()
        # rtde_c.servoL(command, 0.5, 0.5, dt, lookahead_time, gain)


        end = time.time()
        duration = end - start

        if duration < dt:
            time.sleep(dt - duration)
        # i = i+1
        # print(i)
        # time.sleep(0.1)
        
# socket_data()

try:
    # t1 = threading.Thread(target = euler_data)
    t2 = threading.Thread(target = socket_data)
    # t3 = threading.Thread(target = hand_control)
    # t4 = threading.Thread(target = glove_data)

    # t1.start()
    t2.start()
    # t3.start()
    # t4.start()

    ur_move()
except KeyboardInterrupt:
    # ser_hand.write(Hex_str0)
    time.sleep(2)
    # ser_hand.write(Hex_str7)
    # time.sleep(2)
    # ser_hand.close()
    # ser_glove.close()
# # t1 = threading.Thread(target = euler_data)
# # t2 = threading.Thread(target = socket_data)
# t3 = threading.Thread(target = hand_control)
# t4 = threading.Thread(target = glove_data)

# # t1.start()
# # t2.start()
# t3.start()
# t4.start()

# # ur_move()