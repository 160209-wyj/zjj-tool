# -*- coding: utf-8 -*-
# @Time    : 2022/9/19 14:12
# @Author  : Zhilu Ding！！
# @FileName: eye_to_hand.py
# @Company: YiGongLi
from math import *

import cv2
import numpy as np
import transforms3d as tfs
# from tf.transformations import quaternion_from_euler, euler_from_quaternion

# K = np.array([[1720.04, 0, 628.632],
#             [0, 1720.61, 481.287],
#             [0, 0, 1]], dtype=np.float64) #彩色nano相机内参
# K = np.array([[640.944, 0, 653.26],
#             [0, 640.276, 369.501],
#             [0, 0, 1]], dtype=np.float64) #相机内参

K = np.array([[632.838, 0, 635.148],
            [0, 632.189, 369.609],
            [0, 0, 1]], dtype=np.float64) #相机内参

# distCoeffs = np.array([-0.0550687, 0.0643907, 9.23078e-05, 0.000314493, -0.0206842], dtype=np.float64) # 相机畸变系数
distCoeffs = np.array([-0.0551759, 0.0647504, 0.000415655, 0.000962474, -0.0203553], dtype=np.float64) # 相机畸变系数

chess_board_x_num = 11#棋盘格x方向格子数 (不在边沿的角点开始算起)
chess_board_y_num = 8#棋盘格y方向格子数 (不在边沿的角点开始算起)
chess_board_len = 20 #单位棋盘格长度,mm

board_z = 0 ## 棋盘格方在桌子上，高度642.5mm

#用于根据欧拉角计算旋转矩阵
def euler_to_R(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R

def quaternion_to_R(q0, q1, q2, q3):
    row1 = np.array([2*(q0*q0+q1*q1)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    row2 = np.array([2*(q1*q2+q0*q3), 2*(q0*q0+q2*q2)-1, 2*(q2*q3-q0*q1)])
    row3 = np.array([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0*q0+q3*q3)-1])
    return np.vstack((row1, row2, row3))

#用于根据位姿计算变换矩阵
def pose_to_RT(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = euler_to_R(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT = np.column_stack([R, t])  # 列合并
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT

# 用于根据四元数计算变换矩阵
def quaternion_to_RT(q0, q1, q2, q3, Tx, Ty, Tz):
    '''
    q0: 四元组角度w
    q1: 四元组x
    q2: 四元组y
    q3: 四元组z
    Tx: 平移x
    Ty: 平移y
    Tz: 平移z
    '''
    R = quaternion_to_R(q0, q1, q2, q3)
    t = np.array([[Tx], [Ty], [Tz]])
    RT = np.column_stack([R, t])  # 列合并
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT

def quat2mat(q0, q1, q2, q3, Tx, Ty, Tz):   ##将四元数转换为齐次转换矩阵
    rmat = tfs.quaternions.quat2mat([q0, q1, q2, q3])
    T = tfs.affines.compose(np.squeeze(np.asarray((Tx,Ty,Tz))), rmat, [1, 1, 1])
    return T

def mat2quat(RT):
    mat = tfs.quaternions.mat2quat(RT[0:3,0:3]),RT[:3,3:4]

    trans_x = mat[1][0][0]
    trans_y = mat[1][1][0]
    trans_z = mat[1][2][0]
    rotate_x = mat[0][1]
    rotate_y = mat[0][2]
    rotate_z = mat[0][3]
    rotate_w = mat[0][0]

    return trans_x, trans_y, trans_z, rotate_x, rotate_y, rotate_z, rotate_w

#用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path, chess_board_x_num, chess_board_y_num, K, chess_board_len, out_path=None):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    try:
        ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
        #print('corners', corners)       # corners.shape=(88, 1, 2)
        #print("------------", corners.shape)
        # print('corners.shape',corners.shape) #shape = (chess_board_x_num*chess_board_y_num, 1, 2)
        corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:, i] = corners[i, 0, :]
        #print('corner_points', corner_points, corner_points.shape)      # corner_points.shape=(2, 88)
        ## 标定板上的真是坐标，z=board_z，第一个角点作为0点
        object_points = board_z * np.ones((3, chess_board_x_num*chess_board_y_num), dtype=np.float64)
        flag = 0
        for i in range(chess_board_y_num):
            for j in range(chess_board_x_num):
                object_points[:2, flag] = np.array([(chess_board_x_num-j-1)*chess_board_len,
                                                    (chess_board_y_num-i-1)*chess_board_len])
                flag += 1
        print('object_points', object_points, object_points.shape)

        # cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners, ret)  #
        # cv2.imwrite(out_path, img)

        retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, K, distCoeffs=distCoeffs)

        #print('tvec', tvec)

        RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

        # print(retval, rvec, tvec)
        # print(img_path, 'RT: ')
        print('board to cam RT',RT)
        return RT
    except Exception as e:
        print('no corners:', e)
        RT = []
        return RT

def read_endpoint_pos(path):

    result = {}
    with open(path, 'r', encoding='utf-8') as f:

        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            number = line[0]
            pos = [float(line[7]), float(line[4]), float(line[5]), float(line[6]), float(line[1]), float(line[2]), float(line[3])]
            result.update({number: pos})

    return result

def quat2euler(quat):       # w, x, y, z]
    # r = R.from_quat(quat)
    # euler = r.as_euler('zyx', degrees=True)
    w, x, y, z = quat
    rx = atan2(2*(w*x+y*z),1-2*(x*x+y*y)) / np.pi * 180
    ry = asin(2*(w*y-z*x)) / np.pi * 180
    rz = atan2(2*(w*z+x*y),1-2*(z*z+y*y)) / np.pi * 180
    euler = [rx, ry, rz]
    return euler

if __name__ == '__main__':
    board_path = '/home/ygl/catkin_ws/src/ziru/files_record/board_files/eye_to_hand/batch3' #棋盘格图片存放文件夹
    # for i in range(1,21):
    #     path = board_path + '/chess_point_result/{}_result.jpg'.format(str(i))  
    #     get_RT_from_chessboard(board_path + '/{}.jpg'.format(str(i)), chess_board_x_num, chess_board_y_num, K, chess_board_len, path)
    '''
    有些棋盘格点检测得出来, 有些检测不了, 可以通过函数get_RT_from_chessboard的运行时间来判断
    '''
    good_picture = [i for i in range(1, 20)]# 存放可以检测出棋盘格角点的图片
    # good_picture = [1, 2, 3, 4, 5, 6, 7]# 存放可以检测出棋盘格角点的图片
    # good_picture = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]# 存放可以检测出棋盘格角点的图片
    # 9 12 18 19
    # good_picture = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 15, 17, 18, 19, 20] 
    file_num = len(good_picture)

    #计算board to cam 变换矩阵
    R_all_chess_to_cam = []
    T_all_chess_to_cam = []
    for i in good_picture:
        print('processing i: ', i)
        # if i == 15:
        #     continue
        image_path = board_path + '/'+str(i)+'.jpg'
        # print('image_path', image_path)
        RT = get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
        if len(RT) == 0:
            continue
        R_all_chess_to_cam.append(RT[:3, :3])
        T_all_chess_to_cam.append(RT[:3, 3].reshape((3, 1)))

    # print(T_all_chess_to_cam_1)

    # 获取end to base变换矩阵
    end_point_path = board_path + '/calibrate_end_point.txt' #从记录文件读取机器人六个位姿(end to base)
    pos_dict = read_endpoint_pos(end_point_path)

    # 计算转换成base to end变换矩阵
    R_all_base_to_end = []
    T_all_base_to_end = []
    print('pos_dict', pos_dict)
    for i in good_picture:
        # if i == 15:
        #     continue
        RT = quat2mat(pos_dict[str(i)][0],
                              pos_dict[str(i)][1],
                              pos_dict[str(i)][2],
                              pos_dict[str(i)][3],
                              pos_dict[str(i)][4],
                              pos_dict[str(i)][5],
                              pos_dict[str(i)][6])

        ## try
        if len(RT) == 0:
            continue
        RT_inv = np.linalg.inv(RT)

        print('base to end RT:', RT_inv)

        R_all_base_to_end.append(RT_inv[:3, :3])
        T_all_base_to_end.append(RT_inv[:3, 3].reshape((3, 1)))

    R, T = cv2.calibrateHandEye(R_all_base_to_end, T_all_base_to_end, R_all_chess_to_cam, T_all_chess_to_cam)#手眼标定
    RT_cam_to_base = np.column_stack((R, T))
    RT_cam_to_base = np.row_stack((RT_cam_to_base, np.array([0, 0, 0, 1])))#即为cam to base变换矩阵
    print('相机相对于base的变换矩阵为：')
    print(RT_cam_to_base)

    # cam_to_end_quat = [-235.3208054217088, -33.08400475129201, 66.99748638542076,
    #             0.4650963069347222, -0.5173711070447546, -0.5310800385460196, 0.48370089469518873]
    # RT_cam_to_end = quat2mat(cam_to_end_quat[6], cam_to_end_quat[3], cam_to_end_quat[4], cam_to_end_quat[5],
    #                         cam_to_end_quat[0], cam_to_end_quat[1], cam_to_end_quat[2])
    cam_to_base_quat = mat2quat(RT_cam_to_base)
    cam_to_base_euler = quat2euler([cam_to_base_quat[6], cam_to_base_quat[3], cam_to_base_quat[4], cam_to_base_quat[5]])
    print(cam_to_base_quat)
    print(cam_to_base_euler)

    record_axis = np.zeros((len(good_picture), 3))

    #结果验证，原则上来说，每次结果相差较小
    for i in range(len(good_picture)):

        RT_base_to_end = np.column_stack((R_all_base_to_end[i], T_all_base_to_end[i]))
        RT_base_to_end = np.row_stack((RT_base_to_end, np.array([0, 0, 0, 1])))
        # print(RT_base_to_end)

        RT_chess_to_cam = np.column_stack((R_all_chess_to_cam[i], T_all_chess_to_cam[i]))
        RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
        # print(RT_chess_to_cam)

        print('第', i, '次')
        # print(RT_chess_to_base)
        RT_chess_to_end = np.dot(np.dot(RT_base_to_end, RT_cam_to_base), RT_chess_to_cam)
        RT_chess_to_end = np.linalg.inv(RT_chess_to_end)
        
        print(RT_chess_to_end)
        print('')
        record_axis[i, 0:3] = RT_chess_to_end[0:3,3].reshape(1,3)

    print('各次拍照的平移向量')
    print(record_axis)
    print('各轴（x,y,z）标定误差：')
    print(np.std(record_axis,axis=0))
    print(np.max(record_axis, axis=0)-np.min(record_axis, axis=0))

