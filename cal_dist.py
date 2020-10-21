#!usr/bin/python
# -*- coding: utf-8 -*-
# auth: dushuai

import numpy as np
import cv2
import glob


#标定函数
def camera_calibrate(images_path):
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob(images_path)
    for fname in images:
        img = cv2.imread(fname)
        cv2.imshow('img',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        print(ret)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            #print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.imshow('img', img)
            cv2.waitKey(2000)

    print(len(img_points))
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("ret:", ret)
    print("mtx:\n", mtx) # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs ) # 平移向量  # 外参数

    print("-----------------camera_calibrate----------------------")


#消除畸变方法一
def correct_distort(image_path):
    img = cv2.imread(image_path)
    #摄像头内参矩阵和畸变矩阵
    mtx = np.array([[902.07127062, 0, 335.52703493],
                    [0, 904.96104128, 225.22457156],
                    [0, 0, 1]])
    dist = np.array([[-3.45111160e-01, -8.49634274e-01,
                      -1.96772943e-03, -1.58704961e-03, 5.00129970e+00]])
    #使用getOptimalNewCameraMatrix调整视场，为1时视场大小不变,小于1时缩放视场
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #调用undistort函数，消除畸变
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    #cv2.imwrite(save_path, dst)
    return dst
    '''
    #不使用getOptimalNewCameraMatrix
    dst = cv2.undistort(img, mtx, dist, None)
    cv2.imwrite(save_path, dst)
    '''
    print("------------------correct_distort---------------------")


#消除畸变方法二
def correct_distort_remap(image_path, save_path):
    img = cv2.imread(image_path)
    # 摄像头内参矩阵和畸变矩阵
    mtx = np.array([[902.07127062, 0, 335.52703493],
                    [0, 904.96104128, 225.22457156],
                    [0, 0, 1]])
    dist = np.array([[-3.45111160e-01, -8.49634274e-01,
                      -1.96772943e-03, -1.58704961e-03, 5.00129970e+00]])

    # 使用getOptimalNewCameraMatrix调整视场，为1时视场大小不变,小于1时缩放视场
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 使用remap消除畸变
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(save_path, dst)


def distance_to_camera(perWidth, knownWidth, focalLength):
    # 计算物体距摄像头的距离,单位：米
    # 参数：像素宽perWidth，物体已知距离knownWidth，和焦距focalLength
    return (knownWidth * focalLength) / perWidth


def calculate_distance(left, right, knownWidth, focalLength_value):
    # 计算物体到摄像头的距离,单位：米
    # 参数：图像中两点point1, point2，物体已知距离knownWidth，焦距focalLength_value
    # 暂时计算两点间的水平距离
    horizontal_distance = right - left
    # 计算得到目标物体到摄像头的距离
    distance_inches = distance_to_camera(knownWidth, focalLength_value, horizontal_distance)
    return distance_inches
    '''
    # 在图片中显示
    image = cv2.imread(image)
    cv2.imshow("image", image)
    cv2.waitKey(300)
    cv2.putText(image, str(distance_inches) + ' m',
                (image.shape[1] - 300, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 255), 3)
    cv2.imshow("image", image)
    '''


# 找出两帧中相同的车，并将第二帧作为初始状态
car_id = 0


def search_multi_car(pair, frame_time_diff, knownWidth=1.8, focalLength_value=903):
    # car_id = 0  # 初始化全局id
    global car_id
    car_posi_0 = pair[0]  # 第一帧所有车的位置信息[{'frame_id', 'left', 'right', ..., 'center', 'car_id'}]
    car_posi_1 = pair[1]  # 第二帧所有车的位置信息[{'frame_id', 'left', 'right', ..., 'center', 'car_id'，'car_2_cam', 'speed'}]
    for car_0 in car_posi_0:
        car_0.update({'car_id': car_id})  # 为每一辆车分配一个id
        car_id += 1
    all_car_dist = []  # 第二帧中每一辆车与前一帧所有车的距离
    for car_1 in pair[1]:  # 得到all_car_dist
        per_car_dist = []
        for car_inform_0 in car_posi_0:
            # dist = np.sqrt(sum(np.power((one['center'] - car_infor[2]), 2)))
            dist = car_1['center'][1] - car_inform_0['center'][1]
            if dist < 0:  # 为负，则置为99999
                dist = 99999
            width_car = abs(car_1['center'][0] - car_inform_0['center'][0])
            if width_car > 50:
                dist = 99999
            # car_dist = [car_inform_0['car_id'], dist]  # car_id和距离
            per_car_dist.append(dist)
        all_car_dist.append(per_car_dist)
    car_id_2 = [0, 1, 2, 3]
    all_car_dist = np.array(all_car_dist)
    print(all_car_dist)
    num_car_0 = len(all_car_dist[0, :])  # 第一帧车的数量
    print('num_car_0: ', num_car_0)
    # b = list(chain(*a[:,:,-1])) # 将二维数组转为一维数组
    for i, car_1 in enumerate(car_posi_1):
        min_dist = min(all_car_dist[i, :])
        print('min_dist:', min_dist)
        print('----------')
        min_dist_car_index = list(all_car_dist[i, :]).index(min_dist)
        if min_dist < 20:
            car_1.update({'car_id': min_dist_car_index})
            all_car_dist[:, min_dist_car_index] = 9999
            # 计算距离和速度
            car0_2_cam = calculate_distance(car_posi_0[min_dist_car_index]['left'], car_posi_0[min_dist_car_index]['right'], knownWidth, focalLength_value)
            car1_2_cam = calculate_distance(car_1['left'], car_1['right'], knownWidth, focalLength_value)
            car_speed = (car1_2_cam - car0_2_cam) / frame_time_diff
            car_1.update({'car_speed': car_speed})
            car_1.update({'car_2_cam': car1_2_cam})
        else:
            car_id += 1
            car_1.update({'car_id': car_id})
            car1_2_cam = calculate_distance(car_1['left'], car_1['right'], knownWidth, focalLength_value)
            car_1.update({'car_2_cam': car1_2_cam})
        print('=============')
    print(all_car_dist)
    print(car_posi_0)
    print(car_posi_1)

    return car_posi_0, car_posi_1


if __name__ == "__main__":

    focalLength = 903  # 焦距,单位为图像像素
    knownWidth = 1.8  # 实际车宽




