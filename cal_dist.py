#!usr/bin/python
# -*- coding: utf-8 -*-
# auth: dushuai

import numpy as np
import cv2
import glob

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

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def calculate_Distance(focalLength_value, horizontal_distance, knownWidth):
    # 加载每一个图像的路径和焦距，以及要计算的两点坐标和两点间的实际距离
    # 计算到摄像头的距离,单位：米

    #image = cv2.imread(image)
    #cv2.imshow("image", image)
    #cv2.waitKey(300)

    #暂时计算两点间的水平距离
    # 计算得到目标物体到摄像头的距离
    distance_inches = distance_to_camera(knownWidth, focalLength_value, horizontal_distance)
    
    return distance_inches

    #cv2.putText(image, str(distance_inches) + ' m',(image.shape[1] - 300, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
    #cv2.imshow("image", image)


def calculate_Distance_point(focalLength_value, point1, point2, knownWidth):
    # 加载每一个图像的路径和焦距，以及要计算的两点坐标和两点间的实际距离
    # 计算到摄像头的距离,单位：米

    #image = cv2.imread(image)
    #cv2.imshow("image", image)
    #cv2.waitKey(300)

    #暂时计算两点间的水平距离
    horizontal_distance = point2[0] - point1[0]
    # 计算得到目标物体到摄像头的距离
    distance_inches = distance_to_camera(knownWidth, focalLength_value, horizontal_distance)

    cv2.putText(image, str(distance_inches) + ' m',
                (image.shape[1] - 300, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 255), 3)
    cv2.imshow("image", image)


if __name__ == "__main__":

    img_path = './dist_images/0.jpg'
    #save_path = './dist_images/0_calibresult5.jpg'

    #消除畸变
    undist_img = correct_distort(img_path)

    focalLength = 902 #焦距,单位为图像像素
    point1 = [279,219]
    point2 = [493,224]
    knownWidth = 0.5

    calculate_Distance(undist_img, focalLength, point1, point2, knownWidth)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


