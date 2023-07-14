#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author zhouhuawei time:2023/2/19
import cv2


face_name = 'hx'  # 该人脸的名字


# 加载OpenCV人脸检测分类器
face_cascade = cv2.CascadeClassifier("E:\OpenCV 4.7.0\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()  # 准备好识别方法LBPH方法


camera = cv2.VideoCapture(0)  # 0:开启摄像头
success, img = camera.read()  # 从摄像头读取照片
W_size = 0.1 * camera.get(3)  # 在视频流的帧的宽度
H_size = 0.1 * camera.get(4)  # 在视频流的帧的高度


def get_face():
    print("正在从摄像头录入新人脸信息 \n")
    picture_num = 0  # 设置录入照片的初始值
    while True:  # 从摄像头读取图片
        global success  # 设置全局变量
        global img  # 设置全局变量
        ret, frame = camera.read()  # 获得摄像头读取到的数据(ret为返回值,frame为视频中的每一帧)
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
        else:
            break

        face_detector = face_cascade  # 记录摄像头记录的每一帧的数据，让Classifier判断人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)  # gray是要灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors

        for (x, y, w, h) in faces:  # 制造一个矩形框选人脸(xy为左上角的坐标,w为宽，h为高)
            cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))
            picture_num += 1  # 照片数加一
            t = face_name
            cv2.imwrite("./data/1." + str(t) + '.' + str(picture_num) + '.jpg', gray[y:y + h, x:x + w])
            # 保存图像，将脸部的特征转化为二维数组，保存在data文件夹内
        maximums_picture = 100  # 设置摄像头拍摄照片的数量的上限
        if picture_num > maximums_picture:
            break
        cv2.waitKey(1)

get_face()