#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2023/02/21 21:53:04
@Author  :   edward
@Version :   1.0
@Contact :   edward
"""

import cv2
import numpy as np
import os
from sklearn import preprocessing


class PhotoGraph:
    """
    拍照保存
    """

    def __init__(self, photo_path, photo_name):
        """
        :param photo_path: 图片存储路径,路径要存英文
        :param photo_name: 图片明名
        """
        self.photo_path = photo_path
        self.photo_name = photo_name

    def shot(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow("PhotoGraph", frame)
            c = cv2.waitKey(1) & 0xFF
            if c == ord("s"):
                cv2.imwrite(os.path.join(os.getcwd(), self.photo_path, "{}{}".format(self.photo_name, ".jpg")), frame)
                break
        cap.release()
        cv2.destroyAllWindows()


class LabelEncoder(object):
    """标签编码，将同一个人物的图片名字用数字表示"""
    def __init__(self):
        self.le = preprocessing.LabelEncoder()

    # 将单词转换成数字编码的方法
    def encode_labels(self, label_words):
        self.le.fit(label_words)

    # 将输入单词转换成数字
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # 将数字转换为单词
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]


class FaceDistinguish:
    """人脸识别"""
    def __init__(self, photo_path):
        # 加载人脸探测级联文件
        self.face_cascade = cv2.CascadeClassifier('E:\OpenCV 4.7.0\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
        self.data_transformation = LabelEncoder()
        self.photo_path = photo_path
        self.DATA = None

    def get_images_information(self):
        """
        迭代获取image_photograph文件夹下所有 jpg图片 和 对应的名字
        :return:
        """
        # 初始化images存放图片路径， information存放图片信息
        images = []
        information = []

        # 获取路径文件
        for root, dirs, files in os.walk(self.photo_path):
            # 获取每个jpg文件
            root = root.replace("/", "\\")
            for filename in [x for x in files if x.endswith(".jpg")]:
                # 每个图片路径，名字加到对应列表
                images.append(os.path.join(os.getcwd(), root, filename))
                information.append(root.split('\\')[-1])

        # 对名字进行调用标签编码方法
        self.data_transformation.encode_labels(information)
        return images, information

    def get_features(self, images, information):
        """
        采集人脸部分图片，名字进行转数字（标签处理）
        :param images: 所有图片路径的列表
        :param information: 所有图片对应的名字列表
        :return:
        """
        # 初始化：face_images_info存储每张图片人脸部分
        face_images_info = []
        face_name_info = []
        for num in range(len(images)):
            # 将当前图像读取成灰度格式
            ret = cv2.imread(images[num], 0)

            # 获取脸部信息
            faces = self.face_cascade.detectMultiScale(ret, 1.1, 2, minSize=(100, 100))

            # 采集脸部数据，名字进行编码
            for (x, y, w, h) in faces:
                face_images_info.append(ret[y:y + h, x:x + w])
                face_name_info.append(self.data_transformation.word_to_num(information[num]))
        return face_images_info, face_name_info

    def testing(self):
        """"
        训练人脸识别器，采集人脸数据
        再通过摄像头采集人脸数据与内存中的数据做比较
        """
        # 生成局部二值模式直方图人脸识别器
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 图片准备
        images, information = self.get_images_information()

        # 获取脸部信息，人物信息
        face_images_info, face_name_info = self.get_features(images, information)

        # 训练人脸识别器
        recognizer.train(face_images_info, np.array(face_name_info))

        # 0：是开启笔记本内置摄像头
        cap = cv2.VideoCapture(0)
        while True:

            # 读取摄像头每一帧
            ret, frame = cap.read()

            # 判断摄像头是否开启
            if not ret:
                break

            # 调整frame（非必要）
            frame = cv2.resize(frame, dsize=None, fx=1, fy=1,
                               interpolation=cv2.INTER_AREA)

            # 转灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 识别脸部,使用默认参数，可自行调节
            faces = self.face_cascade.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                # 在人脸部位 画框和圈
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                cv2.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)

                # 做数据比较
                predicted_index, conf = recognizer.predict(gray[y:y + h, x:x + w])

                # conf<50:可靠，>80不可靠
                if conf < 50:
                    # 在脸上方写出名字
                    predicted_person = self.data_transformation.num_to_word(predicted_index)
                    cv2.putText(frame, predicted_person, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 255, 255), 4)
                else:
                    cv2.putText(frame, "UKNOW", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 255, 255), 4)
            # 显示画面，并且检测键盘，当按下Esc键查询结束
            cv2.imshow('face detector', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
        # 释放资源，关闭窗口
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = FaceDistinguish(os.path.join(os.getcwd(), "image_photograph"))
    app.testing()

