import os

import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/dlib-data/shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_face(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) != 0:
        for i, face in enumerate(faces):
            left = face.left()
            right = face.right()
            top = face.top()
            bottom = face.bottom()
            print("面部矩阵：%d %d %d %d" % (left,right,top,bottom))
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, "Face #{}".format(i + 1), (left - 10, top - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, face).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                print(idx, pos)

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 4, (0, 255, 0), 1)
                # 利用cv2.putText输出1-68
                # cv2.putText(img, str(idx), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "Not Found Face", (10, 30), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("image", 0)
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    path = "face_img/yy.jpg"
    path2 = "../data/screenshots/person_10/face_8_2021-06-01-11-39-39.jpg"
    detect_face(path2)
    # image_dir = "../data/screenshots/person_1"
    # for file in os.listdir(image_dir):
    #     print(file)
    #     img = cv2.imread(os.path.join(image_dir, file))
    #     if img is None:
    #         print(os.path.join(image_dir, file))
    #         break
