import os

from model import Model
import numpy as np
import cv2
from face_normalize import FaceNormalize
import dlib

detector = dlib.get_frontal_face_detector()

def face_recognition(model_path, stream):
    model = Model()
    model.load_model(model_path)
    # normalize = FaceNormalize()
    while stream.isOpened():
        ret, img_camera = stream.read()
        img_gray = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray, 1)

        if len(faces) != 0:
            for i, face in enumerate(faces):
                left = face.left()
                right = face.right()
                top = face.top()
                bottom = face.bottom()

                img_cut = img_camera[top: bottom, left: right]
                if img_cut.shape[0] != 0 and img_cut.shape[1] != 0:
                    # img_rotate = normalize.face_rotate(img_cut)
                    # img_normalize = normalize.face_normalize(img_cut)
                    img_normalize = cv2.resize(img_cut, (64, 64))
                    # print(img_normalize.shape)
                    cv2.rectangle(img_camera, (left, top), (right, bottom), (0, 255, 0), 2)
                    predict_probability, faceID = model.face_predict(img_normalize)
                    print(predict_probability, faceID)
                    person_path = 'data/screenshots'
                    for person_name in os.listdir(person_path):
                        if person_name.endswith(str(faceID)):
                            cv2.putText(img_camera, person_name, (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("camera", img_camera)
        k = cv2.waitKey(10)
        if k == 27:
            break


def run():
    model_path = 'data/model/model.h5'
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_recognition(model_path, cap)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
    # path = "data/normalized-faces/person_2/4.jpg"
    # image = cv2.imread(path)
    # image = np.array(image)
    # image = image.reshape((1, 64, 64, 3))
    #
    # image = image.astype('float32')
    # image /= 255
    # model = Model()
    # model.load_model('data/model/model.h5')
    # my_model = model.model
    # result = my_model.predict_proba(image)
    # print('result:', result)
    # result = my_model.predict_classes(image)
    # print(result)
