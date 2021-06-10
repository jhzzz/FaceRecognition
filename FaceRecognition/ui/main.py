import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import sys
from controller import Ui_MainWindow
from register import Ui_registerWindow
from choose import Ui_selectWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget
from database import DAO
from face_recognition import get_face_img, face_normalize, model, load_data, face_predict


# 这个窗口继承了用QtDesignner 绘制的窗口
class mainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(mainWindow, self).__init__()
        self.setWindowTitle('人脸识别考勤系统')
        self.window1 = registerWindow()
        self.window2 = selectWindow()
        self.setupUi(self)
        self.img_set_id = ''
        self.model = None

    def openWindow1(self):
        self.window1.show()

    def openWindow2(self):
        self.window2.show()
        # self.data = self.window2.data
        # print(self.data)

    def get_attend_situation(self):
        self.table_title.setText('出勤情况')
        data = self.window2.data
        self.set_table_model(data)

    def normalize_face(self):
        self.img_set_id = self.window1.get_img_set_id()
        normalize = face_normalize.FaceNormalize()
        normalize.run(self.img_set_id)
        self.pushButton_2.setStyleSheet("background: rgb(0,255,0)")
        # path = r'../data/normalized-faces/000005'
        # os.startfile(path)

    def load_model(self):
        dataset = load_data.Dataset()
        dataset.load_dataset()
        dataset.prepare_dataset()
        self.model = model.Model()
        self.model.build_model(dataset)
        self.model.train_model(dataset)
        self.model.evaluate_model(dataset)

    def save_model(self):
        if self.model is not None:
            if os.path.isfile('../data/model/model.h5'):
                os.remove('../data/model/model.h5')
            self.model.save_model('../data/model/model.h5')
            print('save model successfully')
        else:
            print('no exist model')

    def check_attend(self):
        predict = face_predict.Predict()
        predict.run()

    def get_daily_sheet(self):
        self.table_title.setText('日报表')
        data = DAO.get_daily_sheet()
        self.set_table_model(data)

    def get_weekly_sheet(self):
        self.table_title.setText('周报表')
        data = DAO.get_week_sheet()
        self.set_table_model(data)

    def get_monthly_sheet(self):
        self.table_title.setText('月报表')
        data = DAO.get_month_sheet()
        self.set_table_model(data)

    def set_table_model(self, data):
        model = QStandardItemModel()
        model.setColumnCount(3)
        model.setHorizontalHeaderItem(0, QStandardItem("人员ID"))
        model.setHorizontalHeaderItem(1, QStandardItem("姓名"))
        model.setHorizontalHeaderItem(2, QStandardItem("出勤时段"))
        for record in data:
            record_num = data.index(record)
            model.setItem(record_num, 0, QStandardItem(record['staff_id']))
            model.setItem(record_num, 1, QStandardItem(record['staff_name']))
            model.setItem(record_num, 2, QStandardItem(str(record['attend_time'])))
            model.item(record_num, 0).setTextAlignment(Qt.AlignCenter)
            model.item(record_num, 1).setTextAlignment(Qt.AlignCenter)
            model.item(record_num, 2).setTextAlignment(Qt.AlignCenter)
        self.tableView.setModel(model)
        self.tableView.setColumnWidth(0, 150)
        self.tableView.setColumnWidth(1, 150)
        self.tableView.setColumnWidth(2, 260)


class selectWindow(QWidget, Ui_selectWindow):
    def __init__(self):
        super(selectWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('查询出勤情况')
        self.data = []

    def choose(self):
        name = self.name_edit.text()
        start_time = self.start_edit.text()
        end_time = self.end_edit.text()
        data = DAO.get_attendance_by_name_and_periods(name, start_time, end_time)
        self.data = data
        # print(self.data)
        self.close()


class registerWindow(QMainWindow, Ui_registerWindow):
    def __init__(self):
        super(registerWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('录入人脸')
        self.img_set_id = ''

    def get_img_set_id(self):
        return self.img_set_id

    def registerFace(self):
        name = self.name_edit.text()
        sex = ''
        if self.female.isChecked():
            sex = '女'
        elif self.male.isChecked():
            sex = '男'
        phone_number = self.phone_edit.text()
        isSuccess = DAO.staff_register(name, sex, phone_number)
        if isSuccess:
            print('注册成功')
            img_set_id = DAO.get_img_set_id_by_name(name)
            self.img_set_id = img_set_id
            get_face = get_face_img.GetFaceImg(img_set_id)
            get_face.run()
            self.close()
        else:
            print('注册失败')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
