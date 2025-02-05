from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QMainWindow
# Ui_MainWindow是使用Qt Designer创建的界面类
from gui_2_ui import Ui_MainWindow  #从ui文件导入Ui_MainWindow类
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import wfdb
import numpy as np
import pywt
import classification as cla
import tensorflow as tf
import pymysql
from pymysql import Error

def denoise(data):

    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    cA9.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.connect_to_database()# 建立数据库连接
        # 在这里可以连接信号和槽，或者添加其他初始化代码
        # 创建Matplotlib图形画布并添加到布局中
        #constrained_layout试图自动调整图表的布局
        self.canvas = FigureCanvas(Figure(figsize=(5, 3), constrained_layout=True))
        self.layout = QtWidgets.QVBoxLayout(self.groupBox)  # groupBox使用了垂直布局
        self.layout.addWidget(self.canvas)

        self.canvasGroupBox2 = FigureCanvas(Figure(figsize=(5, 3), constrained_layout=True))
        self.layoutGroupBox2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.layoutGroupBox2.addWidget(self.canvasGroupBox2)

        #按钮的连接事件
        self.pushButton.clicked.connect(self.loadAndDisplayECG)
        self.pushButton.clicked.connect(self.performDetection)
        #左移按钮
        self.left.clicked.connect(self.shiftLeft)
        #右移按钮
        self.right.clicked.connect(self.shiftRight)

        # 初始化变量
        self.ecgData = None
        self.currentIndex = 0
        self.dataLength = 2000  # ECG图片显示数据长度

        # 填充下拉菜单选项
        self.comboBox.addItems(['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                           '116', '117', '118', '119', '121', '122', '123', '124', '200', '203', '205', '214', '208',
                           '209', '210', '212', '213', '219', '221', '222', '228', '231', '232', '233', '234'])

        # self.page_1是切换到page_3的按钮，self.page_2是切换到page的按钮
        # self.stackedWidget是QStackedWidget的对象名
        # 连接按钮点击信号到切换页面的槽函数
        self.page_1.clicked.connect(self.showPage3)
        self.page_2.clicked.connect(self.showPage)

        # 全局病人编号列表
        self.patient_numbers = []

        self.add.clicked.connect(self.add_patient_info_to_database)

        # self.tableWidget 是QTableWidget 对象
        self.tableWidget.cellClicked.connect(self.cell_was_clicked)

        #用于存储检测结果的字典
        self.diagnosis_results = {}

        #删除操作 获取当前选择的行
        self.tableWidget.itemSelectionChanged.connect(self.rowSelected)

        #连接“remove_patient”按钮的clicked信号到一个槽函数，该函数负责执行删除操作
        self.remove_patient.clicked.connect(self.removeSelectedPatient)

        self.tableWidget.cellChanged.connect(self.recordChange)  # 记录更改
        self.change_patient.clicked.connect(self.commitChanges)  # 提交更改
        self.changes = {}  # 存储表格更改

    def showPage3(self):
        #page_3为页面一
        # page_3在QStackedWidget中的索引是1
        self.stackedWidget.setCurrentIndex(1)

    def showPage(self):
        # page为页面二
        # page在QStackedWidget中的索引是0
        self.stackedWidget.setCurrentIndex(0)


    # def onPushButtonClicked(self):
    #     pass


    def displayECG(self, data):
        # 清除之前的绘图
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()
        # 设置数据及其对应的x轴范围
        x_range = range(self.currentIndex, self.currentIndex + len(data))
        ax.plot(x_range, data)
        # 设置x轴界限
        ax.set_xlim(self.currentIndex, self.currentIndex + len(data) - 1)
        ax.set_title('ECG去噪后信号', fontproperties='SimHei')  # 指定字体为黑体
        self.canvas.draw()

    def displayECGInGroupBox2(self, data):
        # 清除之前的绘图
        self.canvasGroupBox2.figure.clear()
        ax2 = self.canvasGroupBox2.figure.subplots()
        # 设置数据及其对应的x轴范围
        x_range = range(self.currentIndex, self.currentIndex + len(data))
        ax2.plot(x_range, data)
        # 设置x轴界限
        ax2.set_xlim(self.currentIndex, self.currentIndex + len(data) - 1)
        ax2.set_title('ECG原始信号', fontproperties='SimHei')  # 指定字体为黑体
        self.canvasGroupBox2.draw()


    def updateECGDisplay(self):
        dataSegment = self.ecgData[self.currentIndex:self.currentIndex + self.dataLength]
        data_denoise = denoise(data = self.ecgData)
        dataSegment_denoise = data_denoise[self.currentIndex:self.currentIndex + self.dataLength]
        self.displayECG(dataSegment_denoise)
        self.displayECGInGroupBox2(dataSegment)  #在groupBox_2中显示相同的数据片段


    def loadAndDisplayECG(self):
        selectedNumber = self.comboBox.currentText()
        recordPath = f'D:/大学/毕设/MIT-BIH-360/{selectedNumber}'
        record = wfdb.rdrecord(recordPath, channel_names=['MLII'])
        self.ecgData = record.p_signal.flatten()
        self.currentIndex = 0
        self.updateECGDisplay()  # 使用更新后的方法来显示数据


    #心电图左移
    def shiftLeft(self):
        if self.ecgData is not None and self.currentIndex > 0:
            self.currentIndex = max(0, self.currentIndex - 100)  # 向左移动100个采样点
            self.updateECGDisplay()

    #心电图右移
    def shiftRight(self):
        if self.ecgData is not None and self.currentIndex + self.dataLength < len(self.ecgData):
            self.currentIndex = min(len(self.ecgData) - self.dataLength, self.currentIndex + 100)  # 向右移动100个采样点
            self.updateECGDisplay()

    def ensurePatientExists(self, patient_id):
        # 检查病人是否已在数据库中
        self.cursor.execute("SELECT id FROM patients WHERE id = %s", (patient_id,))
        if not self.cursor.fetchone():  # 如果病人不存在
            # 插入新病人记录到数据库
            self.cursor.execute("INSERT INTO patients (id) VALUES (%s)", (patient_id,))
            self.connection.commit()
            print(f"Patient {patient_id} added to the database.")

    def detectArrhythmia(self):
        selectedNumber = self.comboBox.currentText()
        # 确保病人存在于数据库中
        self.ensurePatientExists(selectedNumber)

        if selectedNumber not in self.patient_numbers:
            self.patient_numbers.append(selectedNumber)
            self.updatePatientDropdown()  # 用于更新页面二的下拉选单

        X, Y = cla.loadData_test(selectedNumber)
        loaded_model = tf.keras.models.load_model('D:\大学\毕设\保存的模型/model_saved_4')
        predictions = loaded_model.predict(X)
        Y_pred_classes = np.argmax(predictions, axis=1)

        # 定义心拍种类名称
        categories = {
            0: "正常心拍",
            1: "房性早搏",
            2: "室性早搏",
            3: "左束支传导阻滞",
            4: "右束支传导阻滞"
        }

        # 统计每个类别的数量
        unique, counts = np.unique(Y_pred_classes, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # 构造结果字符串
        result_str = ""
        for class_id, count in class_counts.items():
            if count > 0:  # 仅包含数量不为0的类别
                if result_str != "":  # 如果不是第一个条目，添加换行符
                    result_str += "\n"
                result_str += f"{categories[class_id]}有{count}个"

                # 将结果存储到字典中，键为selectedNumber，值为result_str
                self.diagnosis_results[selectedNumber] = result_str

        heartbeat_count = len(Y_pred_classes)  # 使用Y_pred_classes的长度作为心拍计数
        data_path = f"D:/大学/毕设/MIT-BIH-360/selectedNumber"  # 示例路径

        # 插入数据库
        self.insertECGData(selectedNumber, heartbeat_count, data_path, result_str)

        return result_str if result_str != "" else "没有检测到异常心拍。"


    def performDetection(self):
        # 调用检测方法获取结果
        detectionResult = self.detectArrhythmia()
        # 更新结果标签的文本
        self.result_3.setText(detectionResult)

    # 页面二监听病人信息更新信号，并更新下拉选单的示例代码片段
    def updatePatientDropdown(self):
        # 清空页面二的下拉选单
        self.number.clear()
        # 使用 MainWindow 实例的 patient_numbers 来更新下拉选单
        for patient_number in self.patient_numbers:
            self.number.addItem(patient_number)


    def add_patient_info_to_database(self):
        selected_number = self.number.currentText()
        age = self.age.text()
        gender = self.gender.text()
        medicine = self.medicine.text()
        try:
            #query = "INSERT INTO patients (id, age, gender, medicine) VALUES (%s, %s, %s, %s)"
            query = """
            INSERT INTO patients (id, age, gender, medicine) 
            VALUES (%s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE 
            age = VALUES(age), gender = VALUES(gender), medicine = VALUES(medicine)
            """
            self.cursor.execute(query, (selected_number, age, gender, medicine))
            self.connection.commit()
            print("Patient info successfully added to database")

            # 添加到QTableWidget
            current_row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row_count)
            self.tableWidget.setItem(current_row_count, 0, QtWidgets.QTableWidgetItem(selected_number))
            self.tableWidget.setItem(current_row_count, 1, QtWidgets.QTableWidgetItem(age))
            self.tableWidget.setItem(current_row_count, 2, QtWidgets.QTableWidgetItem(gender))
            self.tableWidget.setItem(current_row_count, 3, QtWidgets.QTableWidgetItem(medicine))

        except Error as e:
            print("Error while adding patient info to MySQL", e)

    def connect_to_database(self):
        try:
            self.connection = pymysql.connect(
                host='localhost',
                port=3306,
                database='ecg',
                user='root',
                password='123456'
            )
            print("Connected to MySQL database...")
            self.cursor = self.connection.cursor()
        except Error as e:
            print("Error while connecting to MySQL", e)

    def cell_was_clicked(self, row, column):
        patient_number_item = self.tableWidget.item(row, 0)  # 第一列的 QTableWidgetItem
        if patient_number_item is not None:
            patient_id = patient_number_item.text()
            # 有了被点击行的病人编号，可以根据这个编号查询心律失常检测结果
            result_str = self.queryDiagnosisResult(patient_id)
            self.label_8.setText(result_str)

    def queryDiagnosisResult(self, number):
        # 从字典中获取检测结果，如果编号不存在则返回默认信息
        return self.diagnosis_results.get(number, "未找到该病人的检测结果。")

    def rowSelected(self):
        self.currentRow = self.tableWidget.currentRow()


    #删除病人数据
    def removeSelectedPatient(self):
        if self.currentRow >= 0:  # 确保有行被选中
            # 从表格中获取病人编号或ID
            patient_id = self.tableWidget.item(self.currentRow, 0).text()
            # 先删除相关的ECG数据
            try:
                self.cursor.execute("DELETE FROM ecg_data WHERE patient_id = %s", (patient_id,))
                # 然后删除病人信息
                self.cursor.execute("DELETE FROM patients WHERE id = %s", (patient_id,))
                self.connection.commit()
                print("Patient and related ECG data successfully removed from database")
            except Exception as e:
                self.connection.rollback()
                print("Error while removing patient info from MySQL:", e)
                return

            # 从表格中删除选中的行
            self.tableWidget.removeRow(self.currentRow)


    def recordChange(self, row, column):
        new_value = self.tableWidget.item(row, column).text()
        patient_id = self.tableWidget.item(row, 0).text()  # 第0列是病人ID
        self.changes[(patient_id, column)] = new_value

    def commitChanges(self):
        columns = ["id", "age", "gender", "medicine"]  # 根据表格和数据库列对应
        for (patient_id, column), new_value in self.changes.items():
            field = columns[column]
            self.updateDatabase(patient_id, field, new_value)
        self.changes.clear()  # 清空记录的更改

    def updateDatabase(self, patient_id, field, new_value):
        query = f"UPDATE patients SET {field} = %s WHERE id = %s"
        try:
            self.cursor.execute(query, (new_value, patient_id))
            self.connection.commit()
            print(f"Updated record {patient_id} in database.")
        except Exception as e:
            self.connection.rollback()
            print(f"Failed to update database: {e}")

    def insertECGData(self, patient_id, heartbeat_count, data_path, diagnosis):
        query = "INSERT INTO ecg_data (patient_id, heartbeat_count, data_path, diagnosis) VALUES (%s, %s, %s, %s)"
        try:
            self.cursor.execute(query, (patient_id, heartbeat_count, data_path, diagnosis))
            self.connection.commit()
            print("ECG data successfully added to database")
        except Exception as e:
            self.connection.rollback()
            print(f"Failed to insert ECG data into database: {e}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
