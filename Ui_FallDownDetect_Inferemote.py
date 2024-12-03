import os
import cv2
import subprocess
import webbrowser
import shutil  # 用于复制文件
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
from net.fallDownDetectYolo import FallDownDetectYolo

VerSion = "1.9.0"


# 弹出窗口
class RadioDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, current_input=None):
        super(RadioDialog, self).__init__(parent)
        self.current_input = current_input
        self.setWindowTitle("输入格式")
        layout = QtWidgets.QVBoxLayout()
        group_box = QtWidgets.QGroupBox("请选择输入类型")

        self.radio1 = QtWidgets.QRadioButton("图片")
        self.radio2 = QtWidgets.QRadioButton("视频")
        self.radio3 = QtWidgets.QRadioButton("摄像头")
        # 设定初始状态
        if self.current_input == "picture":
            self.radio1.setChecked(True)
        elif self.current_input == "video":
            self.radio2.setChecked(True)
        elif self.current_input == "camera":
            self.radio3.setChecked(True)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        vbox.addWidget(self.radio3)
        group_box.setLayout(vbox)

        hbox = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        hbox.addWidget(group_box)
        hbox.addWidget(self.ok_button)

        layout.addLayout(hbox)
        self.setLayout(layout)

    def get_selected_option(self):
        if self.radio1.isChecked():
            return "picture"
        elif self.radio2.isChecked():
            return "video"
        elif self.radio3.isChecked():
            return "camera"
        else:
            return None


# Worker类用于视频处理
class VideoWorker(QObject):
    image_processed = pyqtSignal(str)  # 信号，传递结果图片路径
    finished = pyqtSignal()  # 信号，处理完成
    log_signal = pyqtSignal(str)  # 信号，发送日志信息

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._is_running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_count = 0
            while self._is_running:
                for i in range(15):
                    ret, frame = cap.read()
                if not ret:
                    self.log_signal.emit("视频处理完成")
                    break  # 视频结束

                frame_count += 15
                self.log_signal.emit(f"处理第 {frame_count} 帧...")

                # 缓存当前帧
                cache_path = "cache.jpg"### cache_path为当前目录下
                cv2.imwrite(cache_path, frame)

                # 运行外部脚本
                command = [
                    "python",
                    "inferemote\\test.py",
                    "-r",
                    "localhost",
                    "-p",
                    "9023",
                    "-i",
                    cache_path,
                ]
                self.log_signal.emit(f"运行命令: {' '.join(command)}")
                subprocess.run(command)

                # 读取输出图片路径
                output_image_path = (
                    "__OUTPUTS__\\result-cache.jpg"
                )

                # 发送信号给主线程更新界面
                self.image_processed.emit(output_image_path)

                # 暂停0.5秒
                time.sleep(0.5)

            cap.release()
        except Exception as e:
            self.log_signal.emit(f"处理视频时发生错误: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1102, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 120, 751, 501))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.mainpicLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.mainpicLayout.setContentsMargins(0, 0, 0, 0)
        self.mainpicLayout.setObjectName("mainpicLayout")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(750, 120, 351, 211))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.secondpicLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.secondpicLayout.setContentsMargins(0, 0, 0, 0)
        self.secondpicLayout.setObjectName("secondpicLayout")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(750, 330, 351, 291))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.textLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.textLayout.setContentsMargins(0, 0, 0, 0)
        self.textLayout.setObjectName("textLayout")
        self.textEdit = QtWidgets.QTextEdit(self.verticalLayoutWidget_2)
        self.textEdit.setObjectName("textEdit")
        self.textLayout.addWidget(self.textEdit)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(0, 0, 1101, 110))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit_2 = QtWidgets.QTextEdit(self.verticalLayoutWidget_3)
        self.textEdit_2.setObjectName("textEdit_2")
        self.verticalLayout.addWidget(self.textEdit_2)
        self.statusLayout = QtWidgets.QHBoxLayout()
        self.statusLayout.setObjectName("statusLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit.setObjectName("lineEdit")
        self.statusLayout.addWidget(self.lineEdit)
        self.choose_button = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.choose_button.setObjectName("choose_button")
        self.statusLayout.addWidget(self.choose_button)
        self.run_button = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.run_button.setObjectName("run_button")
        self.statusLayout.addWidget(self.run_button)
        self.verticalLayout.addLayout(self.statusLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.mainscene = QtWidgets.QGraphicsScene()
        self.mainscene.setObjectName("mainscene")
        self.maingraphicsView = QtWidgets.QGraphicsView(self.mainscene)
        self.maingraphicsView.setObjectName("maingraphicsView")
        self.mainpicLayout.addWidget(self.maingraphicsView)
        self.secondscene = QtWidgets.QGraphicsScene()
        self.secondscene.setObjectName("secondscene")
        self.secondgraphicsView = QtWidgets.QGraphicsView(self.secondscene)
        self.secondgraphicsView.setObjectName("secondgraphicsView")
        self.secondpicLayout.addWidget(self.secondgraphicsView)

        # 按钮控制
        self.choose_button.clicked.connect(self.open_file)
        self.run_button.clicked.connect(self.run_inference)

        # 变量
        self.current_input = "picture"
        self.source_path = ""
        self.is_q_pressed = False
        self.Log_text = ""
        self.worker = None
        self.thread = None

        # 工具栏
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        action1 = self.toolBar.addAction(QtGui.QIcon("icons/1.ico"), "输入格式")
        action1.triggered.connect(self.show_dialog)
        action2 = self.toolBar.addAction(QtGui.QIcon("icons/2.ico"), "保存的图片")
        action2.triggered.connect(self.open_saved)
        action3 = self.toolBar.addAction(QtGui.QIcon("icons/3.ico"), "帮助")
        action3.triggered.connect(self.open_help)

        # 标题
        self.textEdit_2.setReadOnly(True)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(40)
        font.setWeight(QtGui.QFont.Bold)
        self.textEdit_2.setFont(font)
        self.textEdit_2.setText("FallDownDetect_Inferemote     v" + VerSion)
        self.Log(
            "Thanks For Using FallDownDetect v"
            + VerSion
            + "     Choose Input Source to Begin"
        )

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def open_saved(self):
        current_working_directory = os.getcwd()
        folder_path = os.path.join(current_working_directory, "saved")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        os.startfile(folder_path)

    def open_file(self):
        self.download_path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            "",
            "Image Files/Video Files (*.jpg *.png *.mp4 *.mov)",
        )
        if not self.download_path[0].strip():
            return
        else:
            self.lineEdit.setText(self.download_path[0])
            self.source_path = self.download_path[0]
            # 判断图片或者视频,更正状态
            _, file_extension = os.path.splitext(self.download_path[0])
            if (
                file_extension.lower() in [".jpg", ".png"]
            ) and self.current_input != "picture":
                QtWidgets.QMessageBox.information(
                    self,
                    "Info",
                    "当前选择的文件为图片，输入格式已改为对应图片类型",
                    QtWidgets.QMessageBox.Yes,
                    QtWidgets.QMessageBox.Yes,
                )
                self.current_input = "picture"
            elif (
                file_extension.lower() in [".mp4", ".mov"]
                and self.current_input != "video"
            ):
                QtWidgets.QMessageBox.information(
                    self,
                    "Info",
                    "当前选择的文件为视频，输入格式已改为对应视频类型",
                    QtWidgets.QMessageBox.Yes,
                    QtWidgets.QMessageBox.Yes,
                )
                self.current_input = "video"
            # 打印预览图
            if self.current_input == "picture":
                pic = cv2.imread(self.download_path[0])
                if pic is not None:
                    self.show_image_second(pic)
                else:
                    QtWidgets.QMessageBox.warning(self, "Warning", "无法读取图片文件。")
            elif self.current_input == "video":
                cap = cv2.VideoCapture(self.download_path[0])
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.show_image_second(frame)
                    cap.release()
                else:
                    QtWidgets.QMessageBox.warning(self, "Warning", "无法打开视频文件。")
            elif self.current_input == "camera":
                pass  # 可选择添加提示图片或信息
            self.Log("Ready For Run")

    def run_inference(self):
        if self.current_input == "picture":
            self.process_image()
        elif self.current_input == "video":
            self.process_video()
        elif self.current_input == "camera":
            self.process_camera()

    def process_image(self):
        self.Log("开始测试")
        # 缓存图片为指定路径
        cache_image_path = "cache.jpg"
        try:
            shutil.copy(self.source_path, cache_image_path)
            self.Log(f"图片已缓存至 {cache_image_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"缓存图片失败:\n{e}")
            self.Log("缓存图片失败。")
            return

        # 运行外部脚本
        command = [
            "python",
            "inferemote\\test.py",
            "-r",
            "localhost",
            "-p",
            "9023",
            "-i",
            cache_image_path,
        ]
        self.Log(f"运行命令: {' '.join(command)}")
        subprocess.run(command)

        self.Log("脚本运行完成。")

        # 读取输出图片并显示
        output_image_path = "__OUTPUTS__\\result-cache.jpg"
        if os.path.exists(output_image_path):
            self.show_image_main_(output_image_path)
            self.Log(f"显示结果图片: {output_image_path}")
        else:
            QtWidgets.QMessageBox.warning(
                self, "Warning", f"结果图片未找到: {output_image_path}"
            )
            self.Log("结果图片未找到。")
        self.Log("Test Finish")

    def process_video(self):
        if self.source_path == "":
            QtWidgets.QMessageBox.information(
                self,
                "Warning",
                "未选择输入文件",
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Yes,
            )
            return

        self.Log("开始测试视频")
        self.Log("按 Q 键退出")

        # 创建并启动视频处理线程
        self.thread = QtCore.QThread()
        self.worker = VideoWorker(self.source_path)
        self.worker.moveToThread(self.thread)

        # 连接信号与槽
        self.thread.started.connect(self.worker.run)
        self.worker.image_processed.connect(self.display_output_image)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.log_signal.connect(self.Log)  # Connect log_signal to Log

        self.thread.start()

    def display_output_image(self, image_path):
        self.show_image_main_(image_path)
        self.Log(f"显示结果图片: {image_path}")

    def process_camera(self):
        # 直接在终端运行外部命令
        script_path = os.path.join(
            "inferemote", "test.py"
        )  # 确保test2.py的路径正确
        command = [
            "python",
            script_path,
            "-r",
            "localhost",
            "-p",
            "9023",
            "-i",
            "camera",
        ]

        try:
            # 在Windows中使用start命令打开新的命令提示符窗口并执行命令
            subprocess.Popen(["start", "cmd", "/k"] + command, shell=True)
            self.Log("已启动外部摄像头脚本。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"启动摄像头脚本失败:\n{e}")

    def show_dialog(self):
        dialog = RadioDialog(self, self.current_input)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # 选择框结束后的后处理
            selected_option = dialog.get_selected_option()
            if selected_option == "picture":
                self.lineEdit.setText("选择一张图片")
            elif selected_option == "video":
                self.lineEdit.setText("选择一个视频")
            elif selected_option == "camera":
                self.lineEdit.setText("使用默认摄像头")
            self.current_input = selected_option

    def Log(self, out_text):
        datetime = QtCore.QDateTime.currentDateTime()
        date_text = datetime.toString("yyyy-MM-dd HH:mm:ss")
        new_text = f"[{date_text}] {out_text}\n"
        self.Log_text += new_text
        self.textEdit.setText(self.Log_text)
        self.textEdit.moveCursor(QtGui.QTextCursor.End)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Q:
            self.is_q_pressed = True
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.stop()

    # 不同输入打印图片的方法
    def show_image_main(self, cv2likeimage):
        cv2.imwrite("cache_im.jpg", cv2likeimage)
        pixmap = QtGui.QPixmap("cache_im.jpg")
        scaredPixmap = pixmap.scaled(
            QtCore.QSize(720, 960), aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.mainscene.clear()  # 清除之前的图片
        self.mainscene.addItem(scaredPixmap_item)
        os.remove("cache_im.jpg")

    def show_image_second(self, cv2likeimage):
        cv2.imwrite("cache_im.jpg", cv2likeimage)
        pixmap = QtGui.QPixmap("cache_im.jpg")
        scaredPixmap = pixmap.scaled(
            QtCore.QSize(300, 400), aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.secondscene.clear()  # 清除之前的图片
        self.secondscene.addItem(scaredPixmap_item)
        os.remove("cache_im.jpg")

    def show_image_main_(self, image_path):
        if not os.path.exists(image_path):
            QtWidgets.QMessageBox.warning(self, "警告", f"文件 {image_path} 不存在。")
            return
        pixmap = QtGui.QPixmap(image_path)
        scaredPixmap = pixmap.scaled(
            QtCore.QSize(720, 960), aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.mainscene.clear()
        self.mainscene.addItem(scaredPixmap_item)

    def show_image_second_(self, image_path):
        if not os.path.exists(image_path):
            QtWidgets.QMessageBox.warning(self, "警告", f"文件 {image_path} 不存在。")
            return
        pixmap = QtGui.QPixmap(image_path)
        scaredPixmap = pixmap.scaled(
            QtCore.QSize(300, 400), aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.secondscene.clear()
        self.secondscene.addItem(scaredPixmap_item)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FallDownDetect"))
        self.choose_button.setText(_translate("MainWindow", "选择"))
        self.run_button.setText(_translate("MainWindow", "运行"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))

    # detector部分
    def init_detector(self, image_path_):
        self.detector = FallDownDetectYolo(
            yolov5_path=r"net\yolov5",
            weight_path=r"net\best.pt",
            image_path=image_path_,
            video_path=None,  # 如果不使用视频
            pt_or_onnx="pt",
        )

    def run_inferemote(self, cv2likeimage):
        cv2.imwrite("cache_im.jpg", cv2likeimage)
        command = [
            "python",
            "F:\\GitHub\\fall-detection\\inferemote\\test2.py",
            "-r",
            "localhost",
            "-p",
            "9023",
            "-i",
            "image_path_",
        ]
        process = subprocess.run(command, capture_output=True, text=True)

    def run_inferemote_video(self, cv2likeimage):
        cv2.imwrite("cache_im.jpg", cv2likeimage)
        command = [
            "python",
            "F:\\GitHub\fall-detection\\inferemote\\test2.py",
            "-r",
            "localhost",
            "-p",
            "9023",
            "-i",
            "cache_im.jpg",
            "-w",
            "1",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def open_help(self):
        url = "https://github.com/SowingG2333/fall-detection"
        webbrowser.open(url)



