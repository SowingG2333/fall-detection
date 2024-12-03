import os,cv2,subprocess,threading
import webbrowser
from PyQt5 import QtCore, QtGui, QtWidgets
from net.fallDownDetectYolo import FallDownDetectYolo

VerSion = '1.8.0'

#弹出窗口
class RadioDialog(QtWidgets.QDialog):
    def __init__(self, parent = None, current_input = None):
        super(RadioDialog, self).__init__(parent)
        self.current_input = current_input
        self.setWindowTitle('输入格式')
        layout = QtWidgets.QVBoxLayout()
        group_box = QtWidgets.QGroupBox('')
        
        self.radio1 = QtWidgets.QRadioButton('图片')
        self.radio2 = QtWidgets.QRadioButton('视频')
        self.radio3 = QtWidgets.QRadioButton('摄像头')
        #设定初始状态
        if self.current_input == 'picture':
            self.radio1.setChecked(True)
        if self.current_input == 'video':
            self.radio2.setChecked(True)
        if self.current_input == 'camera':
            self.radio3.setChecked(True)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        vbox.addWidget(self.radio3)
        group_box.setLayout(vbox)

        hbox = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton('确定')
        self.ok_button.clicked.connect(self.accept)
        hbox.addWidget(group_box)
        hbox.addWidget(self.ok_button)

        layout.addLayout(hbox)
        self.setLayout(layout)

    def get_selected_option(self):
            if self.radio1.isChecked():
                return 'picture'
            elif self.radio2.isChecked():
                return 'video'
            elif self.radio3.isChecked():
                return 'camera'
            else:
                return None

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
        self.textEdit.setReadOnly(True)
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

        #按钮控制
        self.choose_button.clicked.connect(self.open_file)
        self.run_button.clicked.connect(self.run_inference)

        #变量
        self.current_input = 'picture'
        self.source_path = ''
        self.is_q_pressed = False
        self.Log_text = ''

        #工具栏
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        action1 = self.toolBar.addAction(QtGui.QIcon('icons/1.ico'), "输入格式")
        action1.triggered.connect(self.show_dialog)
        action2 = self.toolBar.addAction(QtGui.QIcon('icons/2.ico'),"保存的图片")
        action2.triggered.connect(self.open_saved)
        action3 = self.toolBar.addAction(QtGui.QIcon('icons/3.ico'),"帮助")
        action3.triggered.connect(self.open_help)


        #标题
        self.textEdit_2.setReadOnly(True)
        font = QtGui.QFont()
        font.setFamily('Helvetica')
        font.setPointSize(40)
        font.setWeight(QtGui.QFont.Bold)
        self.textEdit_2.setFont(font)
        self.textEdit_2.setText('FallDownDetect     v'+VerSion)
        self.Log('Thanks For Using FallDownDetect v'+VerSion+'     Choose Input Source to Begin')

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def open_saved(self):
        current_working_directory = os.getcwd()
        folder_path = os.path.join(current_working_directory, 'saved')
        os.startfile(folder_path)

    def open_file(self):
        self.download_path = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", '', "Image Files/Video Files(*.jpg *.png *.mp4 *.mov)")
        if not self.download_path[0].strip():
            pass
        else:
            self.lineEdit.setText(self.download_path[0])
            self.source_path = self.download_path[0]
            #判断图片或者视频,更正状态
            _, file_extension = os.path.splitext(self.download_path[0])
            if (file_extension == '.jpg' or file_extension == '.png') and self.current_input != 'picture':
                QtWidgets.QMessageBox.information(self,"info","当前选择的文件为图片，输入格式已改为对应图片类型",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                self.current_input = 'picture'
            if (file_extension == '.mp4' or file_extension == '.mov')and self.current_input != 'video':
                QtWidgets.QMessageBox.information(self,"info","当前选择的文件为视频，输入格式已改为对应视频类型",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                self.current_input = 'video'
            #打印预览图
            if self.current_input == 'picture':
                pic = cv2.imread(self.download_path[0])
                self.show_image_second(pic)
            if self.current_input == 'video':
                cap = cv2.VideoCapture(self.download_path[0])
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret :
                        self.show_image_second(frame)
            if self.current_input == 'camera':
                pass  #准备一张提示图片？
            self.Log('Ready For Run')

    def run_inference(self):
        self.is_q_pressed = False
        current_working_directory = os.getcwd()
        save_directory = os.path.join(current_working_directory, 'saved/')
 
        
        if self.current_input == 'picture':
            if self.source_path == '':
                QtWidgets.QMessageBox.information(self,"Warning","未选择输入",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                return 
            self.Log('Start Testing')
            #这里放推理API
            self.init_detector('')
            image = cv2.imread(self.source_path)
            results =  self.detector.img_inference_(image)
            img_draw = self.detector.draw_detections(image.copy(), results)
            self.show_image_main(img_draw)

            down_elements = results[results['class_name'] == 'down'] # 计算这些行的数量
            down_count = len(down_elements)
            self.Log('Test Finish, Fall Down Number is '+str(down_count))
 
        if self.current_input == 'video':
            if self.source_path == '':
                QtWidgets.QMessageBox.information(self,"Warning","未选择输入",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                return
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                QtWidgets.QMessageBox.information(self,"Warning","视频文件不可用",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                return
            
            self.init_detector('')
            self.Log('Start Testing')
            self.Log('Press Q to Quit')
            falldown_count = 0
            frame_count = 0
            save_count = 0
            while True :
                ret, frame = cap.read()
                if not ret:
                    break  # 视频结束
                frame_count += 1
                #print(f"处理第 {frame_count} 帧...", end='\r')
                self.Log('处理第'+str(frame_count)+'帧...')
                ##这里放推理API
                
                results =  self.detector.img_inference_(frame)

                # 打印感兴趣的列
                #print("\nPyTorch 推理结果:")
                #print(results[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

                # 绘制检测框
                img_draw = self.detector.draw_detections(frame.copy(), results)
                self.show_image_second(frame)
                self.show_image_main(img_draw)
                
                # 设定一个置信度阈值，例如0.6
                confidence_threshold = 0.6

                # 筛选出class_name为'down'且confidence大于阈值的行
                fall_detections = results[(results['class_name'] == 'down') & (results['confidence'] > confidence_threshold)]

                if not fall_detections.empty:
                    falldown_count += 1
                if falldown_count == 2:
                    self.Log('[WARNING] Fall Down Alert')
                    #打印保存图片时间
                    datetime = QtCore.QDateTime.currentDateTime()
                    text = datetime.toString("yyyy-MM-dd_HH-mm-ss")
                    save_image_path = os.path.join(save_directory, text+'.jpg')
                    cv2.imwrite(save_image_path, img_draw)
                    falldown_count = 0
                    save_count += 1

                # 按 'q' 键退出
                cv2.waitKey(10)
                if self.is_q_pressed:
                    break
            self.Log('Test Finish, '+str(save_count)+' pictures was saved')
        if self.current_input == 'camera':
            #默认系统摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                QtWidgets.QMessageBox.information(self,"Warning","摄像头不可用",QtWidgets.QMessageBox.Yes,QtWidgets.QMessageBox.Yes)
                return
            
            self.init_detector('')
            self.Log('Start Testing')
            self.Log('Press Q to Quit')
            falldown_count = 0
            frame_count = 0
            while True :
                ret, frame = cap.read()
                if not ret:
                    break  # 视频结束
                frame_count += 1
                print(f"处理第 {frame_count} 帧...", end='\r')
                ##这里放推理API
                
                results =  self.detector.img_inference_(frame)

                # 打印感兴趣的列
                #print("\nPyTorch 推理结果:")
                #print(results[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

                # 绘制检测框
                img_draw = self.detector.draw_detections(frame.copy(), results)
                self.show_image_second(frame)
                self.show_image_main(img_draw)
                
                # 设定一个置信度阈值，例如0.6
                confidence_threshold = 0.6

                # 筛选出class_name为'down'且confidence大于阈值的行
                fall_detections = results[(results['class_name'] == 'down') & (results['confidence'] > confidence_threshold)]

                if not fall_detections.empty:
                    falldown_count += 1
                if falldown_count == 5:
                    self.Log('[WARNING] Fall Down Alert')
                    #打印保存图片时间
                    datetime = QtCore.QDateTime.currentDateTime()
                    text = datetime.toString("yyyy-MM-dd_HH-mm-ss")
                    save_image_path = os.path.join(save_directory, text+'.jpg')
                    cv2.imwrite(save_image_path, img_draw)
                    falldown_count = 0
                # 按 'q' 键退出
                cv2.waitKey(30)
                if self.is_q_pressed:
                    break


    def show_dialog(self):
        dialog = RadioDialog(self, self.current_input)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            #选择框结束后的后处理
            selected_option = dialog.get_selected_option()
            if selected_option == 'picture':
                self.lineEdit.setText("choose a picture")
            if selected_option == "video":
                self.lineEdit.setText("choose a video")
            if selected_option == 'camera':
                self.lineEdit.setText("Using Default Camera")
                self.Log('Ready For Run')
            self.current_input = selected_option
    
    def Log(self, out_text):
        datetime = QtCore.QDateTime.currentDateTime()
        date_text = datetime.toString("yyyy-MM-dd HH:mm:ss")
        new_text = '[' + date_text + ']' + out_text + '\n'
        self.Log_text += new_text
        self.textEdit.setText(self.Log_text)
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Q:
            self.is_q_pressed = True
    
    #不同输入打印图片的方法
    def show_image_main(self,cv2likeimage):
        cv2.imwrite("cache_im.jpg",cv2likeimage)
        pixmap = QtGui.QPixmap("cache_im.jpg")
        scaredPixmap = pixmap.scaled(QtCore.QSize(720,960),aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.mainscene.clear()
        self.mainscene.addItem(scaredPixmap_item)
        os.remove("cache_im.jpg")

    def show_image_second(self,cv2likeimage):
        cv2.imwrite("cache_im.jpg",cv2likeimage)
        pixmap = QtGui.QPixmap("cache_im.jpg")
        scaredPixmap = pixmap.scaled(QtCore.QSize(300,400),aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.secondscene.clear()
        self.secondscene.addItem(scaredPixmap_item)
        os.remove("cache_im.jpg")
    
    def show_image_main_(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        scaredPixmap = pixmap.scaled(QtCore.QSize(720,960),aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.mainscene.clear()
        self.mainscene.addItem(scaredPixmap_item)

    def show_image_second_(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        scaredPixmap = pixmap.scaled(QtCore.QSize(300,400),aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        scaredPixmap_item = QtWidgets.QGraphicsPixmapItem(scaredPixmap)
        self.secondscene.clear()
        self.secondscene.addItem(scaredPixmap_item)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.choose_button.setText(_translate("MainWindow", "选择"))
        self.run_button.setText(_translate("MainWindow","运行"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))


    #detector部分
    def init_detector(self, image_path_):
        self.detector = FallDownDetectYolo(
            yolov5_path='net\\yolov5',
            weight_path='net\\best.pt',
            image_path=image_path_,
            video_path=None,  # 如果不使用视频
            pt_or_onnx='pt'
        )

    def open_help(self):
        url = 'https://github.com/SowingG2333/fall-detection'
        webbrowser.open(url)
