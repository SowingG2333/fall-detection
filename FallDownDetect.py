import sys,os
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
import Ui_FallDownDetect
import Ui_FallDownDetect_Inferemote
from PyQt5 import QtCore,QtGui
 
def read_properties(file_path):
    properties = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                # 去除前后空白字符，并跳过空行和注释行（假设以 # 开头的行为注释）
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    key, value = stripped_line.split('=')
                    properties[key.strip()] = value.strip()
    return properties

properties_file_path = 'property.txt'
properties = read_properties(properties_file_path)

mode = properties.get('mode', 'offline')

class MyMainWindow(QMainWindow,Ui_FallDownDetect.Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("FallDownDetect")
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon('icons\main.png'))
        # 获取当前工作目录
        current_working_directory = os.getcwd()
        
        # 构建要检查/创建的文件夹路径
        folder_path = os.path.join(current_working_directory, 'saved')
        
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

class MyMainWindow_Inferemote(QMainWindow,Ui_FallDownDetect_Inferemote.Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow_Inferemote,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("FallDownDetect")
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon('icons\main.png'))
        # 获取当前工作目录
        current_working_directory = os.getcwd()
        
        # 构建要检查/创建的文件夹路径
        folder_path = os.path.join(current_working_directory, 'saved')
        
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

 
if __name__ == "__main__":
    if mode == 'offline':
        app = QApplication(sys.argv)
        myWin = MyMainWindow()
        myWin.show()
        sys.exit(app.exec_())   
    elif mode == 'online':
        app = QApplication(sys.argv)
        myWin = MyMainWindow_Inferemote()
        myWin.show()
        sys.exit(app.exec_())  
    else:
        print("[ERROR] Invalid mode, please check property")