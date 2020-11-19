import sys,os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore,QtGui,QtWidgets,QtWebEngineWidgets
from demo_gui import main
from calligraphy.code import draw_pic
from argparse import ArgumentParser

windows = []
meetingUrl="https://meet.google.com/ose-krmk-zzg"
noise=0
word_idx=42
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QtWidgets.QMainWindow.setFixedSize(self,1600,800)
        self.webview = WebEngineView()
        self.webview.load(QtCore.QUrl(meetingUrl))
        self.setCentralWidget(self.webview)
class WebEngineView(QtWebEngineWidgets.QWebEngineView):
    windows = [] #创建一个容器存储每个窗口，不然会崩溃，因为是createwindow函数里面的临时变量
    def createWindow(self, QWebEnginePage_WebWindowType):
        newtab =   WebEngineView()
        newwindow= MainWindow()
        newwindow.setCentralWidget(newtab)
        newwindow.show()
        self.windows.append(newwindow)
        return newtab
        
class ExComboBox(object):
    def __init__(self,w):
        print(w.comboBox.currentText())
        w.comboBox.activated[str].connect(self.words)
    def words(self):
        global word_idx
        w.input.setPixmap(QPixmap("./imgs/loading.gif"))
        w.slim.setPixmap(QPixmap("./imgs/loading.gif"))
        print(w.comboBox.currentText())
        if w.comboBox.currentText() == "永":
            w.target.setPixmap(QPixmap("imgs/YONG.jpg"))
            word_idx = 42
        elif w.comboBox.currentText() == "史":
            w.target.setPixmap(QPixmap("imgs/SHI.jpg"))
            # w.slim.setPixmap(QPixmap("output/test_char/test_all_compare.png"))
            word_idx = 436
        elif w.comboBox.currentText() == "殺":
            w.target.setPixmap(QPixmap("imgs/SHA.jpg"))
            word_idx = 312
        elif w.comboBox.currentText() == "并":
            w.target.setPixmap(QPixmap("imgs/BING.jpg"))
            word_idx = 773
        elif w.comboBox.currentText() == "引":
            w.target.setPixmap(QPixmap("imgs/YIN.jpg"))
            word_idx = 277

def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--efficient', default=False, action='store_true',
                        help='improve demo execution time (default: False)')

    return parser.parse_args()


def go_web(args):
    
    noise = w.doubleSpinBox.value()
    main(noise,word_idx,args.efficient)
    draw_pic()
    w.slim.setPixmap(QPixmap("output/test_char/test_all_compare.png"))
    w.input.setPixmap(QPixmap("./output/visual/test_all_input.png"))
    newtab =   WebEngineView()
    newtab.load(QtCore.QUrl(meetingUrl))
    newwindow= MainWindow()
    newwindow.setCentralWidget(newtab)
    newwindow.show()
    windows.append(newwindow)
    

if __name__ == "__main__":
    args = argument_setting()
    app = QApplication(sys.argv)
    w = loadUi('demo.ui')
    windows.append(w)
    ui=ExComboBox(w)
    w.pushButton.clicked.connect(lambda: go_web(args))
    w.show()
    app.exec_()