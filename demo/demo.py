import sys,os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap, QMovie
from PyQt5 import QtCore,QtGui,QtWidgets
from calligraphy.code import draw_pic
from demo_utils import argument_setting
from demo_env import demo_main, model_env, efficient_demo

windows = []
meetingUrl="https://meet.google.com/ose-krmk-zzg"
noise=0
word_idx=42
# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         QtWidgets.QMainWindow.setFixedSize(self,1600,400)
#         self.webview = WebEngineView()
#         self.webview.load(QtCore.QUrl(meetingUrl))
#         self.setCentralWidget(self.webview)
# class WebEngineView(QtWebEngineWidgets.QWebEngineView):
#     windows = [] #创建一个容器存储每个窗口，不然会崩溃，因为是createwindow函数里面的临时变量
#     def createWindow(self, QWebEnginePage_WebWindowType):
#         newtab =   WebEngineView()
#         newwindow= MainWindow()
#         newwindow.setCentralWidget(newtab)
#         newwindow.show()
#         self.windows.append(newwindow)
#         return newtab

class CalligrahpyUI(QMainWindow):
    def __init__(self, args, ui='demo.ui', resc_root='./'):
        super(CalligrahpyUI, self).__init__()
        
        # resource path setting
        self.resc_root = resc_root
        self.img_path = os.path.join(resc_root, 'imgs')
        self.output_path = os.path.join(resc_root, 'output')

        # img setting
        self.loading_graph = os.path.join(self.img_path, 'loading.gif')
        self.word_dict =  {
            "永": (42, "YONG.jpg"),
            "史": (436, "SHI.jpg"),
            "殺": (312, "SHA.jpg"),
            "并": (773, "BING.jpg"),
            "引": (277, "YIN.jpg"),
        }

        # load ui file to self
        loadUi(ui, self)

        # dynamic size
        self.label.setScaledContents(True)

        # init value
        self.args = args
        self.args.test_char, _ = self.word_dict.get(self.comboBox.currentText())
        
        # model construction
        self.args.model, self.args.criterion, self.args.extractor = model_env(args)

        # externel func
        self.result_vis = draw_pic
        self.demo_func = efficient_demo
        # self.data_env = data_env

        # event binding
        self.comboBox.currentIndexChanged.connect(self.set_word)
        self.pushButton.clicked.connect(self.data_eval)
        # self.pushButton.clicked.connect(self.p_test)

    def set_word(self):
        # current text
        cur_text = self.comboBox.currentText()
        print(f'change to {cur_text}')

        self.args.test_char, img_name = self.word_dict.get(cur_text)
        word_path = os.path.join(self.img_path, img_name)

        # load input and slim gif
        self.progress_bar = QMovie(self.loading_graph)
        self.input.setMovie(self.progress_bar)
        self.slim.setMovie(self.progress_bar)
        self.progress_bar.start()
        # self.progress_bar.jumpToNextFrame()

        # load target image
        self.target.setPixmap(QPixmap(word_path))

    def p_test(self):
        print('clicked')
        self.progress_bar.start()

    def data_eval(self):
        # current noise
        noise = self.doubleSpinBox.value()
        self.args.noise = [-noise, noise]

        # activate progress bar
        # self.progress_bar.start()

        # pass data to model
        self.demo_func(self.args)

        # load result img
        self.result_vis()

        # set path
        vis_path = os.path.join(self.output_path, 'visual', 'test_all_input.png')
        slim_path = os.path.join(self.output_path, 'test_char', 'test_all_compare.png')

        # set result image
        self.input.setPixmap(QPixmap(vis_path))
        self.slim.setPixmap(QPixmap(slim_path))


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
 
def go_web(args):
    
    noise = w.doubleSpinBox.value()

    # call demo function
    demo_main(args, noise, word_idx)
    draw_pic()
    w.slim.setPixmap(QPixmap("output/test_char/test_all_compare.png"))
    w.input.setPixmap(QPixmap("./output/visual/test_all_input.png"))
    # newtab =   WebEngineView()
    # newtab.load(QtCore.QUrl(meetingUrl))
    # newwindow= MainWindow()
    # newwindow.setCentralWidget(newtab)
    # newwindow.show()
    # windows.append(newwindow)
    

if __name__ == "__main__":
    args = argument_setting()

    # construction env first
    if args.non_efficient is False:
        args.model, args.critetion, args.extractor = model_env(args)

    if args.gui:
        # execute refactor version
        app = QApplication(sys.argv)
        w = CalligrahpyUI(args)
        w.show()
        app.exec_()

        # app = QApplication(sys.argv)
        # w = loadUi('demo.ui')
        # windows.append(w)
        # ui=ExComboBox(w)
        # w.label.setScaledContents(True)
        # w.pushButton.clicked.connect(lambda: go_web(args))
        # # w.showFullScreen()
        # # w.setFixedSize(800,800)
        # w.show()
        # app.exec_()
    else:
        demo_main(args)