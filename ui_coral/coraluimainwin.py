#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/9/13 16:15
@description: A graphical interface that integrates functions such as
coral data visualization and coral classification.
"""

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import yaml

from ui_coral.coralui import Ui_CoralGUI
from ui_coral.coral_recog_api import recog
from datetime import datetime

class CoralGUIMainWindow(QMainWindow, Ui_CoralGUI):
    def __init__(self):
        super(CoralGUIMainWindow, self).__init__()
        self.imagepath = None
        self.img = None
        self.model = None
        self.modellist = None
        self.corallist = None
        self.setupUi(self)
        self.initialization()
        self.ViewDataSet.clicked.connect(self.viewdataset)
        self.LoadImage.clicked.connect(self.loadimage)
        self.ImageDisplay.setAlignment(Qt.AlignCenter)
        self.ModelSelection.currentIndexChanged.connect(self.modelselection)
        self.Quit.clicked.connect(self.quit)
        # Image Display
        self.CoralImage = QVBoxLayout(self.CoralDisplay)
        self.CoralImage.addWidget(self.ImageDisplay)

    # Initialization of the graphical interface
    def initialization(self):
        f = open('models/models.yaml', 'r', encoding='utf-8')
        modelfile = f.read()
        self.modellist = yaml.full_load(modelfile)
        f = open('data/coral.yaml', 'r', encoding='utf-8')
        modelfile = f.read()
        self.corallist = yaml.full_load(modelfile)
        # setting
        self.ModelArcText.setReadOnly(True)
        self.BaseLineText.setReadOnly(True)
        self.TrainSetText.setReadOnly(True)
        self.TestSetText.setReadOnly(True)
        self.CoralCategoryText.setReadOnly(True)
        self.AccuracyText.setReadOnly(True)
        self.InferenceTimeText.setReadOnly(True)
        self.IntroductionText.setReadOnly(True)
        self.modelselect_init()
        return

    # Initialization of the model selection
    def modelselect_init(self):
        _translate = QCoreApplication.translate
        for i, key in enumerate(self.modellist.keys()):
            self.ModelSelection.addItem("")
            self.ModelSelection.setItemText(i + 1, _translate("CoralGUI", key))
        return

    # view dataset
    def viewdataset(self):
        QFileDialog.getOpenFileName(self, 'OpenFile', '.')
        return

    # load image
    def loadimage(self):
        self.imagepath, _ = QFileDialog.getOpenFileName(self, 'OpenFile', '.')
        if self.imagepath:
            self.img = cv2.imread(self.imagepath)
            self.displayimage()
            self.PromptInformationText.setPlaceholderText("Image File:" + self.imagepath)
            if self.model is not None:
                self.coral_identification()
        else:
            return

    # quit
    def quit(self):
        QApplication.instance().quit()
        return

    # display image
    def displayimage(self):
        if self.img is not None:
            frame = cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), (571, 351))
            Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
            self.ImageDisplay.setPixmap(QPixmap.fromImage(Qframe))
        return

    # model selection
    def modelselection(self):
        modelname = self.ModelSelection.currentText()
        self.loadmodel(modelname)
        if self.img is not None:
            self.coral_identification()
        return

    # coral identifition using api
    def coral_identification(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        t1 = datetime.now()
        coral_class, perc = self.model.coral_recog(img)
        t2 = datetime.now()
        t = '%.4f'%((t2 - t1).total_seconds()) + 's'
        perc = '%.2f'%(perc*100) + '%'
        self.CoralCategoryText.setPlaceholderText(coral_class.capitalize())
        self.AccuracyText.setPlaceholderText(perc)
        self.InferenceTimeText.setPlaceholderText(t)
        self.IntroductionText.setText(self.corallist[coral_class.capitalize()])
        return

    # load model
    def loadmodel(self,modelname):
        self.ModelArcText.setPlaceholderText(self.modellist[modelname]['model_architecture'])
        self.BaseLineText.setPlaceholderText(self.modellist[modelname]['baseline'])
        self.TrainSetText.setPlaceholderText(self.modellist[modelname]['train_set_acc'])
        self.TestSetText.setPlaceholderText(self.modellist[modelname]['test_set_acc'])
        self.model = recog(model=self.modellist[modelname]['model_path'], device="cpu")
        return