#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/9/13 16:15
"""
from PyQt5.QtWidgets import *
from ui_coral import *
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Windows'))
    mymain = CoralGUIMainWindow()
    mymain.show()
    sys.exit(app.exec())