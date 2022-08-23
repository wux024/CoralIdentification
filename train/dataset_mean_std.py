#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/9/10 9:03
@description: Calculates the mean and standard deviation of the dataset
"""
import os
import numpy as np
import cv2

# dataset path
train_data_path = "../data/train"
val_data_path = "../data/val"
train_corals_path = os.listdir(train_data_path)
val_corals_path = os.listdir(val_data_path)
train_corals_path = [train_data_path + '/' + s for s in train_corals_path]
val_corals_path = [val_data_path + '/' + s for s in val_corals_path]
corals_path = train_corals_path + val_corals_path
coral_path_ = []
for s in corals_path:
    coral_path = os.listdir(s)
    coral_path_ += [s + '/' + ss for ss in coral_path]

b_m, g_m, r_m = 0, 0, 0
b_s, g_s, r_s = 0, 0, 0

for i,s in enumerate(coral_path_):
    img = cv2.imread(s)
    img = img.astype(np.float32)/255.
    b_m += img[0, :, :].mean()
    g_m += img[1, :, :].mean()
    r_m += img[2, :, :].mean()

    b_s += img[0, :, :].std()
    g_s += img[1, :, :].std()
    r_s += img[2, :, :].std()

b_m /= len(coral_path_)
g_m /= len(coral_path_)
r_m /= len(coral_path_)

b_s /= len(coral_path_)
g_s /= len(coral_path_)
r_s /= len(coral_path_)

print("---------mean----------")
print("Bule  channel mean:", b_m)
print("Green channel mean:", g_m)
print("Red   channel mean:", r_m)
print("---------standard deviation----------")
print("Bule  channel standard deviation:", b_s)
print("Green channel standard deviation:", g_s)
print("Red   channel standard deviation:", r_s)