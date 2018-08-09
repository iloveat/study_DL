# -*- coding: utf-8 -*-
import zipfile


data_dir = '../data/cifar10'

for fin in ['train_tiny.zip', 'test_tiny.zip']:
    with zipfile.ZipFile(data_dir + '/' + fin, 'r') as zin:
        zin.extractall(data_dir + '/')


