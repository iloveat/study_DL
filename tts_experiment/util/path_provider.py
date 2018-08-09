# -*- coding: utf-8 -*-
import os


def read_file_list(file_name):
    file_name_list = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_name_list.append(line)
    fid.close()
    return file_name_list


def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_path_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_path_list.append(file_name)
    return file_path_list









































