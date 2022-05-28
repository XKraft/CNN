import os
from PIL import Image
import numpy as np
def generate_dataset(dataset):
    current_path = os.path.dirname(__file__)
    if dataset == "train_and_val":
        #生成训练集数据
        with open(current_path + "/data/train_data.txt", 'r') as f:
            contents = f.readlines()
        x_train, y_train = [], []
        for content in contents:
            value = content.split()
            image_path = current_path + "/data/raw_data/" + value[0]
            img = Image.open(image_path)
            img = np.array(img.convert('L')) #将图片转换成8位灰度图的np.array格式
            img = img / 256 #数据归一化
            x_train.append(img)
            y_train.append(value[1])
            print("generate:" + value[0] + "\n")
        x_train = np.array(x_train)
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        y_train = np.array(y_train)
        np.save(current_path + "/data/x_train_data.npy", x_train_save)#保存数据
        np.save(current_path + "/data/y_train_data.npy", y_train)
        print("===================save train_data===================")

        #生成验证集数据
        with open(current_path + "/data/val_data.txt", 'r') as f:
            contents = f.readlines()
        x_val, y_val = [], []
        for content in contents:
            value = content.split()
            image_path = current_path + "/data/raw_data/" + value[0]
            img = Image.open(image_path)
            img = np.array(img.convert('L')) #将图片转换成8位灰度图的np.array格式
            img = img / 256 #数据归一化
            x_val.append(img)
            y_val.append(value[1])
            print("generate:" + value[0] + "\n")
        x_val = np.array(x_val)
        x_val_save = np.reshape(x_val, (len(x_val), -1))
        y_val = np.array(y_val)
        np.save(current_path + "/data/x_val_data.npy", x_val_save)#保存数据
        np.save(current_path + "/data/y_val_data.npy", y_val)
        print("===================save val_data===================")

    elif dataset == "test":
        #生成测试集数据
        with open(current_path + "/data/test_data.txt", 'r') as f:
            contents = f.readlines()
        x_test, y_test = [], []
        for content in contents:
            value = content.split()
            image_path = current_path + "/data/raw_data/" + value[0]
            img = Image.open(image_path)
            img = np.array(img.convert('L')) #将图片转换成8位灰度图的np.array格式
            img = img / 256 #数据归一化
            x_test.append(img)
            y_test.append(value[1])
            print("generate:" + value[0] + "\n")
        x_test = np.array(x_test)
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        y_test = np.array(y_test)
        np.save(current_path + "/data/x_test_data.npy", x_test_save)#保存数据
        np.save(current_path + "/data/y_test_data.npy", y_test)
        print("===================save test_data===================")

def creat_index():
    current_path = os.path.dirname(__file__)
    f = open(current_path + "/data/train_data.txt", 'w')
    for i in range(1, 71, 1):
        for j in range(1, 11, 1):
            for k in range(2, 12, 1):
                img_name = "Locate{" + str(i) + "," + str(j) + "," + str(k) + "}.jpg"
                f.write(img_name)
                f.write(" ")
                f.write(str(k - 2))
                f.write("\n")
    f.close()

    f = open(current_path + "/data/val_data.txt", 'w')
    for i in range(71, 91, 1):
        for j in range(1, 11, 1):
            for k in range(2, 12, 1):
                img_name = "Locate{" + str(i) + "," + str(j) + "," + str(k) + "}.jpg"
                f.write(img_name)
                f.write(" ")
                f.write(str(k - 2))
                f.write("\n")
    f.close()

    f = open(current_path + "/data/test_data.txt", 'w')
    for i in range(91, 101, 1):
        for j in range(1, 11, 1):
            for k in range(2, 12, 1):
                img_name = "Locate{" + str(i) + "," + str(j) + "," + str(k) + "}.jpg"
                f.write(img_name)
                f.write(" ")
                f.write(str(k - 2))
                f.write("\n")
    f.close()

def get_train_and_val_data():
    current_path = os.path.dirname(__file__)
    if os.path.exists(current_path + "/data/x_train_data.npy") and os.path.exists(current_path + "/data/y_train_data.npy") and os.path.exists(current_path + "/data/x_val_data.npy") and os.path.exists(current_path + "/data/y_val_data.npy"):
        print("===========================load train_and_val_data===========================")
        x_train_save = np.load(current_path + "/data/x_train_data.npy")
        y_train = np.load(current_path + "/data/y_train_data.npy")
        x_train = np.reshape(x_train_save, (len(x_train_save), 64, 64, 1))
        x_val_save = np.load(current_path + "/data/x_val_data.npy")
        y_val = np.load(current_path + "/data/y_val_data.npy")
        x_val = np.reshape(x_val_save, (len(x_val_save), 64, 64, 1))
    else:
        print("===========================generate train_and_val_data===========================")
        generate_dataset("train_and_val")
        x_train_save = np.load(current_path + "/data/x_train_data.npy")
        y_train = np.load(current_path + "/data/y_train_data.npy")
        x_train = np.reshape(x_train_save, (len(x_train_save), 64, 64, 1))
        x_val_save = np.load(current_path + "/data/x_val_data.npy")
        y_val = np.load(current_path + "/data/y_val_data.npy")
        x_val = np.reshape(x_val_save, (len(x_val_save), 64, 64, 1))
    return x_train, y_train, x_val, y_val

def get_test_data():
    current_path = os.path.dirname(__file__)
    if os.path.exists(current_path + "/data/x_test_data.npy") and os.path.exists(current_path + "/data/y_test_data.npy"):
        print("===========================load train_and_val_data===========================")
        x_test_save = np.load(current_path + "/data/x_test_data.npy")
        y_test = np.load(current_path + "/data/y_test_data.npy")
        x_test = np.reshape(x_test_save, (len(x_test_save), 64, 64, 1))
    else:
        print("===========================generate train_and_val_data===========================")
        generate_dataset("test")
        x_test_save = np.load(current_path + "/data/x_test_data.npy")
        y_test = np.load(current_path + "/data/y_test_data.npy")
        x_test = np.reshape(x_test_save, (len(x_test_save), 64, 64, 1))
    return x_test, y_test

