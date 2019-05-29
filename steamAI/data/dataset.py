import os
import sys
import torch

def str2float(strlist:list) -> list:
    for i in range(len(strlist)):
        strlist[i] = float(strlist[i])
    return strlist

def readfile(path:str) -> list:
    """读取文件数据，返回列表"""
    datalist = []
    with open(path,"r") as F:
        # print(f.readlines())
        for i,line in enumerate(F.readlines()):
            if not(i):
                continue
            datalist.append(str2float(list(line.split("\n")[0].split("\t"))))
    return datalist

def traindataset(datalist:list):
    """训练数据集设置"""
    traindatas = []
    traintargets = []
    for data in datalist:
        traindatas.append(data[:-1])  #前38位是输入
        traintargets.append(data[-1])  #最后一位是输出
    return traindatas,traintargets

def testdataset(datalist:list):
    """测试数据集设置"""
    testdatas = datalist
    return testdatas



if __name__ == "__main__":
    """测试"""
    datalist = readfile("./zhengqi_train.txt")
    traindatas,targets = traindataset(datalist)
    datalist = readfile("./zhengqi_test.txt")
    testdatas = testdataset(datalist)

    for data in traindatas:
        print(data)
    print(len(traindatas[1]))
    # print(targets)
    print(len(traindatas))
    print(len(targets))
    print(len(testdatas))
