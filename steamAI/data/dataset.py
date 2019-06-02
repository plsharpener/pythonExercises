import os
import sys
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import pysnooper

class traindataset(Dataset):
    def __init__(self,txt_file:str):
        datalist = self.readfile(txt_file)
        self.datas = self.traindataset(datalist)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        return self.datas[idx]

    # @pysnooper.snoop(output="./log.txt")  #保存中间结果，用于调试
    def str2float(self,strlist:list) -> list:
        """字符串转化为float"""
        for i in range(len(strlist)):
            strlist[i] = float(strlist[i])
        return strlist

    def readfile(self,path:str) -> list:
        """读取文件数据，返回列表"""
        datalist = []
        with open(path,"r") as F:
        # print(f.readlines())
            for i,line in enumerate(F.readlines()):
                if not(i):
                    continue
                datalist.append(self.str2float(list(line.split("\n")[0].split("\t"))))
        return datalist

    def traindataset(self,datalist:list):
        """训练数据集设置"""
        traindatas = []
        for data in datalist:
            traindatas.append((Variable(torch.tensor(data[:-1],dtype =torch.float32)),Variable(torch.tensor(data[-1:],dtype=torch.float32))))  #前38位是输入,最后一位是输出
        return traindatas

    def testdataset(self,datalist:list):
        """测试数据集设置"""
        testdatas = datalist
        return testdatas



if __name__ == "__main__":
    """测试"""
    traindatas = traindataset("./zhengqi_train.txt")
    # print(traindatas[0])
    data,terget = traindatas[0]
    print(type(data),terget)