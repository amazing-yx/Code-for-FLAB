#   -*- coding = utf-8 -*-
#   @time : 2021/6/14 18:47
#   @ File : creditcard_lianbang.py
#   @Software: PyCharm
#   @time : 2021/6/24 10:47 修改
#   目的是利用Ri = （k-1）/k * Rik + 1/k * rik     Ri为信用值，k为次数，Rik为上一次信用值，rik为完成度其中rik只为0或者1
#   使用记得修改epoch！！！

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import regularizers
from Block import Block
import datetime

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics
from tensorflow.keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Data():
    def __init__(self, data):
        self.data = data

    def sample(self, start, end):
        size = len(self.data)
        return self.data[int(size * start):int(size * end)]

    def sample_smote(self, start, end):
        data = self.sample(start, end)
        dataSmote = np.array(data.drop("Class", axis=1))
        y = np.array(data[['Class']])
        smt = SMOTE()
        dataSmote, y = smt.fit_resample(dataSmote, y)  # SMOTE 重采样
        dataSmote = pd.DataFrame(dataSmote)
        y = pd.DataFrame(y)
        dataSmote['Class'] = y
        dataSmote = dataSmote.sample(frac=1)  # 打乱顺序
        dataSmote.columns = list(data.columns)  # 将原始数据 data 的列名设置为 dataSmote 的列名
        return dataSmote


class Aggregator():

    def __init__(self):
        self.wB1 = 0.05
        self.wB2 = 0.15
        self.wB3 = 0.05
        self.wB4 = 0.05
        self.wB5 = 0.08
        self.wB6 = 0.06
        self.wB7 = 0.10
        self.wB8 = 0.07
        self.wB9 = 0.09
        self.wB10 = 0.30

    def aggregate(self, delta, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10):
        delta = np.array(delta, dtype=object)
        temp = (self.wB1 * np.array(B1, dtype=object) + self.wB2 * np.array(B2, dtype=object) + self.wB3 * np.array(B3,
                                                                                                                    dtype=object)) + self.wB4 * np.array(
            B4, dtype=object) + self.wB5 * np.array(B5, dtype=object) + self.wB6 * np.array(B6,
                                                                                            dtype=object) + self.wB7 * np.array(
            B7, dtype=object) + self.wB8 * np.array(B8, dtype=object) + self.wB9 * np.array(B9,
                                                                                            dtype=object) + self.wB10 * np.array(
            B10, dtype=object)
        temp -= delta
        delta += temp

        return delta


class Model():

    def __init__(self):
        self.input_shape = (30,)
        self.model = Sequential()
        self.model.add(
            Dense(32, activation='relu', input_shape=self.input_shape, kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',  # rmsprop
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def saveModel(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def loadModel(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    def run(self, X, Y, validation_split=0.1, load=True):
        if (load):
            self.model = self.loadModel()
        self.model.fit(X, Y, epochs=5, validation_split=validation_split, verbose=1)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)[1] * 100

    def loss(self, X, Y):
        return self.model.evaluate(X, Y)[0]

    def predict(self, X):
        return self.model.predict(X)

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weight):
        self.model.set_weights(weight)


class Bank(Model):

    def __init__(self, data, split_size=0):
        super().__init__()
        self.data = data
        self.split(split_size)

    def setData(self, data, split_size=0):
        self.data = data
        self.split(split_size)

    def getData(self):
        return self.data

    def split(self, split_size):
        X = self.data.copy()
        X.drop(['Class'], axis=1, inplace=True)
        Y = self.data[['Class']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=split_size)


blockchains = [Block.create_genesis_block()]
credit_node_blockchains = [Block.create_genesis_block()]
print('交互信息区块链(交易链)的创世区块是：', blockchains[0].hash)
print('记录信用值区块链（信用链）的创世区块是：', credit_node_blockchains[0].hash)
data = pd.read_csv('creditcard.csv')
data.head()

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(
    data['Amount'].values.reshape(-1, 1))  # 标准化 'Amount' 列到 'scaled_amount' 列
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))  # 标准化 'Time' 列到 'scaled_time' 列
data.drop(['Time', 'Amount'], axis=1, inplace=True)  # 去掉原来的 'Amount' 列和 'Time' 列

scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)  # 将 'scaled_amount' 列放到 data 中第一列
data.insert(1, 'scaled_time', scaled_time)  # 将 'scaled_time' 列放到 data 中第二列

data.head()

data = data.sample(frac=1)  # 对 data 打乱顺序

# amount of fraud classes 492 rows.
fraud_data = data.loc[data['Class'] == 1]  # 取出 data 中所有 1 类样本放到 fraud_data 中
non_fraud_data = data.loc[data['Class'] == 0]  # 取出 data 中所有 0 类样本放到 non_fraud_data 中

normal_distributed_data = pd.concat([fraud_data, non_fraud_data])  # 将 fraud_data 表格和 non_fraud_data 表格整合

# Shuffle dataframe rows
new_data = normal_distributed_data.sample(frac=1, random_state=42)  # 对 normal_distributed_data 打乱顺序并赋给 new_data

new_data.head()

results = {}
aggregator = Aggregator()

datum = Data(data)

Data_Global = datum.sample_smote(0, 0.1)     # 将数据集前百分之10作为初始数据集
Data_Model_1 = datum.sample_smote(0.1, 0.13)
Data_Model_2 = datum.sample_smote(0.13, 0.22)
Data_Model_3 = datum.sample_smote(0.22, 0.25)
Data_Model_4 = datum.sample_smote(0.25, 0.3)
Data_Model_5 = datum.sample_smote(0.3, 0.35)            # 将中间百分之80，不同比例分给10个联邦节点
Data_Model_6 = datum.sample_smote(0.35, 0.4)
Data_Model_7 = datum.sample_smote(0.4, 0.5)
Data_Model_8 = datum.sample_smote(0.5, 0.55)
Data_Model_9 = datum.sample_smote(0.55, 0.6)
Data_Model_10 = datum.sample_smote(0.6, 0.9)
Data_Test = datum.sample_smote(0.9, 1.0)    # 将数据集后百分之10作为测试数据集

GlobalBank = Bank(Data_Global, 0.2)  # 初始化服务器模型
GlobalBank.run(GlobalBank.X_train, GlobalBank.Y_train, load=False)

results['BankG.1'] = GlobalBank.evaluate(GlobalBank.X_test, GlobalBank.Y_test)

GlobalBank.saveModel()

# 初始化三个客户端模型
Bank1 = Bank(Data_Model_1, 0.2)   # Bank1初始化，将百分之80的数据训练，百分之20的数据测试
Bank2 = Bank(Data_Model_2, 0.2)
Bank3 = Bank(Data_Model_3, 0.2)
Bank4 = Bank(Data_Model_4, 0.2)
Bank5 = Bank(Data_Model_5, 0.2)
Bank6 = Bank(Data_Model_6, 0.2)
Bank7 = Bank(Data_Model_7, 0.2)
Bank8 = Bank(Data_Model_8, 0.2)
Bank9 = Bank(Data_Model_9, 0.2)
Bank10 = Bank(Data_Model_10, 0.2)

i = 1
j = 1
k = 1
benlun_xinyong = []
acc_mark = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
jiedian1 = []
jiedian2 = []
jiedian3 = []
jiedian4 = []
jiedian5 = []
jiedian6 = []
jiedian7 = []
jiedian8 = []
jiedian9 = []
jiedian10 = []
for _ in range(151):
    # 将服务器模型参数传递给客户端模型
    Bank1.setWeights(GlobalBank.getWeights())
    transaction1 = str(i) + 'server => node1'
    transaction1_biaoda = str('服务器第' + str(i) + '次将模型传给联邦一节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction1, datetime.datetime.now()))
    print(transaction1_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank2.setWeights(GlobalBank.getWeights())
    transaction2 = str(i) + 'server => node2'
    transaction2_biaoda = str('服务器第' + str(i) + '次将模型传给联邦二节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction2, datetime.datetime.now()))
    print(transaction2_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank3.setWeights(GlobalBank.getWeights())
    transaction3 = str(i) + 'server => node3'
    transaction3_biaoda = str('服务器第' + str(i) + '次将模型传给联邦三节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction3, datetime.datetime.now()))
    print(transaction3_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank4.setWeights(GlobalBank.getWeights())
    transaction8 = str(i) + 'server => node4'
    transaction8_biaoda = str('服务器第' + str(i) + '次将模型传给联邦四节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction8, datetime.datetime.now()))
    print(transaction8_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank5.setWeights(GlobalBank.getWeights())
    transaction9 = str(i) + 'server => node5'
    transaction9_biaoda = str('服务器第' + str(i) + '次将模型传给联邦五节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction9, datetime.datetime.now()))
    print(transaction9_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank6.setWeights(GlobalBank.getWeights())
    transaction10 = str(i) + 'server => node6'
    transaction10_biaoda = str('服务器第' + str(i) + '次将模型传给联邦六节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction10, datetime.datetime.now()))
    print(transaction10_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank7.setWeights(GlobalBank.getWeights())
    transaction11 = str(i) + 'server => node7'
    transaction11_biaoda = str('服务器第' + str(i) + '次将模型传给联邦七节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction11, datetime.datetime.now()))
    print(transaction11_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank8.setWeights(GlobalBank.getWeights())
    transaction12 = str(i) + 'server => node8'
    transaction12_biaoda = str('服务器第' + str(i) + '次将模型传给联邦八节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction12, datetime.datetime.now()))
    print(transaction12_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank9.setWeights(GlobalBank.getWeights())
    transaction13 = str(i) + 'server => node9'
    transaction13_biaoda = str('服务器第' + str(i) + '次将模型传给联邦九节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction13, datetime.datetime.now()))
    print(transaction13_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank10.setWeights(GlobalBank.getWeights())
    transaction14 = str(i) + 'server => node10'
    transaction14_biaoda = str('服务器第' + str(i) + '次将模型传给联邦十节点')
    blockchains.append(Block(blockchains[j - 1].hash, transaction14, datetime.datetime.now()))
    print(transaction14_biaoda + '录入区块' + str(j))
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1

    # 使用对应数据训练客户端模型
    Bank1.run(Bank1.X_train, Bank1.Y_train)
    transaction4 = str(i) + 'node1 => server'
    transaction4_biaoda = str('联邦一节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction4, datetime.datetime.now()))
    print(transaction4_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank2.run(Bank2.X_train, Bank2.Y_train)
    transaction5 = str(i) + 'node2 => server'
    transaction5_biaoda = str('联邦二节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction5, datetime.datetime.now()))
    print(transaction5_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank3.run(Bank3.X_train, Bank3.Y_train)
    transaction6 = str(i) + 'node3 => server'
    transaction6_biaoda = str('联邦三节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction6, datetime.datetime.now()))
    print(transaction6_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank4.run(Bank4.X_train, Bank4.Y_train)
    transaction15 = str(i) + 'node4 => server'
    transaction15_biaoda = str('联邦四节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction15, datetime.datetime.now()))
    print(transaction15_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank5.run(Bank5.X_train, Bank5.Y_train)
    transaction16 = str(i) + 'node5 => server'
    transaction16_biaoda = str('联邦五节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction16, datetime.datetime.now()))
    print(transaction16_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank6.run(Bank6.X_train, Bank6.Y_train)
    transaction17 = str(i) + 'node6 => server'
    transaction17_biaoda = str('联邦六节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction17, datetime.datetime.now()))
    print(transaction17_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank7.run(Bank7.X_train, Bank7.Y_train)
    transaction18 = str(i) + 'node7 => server'
    transaction18_biaoda = str('联邦七节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction18, datetime.datetime.now()))
    print(transaction18_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank8.run(Bank8.X_train, Bank8.Y_train)
    transaction19 = str(i) + 'node8 => server'
    transaction19_biaoda = str('联邦八节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction19, datetime.datetime.now()))
    print(transaction19_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank9.run(Bank9.X_train, Bank9.Y_train)
    transaction20 = str(i) + 'node9 => server'
    transaction20_biaoda = str('联邦九节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction20, datetime.datetime.now()))
    print(transaction20_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1
    Bank10.run(Bank10.X_train, Bank10.Y_train)
    transaction21 = str(i) + 'node10 => server'
    transaction21_biaoda = str('联邦十节点第' + str(i) + '次训练完毕，向服务器发送模型,并录入区块' + str(j))
    blockchains.append(Block(blockchains[j - 1].hash, transaction21, datetime.datetime.now()))
    print(transaction21_biaoda)
    print("交易链的区块的哈希是", blockchains[j].hash)
    j = j + 1

    # 信用评估机制
    acc1 = Bank1.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦一节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc1)
    acc2 = Bank2.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦二节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc2)
    acc3 = Bank3.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦三节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc3)
    acc4 = Bank4.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦一节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc4)
    acc5 = Bank5.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦二节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc5)
    acc6 = Bank6.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦三节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc6)
    acc7 = Bank7.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦一节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc7)
    acc8 = Bank8.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦二节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc8)
    acc9 = Bank9.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦三节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc9)
    acc10 = Bank10.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 联邦三节点此时模型对于服务器内部数据集准确度
    benlun_xinyong.append(acc10)
    benlun_max = benlun_xinyong.index(max(benlun_xinyong))
    print('------------------------------------------------------------------------------')
    print(benlun_xinyong)
    print(benlun_max)
    # acc_mark[benlun_max] = acc_mark[benlun_max] + 1
    top_avg = max(np.median(benlun_xinyong), np.average(benlun_xinyong))
    down_avg = min(np.median(benlun_xinyong), np.average(benlun_xinyong))
    # print(top_avg)
    for q in range(10):
        if benlun_xinyong[q] == benlun_max:
            acc_mark[q] = i/(i+1) * acc_mark[q] + 1/(k+1) * 1
            print('第' + str(q + 1) + '个联邦节点加入本轮聚合队列')
        elif benlun_xinyong[q] >= top_avg:
            acc_mark[q] = i/(i+1) * acc_mark[q] + 1/(k+1) * 3/4
            print('第'+str(q + 1)+'个联邦节点加入本轮聚合队列')
        elif benlun_xinyong[q] >= down_avg:
            acc_mark[q] = i/(i+1) * acc_mark[q] + 1/(k+1) * 1/2
        else:
            acc_mark[q] = i/(i+1) * acc_mark[q]
    print('本轮各联邦节点的信用：', acc_mark)
    jiedian1.append(acc_mark[0])
    jiedian2.append(acc_mark[1])
    jiedian3.append(acc_mark[2])
    jiedian4.append(acc_mark[3])
    jiedian5.append(acc_mark[4])
    jiedian6.append(acc_mark[5])
    jiedian7.append(acc_mark[6])
    jiedian8.append(acc_mark[7])
    jiedian9.append(acc_mark[8])
    jiedian10.append(acc_mark[9])
    # 将实时的信用值上链
    now_time = datetime.datetime.now()
    transaction7 = str(now_time) + str(acc_mark[0]) + str(acc_mark[1]) + str(acc_mark[2]) + str(acc_mark[3]) + str(acc_mark[4]) + str(acc_mark[5]) + str(acc_mark[6]) + str(acc_mark[7]) + str(acc_mark[8]) + str(acc_mark[9])
    credit_node_blockchains.append(Block(blockchains[i - 1].hash, transaction7, datetime.datetime.now()))
    print('更新第' + str(i) + '次信用值,记录进' + str(k) + '个区块')
    print("该信用链的区块哈希是", credit_node_blockchains[i].hash)
    k = k + 1

    # 集成客户端模型参数
    delta = aggregator.aggregate(GlobalBank.getWeights(), Bank1.getWeights(), Bank2.getWeights(), Bank3.getWeights(),
                                 Bank4.getWeights(), Bank5.getWeights(), Bank6.getWeights(), Bank7.getWeights(),
                                 Bank8.getWeights(), Bank9.getWeights(), Bank10.getWeights())

    # 将集成后的客户端模型参数分配到服务器模型上
    GlobalBank.setWeights(delta)
    print('-----------------------------------')
    print('服务器第' + str(i) + '次聚合模型完毕')
    print('-----------------------------------')
    i = i + 1
    benlun_xinyong.clear()

GlobalBank.saveModel()

results['Bank1.1'] = Bank1.evaluate(Bank1.X_test, Bank1.Y_test)  # 用 Bank1 客户端数据测试 Bank1 模型
results['Bank2.1'] = Bank2.evaluate(Bank2.X_test, Bank2.Y_test)  # 用 Bank2 客户端数据测试 Bank2 模型
results['Bank3.1'] = Bank3.evaluate(Bank3.X_test, Bank3.Y_test)  # 用 Bank3 客户端数据测试 Bank3 模型
results['Bank4.4'] = Bank4.evaluate(Bank4.X_test, Bank4.Y_test)  # 用 Bank4 客户端数据测试 Bank4 模型

results['BankG.2'] = GlobalBank.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 用 GlobalBank 服务器数据测试 GlobalBank 模型

GlobalBank.setData(Data_Test, 0.9)  # 将服务器数据改为 Data_Test
results['BankG.3'] = GlobalBank.evaluate(GlobalBank.X_test, GlobalBank.Y_test)  # 用 Data_Test 数据测试 GlobalBank 模型
# for m in range(10):
#     print(acc_mark[m])

print(blockchains)
print(credit_node_blockchains)
print(jiedian1)
print(jiedian2)
print(jiedian3)
print(jiedian4)
print(jiedian5)
print(jiedian6)
print(jiedian7)
print(jiedian8)
print(jiedian9)
print(jiedian10)
results
