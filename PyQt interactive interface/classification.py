import wfdb
import pywt
#import seaborn
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
import random
#归一化
from sklearn.preprocessing import MinMaxScaler
RATIO = 0.2
random_seed = 42
random.seed(random_seed )
np.random.seed(random_seed )
tf.random.set_seed(random_seed )

#X是长度为300的心拍序列
def loadData_test(selectedNumber):
    numberSet = [selectedNumber]
    dataSet = []
    lableSet = []
    for n in numberSet:
        #对心电信号进行去噪、心拍分割
        getDataSet(n, dataSet, lableSet)
    # 转numpy数组
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    X = dataSet.reshape(-1, 300, 1)
    Y = lableSet.reshape(-1)
    #对心拍进行归一化
    scaler = MinMaxScaler()
    print('开始归一化')
    # 对每个样本的特征进行归一化
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.fit_transform(X[i, :, :])
    print('归一化结束')
    return X, Y

# 小波去噪预处理
def Denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('D:/大学/毕设/MIT-BIH-360/' + number, channel_names=['MLII'])
    #p_signal：模拟信号值（physical = True时可调用）
    #d_signal：数字信号值（physical = False时可调用）
    data = record.p_signal.flatten()
    rdata = Denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('D:/大学/毕设/MIT-BIH-360/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            # 基于经验值，基于R峰向前取100个点，向后取200个点
            x_train = rdata[Rlocation[i] - 100:Rlocation[i] + 200]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return