# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
from scipy import signal
import scipy.io as sio


#读取特征矩阵
def load_data():
    if os.path.isdir(r'G:/BNU_second_time_data近红外/'):
        root_dir = r'G:/BNU_second_time_data近红外/4850_single_mat/'
        damaged_file = r'G:/BNU_second_time_data近红外/spec_cleaning_damaged.csv'
        healthy_file = r'G:/BNU_second_time_data近红外/spec_cleaning_healthy.csv'
        moldy01_file = r'G:/BNU_second_time_data近红外/spec_cleaning_moldy01.csv'
        moldy02_file = r'G:/BNU_second_time_data近红外/spec_cleaning_moldy02.csv'
        moldy03_file = r'G:/BNU_second_time_data近红外/spec_cleaning_moldy03.csv'
        test01_file = r'G:/BNU_second_time_data近红外/spec_cleaning_test01.csv'
        test02_file = r'G:/BNU_second_time_data近红外/spec_cleaning_test02.csv'
        test03_file = r'G:/BNU_second_time_data近红外/spec_cleaning_test03.csv'
        whitemoldy_file = r'G:/BNU_second_time_data近红外/spec_cleaning_whitemoldy.csv'
    else:
        root_dir = r'D:/ALL_DATA/BNU_second_time_data/4850_single_mat/'
        damaged_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_damaged.csv'
        healthy_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_healthy.csv'
        moldy01_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_moldy01.csv'
        moldy02_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_moldy02.csv'
        moldy03_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_moldy03.csv'
        test01_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_test01.csv'
        test02_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_test02.csv'
        test03_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_test03.csv'
        whitemoldy_file = r'D:/ALL_DATA/BNU_second_time_data/spec_cleaning_whitemoldy.csv'
    # 读取所有光谱数据
    df_damaged = pd.read_csv(damaged_file)
    df_healthy = pd.read_csv(healthy_file)
    df_moldy01 = pd.read_csv(moldy01_file)
    df_moldy02 = pd.read_csv(moldy02_file)
    df_moldy03 = pd.read_csv(moldy03_file)
    df_whitemoldy = pd.read_csv(whitemoldy_file)
    
    df_test01 = pd.read_csv(test01_file)
    df_test02 = pd.read_csv(test02_file)
    df_test03 = pd.read_csv(test03_file)
    # 合并训练集和测试集
    train_all_data = pd.concat([df_damaged, df_healthy, df_moldy01, df_moldy02, df_moldy03, df_whitemoldy],axis=0)
    test_all_data = pd.concat([df_test01, df_test02, df_test03],axis=0)
    return train_all_data, test_all_data

def load_ti_data(train_all_data, test_all_data,pinzhong_i,idx, ti, train_spectral=True):
    # 提取单品种,包含三个梯度含水
    train_pinzhong_i = train_all_data[train_all_data['file_path'].str.contains(pinzhong_i)]
    test_pinzhong_i = test_all_data[test_all_data['file_path'].str.contains(pinzhong_i)]
    test_mix_i = test_all_data[test_all_data['file_path'].str.contains('mix-')]
    train_t1 = train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('t1-')]
    train_t2 = train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('t2-')]
    train_t3 = train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('t3-')]
    # print('train_pinzhong_i:',train_pinzhong_i.shape,'test_pinzhong_i, no mix:',test_pinzhong_i.shape,
    #         'train_t1:', train_t1.shape, 'train_t2:', train_t2.shape, 'train_t3:', train_t3.shape)
    if ti == 1:
        train_ti = train_t1
    elif ti == 2:
        train_ti = train_t2
    elif ti == 3:
        train_ti = train_t3
    # 训练数据，根据文件夹提取数据，文件名可能改变了，但是文件夹里的数据是准确的
    hp_train = train_ti[train_ti['file_path'].str.contains('healthy')]
    dp_train = train_ti[train_ti['file_path'].str.contains('damaged')]
    mp01_train = train_ti[train_ti['file_path'].str.contains('moldy01')]
    mp02_train = train_ti[train_ti['file_path'].str.contains('moldy02')]
    mp03_train = train_ti[train_ti['file_path'].str.contains('moldy03')]
    wp_train = train_ti[train_ti['file_path'].str.contains('whitemoldy')]
    data_train = pd.concat([hp_train, dp_train, mp01_train, mp02_train, mp03_train, wp_train],axis=0)
    # 根据路径里的标签 提取数据
    hp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +0)+'-')]
    dp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +1)+'-')]
    mp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +2)+'-')]
    wp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +3)+'-')]
    data_test_ = pd.concat([hp_test, dp_test, mp_test, wp_test],axis=0)
    hp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +0)+'-')]
    dp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +1)+'-')]
    mp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +2)+'-')]
    wp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +3)+'-')]
    data_mix = pd.concat([hp_mix, dp_mix, mp_mix, wp_mix],axis=0)
    data_test = pd.concat([data_test_, data_mix],axis=0)
    if train_spectral:
        # 去掉路径, 提取出光谱
        train_data = data_train.iloc[:, 1:]
        test_data = data_test.iloc[:, 1:]
        data_train_arr = np.array(train_data)
        data_test_arr = np.array(test_data)
    else:
        # 去掉路光谱, 提取出路径
        drop_lie_list = list(range(0,288)); drop_lie = [str(i) for i in drop_lie_list]
        data_train.drop(drop_lie, axis=1, inplace=True)
        data_test.drop(drop_lie, axis=1, inplace=True)
        data_train_arr = np.array(data_train)
        data_test_arr = np.array(data_test)
    return data_train_arr, data_test_arr

def load_t123_data(train_all_data, test_all_data,pinzhong_i,idx,all_data=3):
    # 提取单品种,包含三个梯度含水
    train_pinzhong_i = train_all_data[train_all_data['file_path'].str.contains(pinzhong_i)]
    test_pinzhong_i = test_all_data[test_all_data['file_path'].str.contains(pinzhong_i)]
    test_mix_i = test_all_data[test_all_data['file_path'].str.contains('mix-')]
    hp_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('healthy')]).iloc[:, 1:]
    dp_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('damaged')]).iloc[:, 1:]
    mp01_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('moldy01')]).iloc[:, 1:]
    mp02_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('moldy02')]).iloc[:, 1:]
    mp03_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('moldy03')]).iloc[:, 1:]
    wp_train = (train_pinzhong_i[train_pinzhong_i['file_path'].str.contains('whitemoldy')]).iloc[:, 1:]
    # 打乱随机取1/3的数据
    hp_train = np.array(hp_train)
    dp_train = np.array(dp_train)
    mp01_train = np.array(mp01_train)
    mp02_train = np.array(mp02_train)
    mp03_train = np.array(mp03_train)
    wp_train = np.array(wp_train)
    hp_train_part = shuffle(hp_train, random_state=0)[:(hp_train.shape[0])//all_data]
    dp_train_part = shuffle(dp_train, random_state=0)[:(dp_train.shape[0])//all_data]
    mp01_train_part = shuffle(mp01_train, random_state=0)[:(mp01_train.shape[0])//all_data]
    mp02_train_part = shuffle(mp02_train, random_state=0)[:(mp02_train.shape[0])//all_data]
    mp03_train_part = shuffle(mp03_train, random_state=0)[:(mp03_train.shape[0])//all_data]
    wp_train_part = shuffle(wp_train, random_state=0)[:(wp_train.shape[0])//all_data]
    data_train = np.concatenate([hp_train_part, dp_train_part, mp01_train_part, 
                            mp02_train_part, mp03_train_part, wp_train_part],axis=0)
    # 根据标签提取数据
    hp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +0)+'-')]
    dp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +1)+'-')]
    mp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +2)+'-')]
    wp_test = test_pinzhong_i[test_pinzhong_i['file_path'].str.contains('/'+str(4*idx +3)+'-')]
    data_test_ = pd.concat([hp_test, dp_test, mp_test, wp_test],axis=0)
    hp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +0)+'-')]
    dp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +1)+'-')]
    mp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +2)+'-')]
    wp_mix = test_mix_i[test_mix_i['file_path'].str.contains('/'+str(4*idx +3)+'-')]
    data_mix = pd.concat([hp_mix, dp_mix, mp_mix, wp_mix],axis=0)
    data_test = pd.concat([data_test_, data_mix],axis=0)
    # 去掉路径, 提取出光谱
    train_data = data_train
    test_data = data_test.iloc[:, 1:]
    data_train_arr = np.array(train_data)
    data_test_arr = np.array(test_data)
    return data_train_arr, data_test_arr

def load_acgan_data(seed_i, pinzhong_i,T123, da_times):
    epoch_i = str(seed_i + 3991)
    excel_path = r'D:/research3/Train_model/GAN_data/'+ pinzhong_i+'/' + \
                    pinzhong_i + T123 +'gan_dax'+ str(da_times+1)+'_'+epoch_i+'.csv'
    print('load:', excel_path)
    da_data = pd.read_csv(excel_path)
    da_data_arr = np.array(da_data)
    return da_data_arr

def load_DAacgan_data(seed_i, pinzhong_i,T123, da_times):
    epoch_i = str(seed_i + 4991)
    excel_path = r'D:/research3/Train_model/DAGAN_data/'+ pinzhong_i+'DA/' + \
                    pinzhong_i + T123 +'gan_dax'+ str(da_times+1)+'_'+epoch_i+'.csv'
    print('load:', excel_path)
    da_data = pd.read_csv(excel_path)
    da_data_arr = np.array(da_data)
    return da_data_arr

def write_excel(acc, f1, kappa, pre, recall, out_excel):
    col_lie = ['黑花生','大白沙','七彩','四粒红','小白沙','多品种']
    index_hang = pd.Series(['1_None_x2', '2_Erase_x2', '3_Noise_x2', '4_TPW_x2', '5_DA_x2', 
            '6_None_x4', '7_Erase_x4', '8_Noise_x4', '9_TPW_x4', '10_DA_x4', ])
    if os.path.exists(out_excel):
        print('文件存在! 删除', out_excel)
        os.remove(out_excel)
    writer = pd.ExcelWriter(out_excel,engine='openpyxl')# pylint: disable=abstract-class-instantiated
    acc_data = pd.DataFrame(acc)
    acc_data.columns = col_lie
    acc_data.index = index_hang
    acc_data.to_excel(writer, sheet_name='Accuracy',)
    f1_data = pd.DataFrame(f1)
    f1_data.columns = col_lie
    f1_data.index = index_hang
    f1_data.to_excel(writer, sheet_name='F1',)
    kappa_data = pd.DataFrame(kappa)
    kappa_data.columns = col_lie
    kappa_data.index = index_hang
    kappa_data.to_excel(writer, sheet_name='Kappa',)
    pre_data = pd.DataFrame(pre)
    pre_data.columns = col_lie
    pre_data.index = index_hang
    pre_data.to_excel(writer, sheet_name='Precision',)
    recall_data = pd.DataFrame(recall)
    recall_data.columns = col_lie
    recall_data.index = index_hang
    recall_data.to_excel(writer, sheet_name='Recall',)
    writer.close()
    return

def single_data_train12(da_method, methods, pinzhong_i, train_all_data, test_all_data, eb, da_times, seed_i):
    data_train_arrt1, data_test_arrt1 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=1)
    data_train_arrt2, data_test_arrt2 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=2)
    # data_train_arrt3, data_test_arrt3 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=3)
    train_data = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
    # test_data1 = np.concatenate((data_test_arrt1, data_test_arrt2), axis=0)
    # test_data = np.concatenate((test_data1, data_test_arrt3), axis=0)
    test_data = data_test_arrt1
    # 改标签
    train_data[:, -1] = train_data[:, -1] - idx*4  # 标签改为0-3
    test_data[:, -1] = test_data[:, -1] - idx*4  # 标签改为0-3
    data_train_shuf = shuffle(train_data, random_state=seed_i)
    data_test_shuf = shuffle(test_data, random_state=0)
    # 选取20%的数据训练 80002000
    data_num =  data_train_shuf.shape[0]  # 2000  
    data_train_ = data_train_shuf[: data_num, :]
    # DA
    data_train_DA = choose_da(seed_i, pinzhong_i, data_train_, eb, 
                            ti='T12', da_method=da_method_i, times=da_times)
    data_train_ti = shuffle(data_train_DA, random_state=0)
    data_test_ti = shuffle(data_test_shuf, random_state=0)
    data_train_ti = choose_eb(eb, data_train_ti)
    data_test_ti = choose_eb(eb, data_test_ti)
    print('train.shape', data_train_ti.shape, 'test.shape', data_test_ti.shape, 
            da_method_i, pinzhong_i, methods_i, 't12', data_num)
    train_spectral = data_train_ti[:, :-1]
    train_label = np.int64(data_train_ti[:, -1])
    test_spectral = data_test_ti[:, :-1]
    test_label = np.int64(data_test_ti[:, -1])
    weight_path = gen_weight_path(methods_i, da_method_i, pinzhong_i, ti=12)
    y_pred = choose_method(train_spectral, train_label, test_spectral, test_label, methods_i,
                            weight_path=weight_path, pinzhong_i=pinzhong_i)
    ACC = gen_acc(methods_i, da_method_i, pinzhong_i,y_pred, test_label, ti=12)
    return ACC


def DA_big_smal(data_train, ti=None, plus_xianyan=False, times=None):
    DA_data = np.array([])
    if ti=='t12':
        start = -200; end = 800
    else:
        start = -500; end = 500
    if plus_xianyan:
        if ti == 1: # 梯度1，(-0.2, 1.5)
            start = -200
            end = 1500
        elif ti == 2: # 梯度1，(-1.2, 1.2)
            start = -1200
            end = 1200
        elif ti == 3: # 梯度1，(-1.5, 0.2)
            start = -1500
            end = 200
        else: #elif ti == 123: # 梯度123，(-1.1, 1.1)
            pass
    class_num = int(np.max(data_train[:, -1]) + 1)
    for label_i in range(class_num): # 遍历类别
        # 绘制子图
        # 1.分别计算每个梯度的类内光谱差
        # 1.1计算每类的平均光谱
        class_i = data_train[data_train[:, -1]==label_i] # t1
        # 1.2按1901nm处的波段排序, 第175个波段
        train_t1_sort = class_i[class_i[:,174].argsort()]
        # 1.2 计算类内sd, 不包括标签
        big_sample = train_t1_sort[ : train_t1_sort.shape[0]//2, :]
        small_sample = train_t1_sort[train_t1_sort.shape[0]//2 : , :]
        class_mean_t1_big = np.mean(big_sample, axis=0)
        class_mean_t1_smal = np.mean(small_sample, axis=0)

        sd_t1 = np.abs(class_mean_t1_big-class_mean_t1_smal)
        sd_t1_2d = np.expand_dims(sd_t1, axis=0)
        # 生成随机倍数, 4倍原始数据
        for i in range(times):
            # random_1 = np.expand_dims(np.random.randint(900, 1100,size=class_i.shape[0])/1000, axis=-1)
            random_1 = np.expand_dims(np.random.randint(start, end,size=class_i.shape[0])/1000, axis=-1)
            sd_t1_2d_repeat = sd_t1_2d.repeat(class_i.shape[0], axis=0)
            sd_da_1 = sd_t1_2d_repeat * random_1
            sd_da_1[:, -1] = 0 # 最后一列的标签不处理
            class_i_da = class_i + sd_da_1
            if np.any(DA_data):
                DA_data = np.append(DA_data, class_i_da, axis=0)
            else:
                DA_data = class_i_da

            # plt.plot(np.max(class_i_da[:, :-1],axis=0), color='darkred',label='da-max')
            # plt.plot(np.min(class_i_da[:, :-1],axis=0), color='darkgreen',label='da-min')
            # plt.plot(np.max(class_i[:, :-1],axis=0), color='red',label='or-max')
            # plt.plot(np.min(class_i[:, :-1],axis=0), color='lime',label='or-min')
            # plt.legend()
            # plt.show()
    DA_data = np.append(DA_data, data_train, axis=0) # 相当于原始数据的5倍，1+4
    # DA_arr = np.array(DA_data)
    return DA_data

def DArandom_erase(data_train, times=None):
    data_train_spe = data_train[:, :-1]
    data_label = np.expand_dims(data_train[:, -1], axis=-1)
    DA_data = np.array([])
    for i in range(times):
        second_para = np.random.randint(0, data_train_spe.shape[1]-10, size=data_train_spe.shape[0])
        data_spe_erase = data_train_spe.copy()
        for idx in range(data_spe_erase.shape[0]):
            data_spe_erase[idx, second_para[idx]:second_para[idx]+10]=0
        data_erase_i = np.concatenate((data_spe_erase, data_label), axis=1)
        if np.any(DA_data):
            DA_data = np.append(DA_data, data_erase_i, axis=0)
        else:
            DA_data = data_erase_i
    DA_data = np.append(DA_data, data_train, axis=0)
    return DA_data


def DA_tpw(data_train, times=None):
    DA_data = np.array([])
    class_num = int(np.max(data_train[:, -1]) + 1)
    for label_i in range(class_num): # 遍历类别
        # 生成随机倍数, 4倍原始数据
        for i in range(times):
            class_i = data_train[data_train[:, -1]==label_i] # t1
            # 生成随机权重
            random_value = np.expand_dims(np.random.uniform(0, 1, class_i.shape[0]), axis=-1)
            # 生成随机挑选的光谱的索引
            random_spe01 = np.random.randint(0, class_i.shape[0], size=class_i.shape[0])
            random_spe02 = np.random.randint(0, class_i.shape[0], size=class_i.shape[0])
            spe01 = class_i[random_spe01]
            spe02 = class_i[random_spe02]
            da_spe = spe01 * random_value + spe02 * (1-random_value)
            da_spe[:, -1] = label_i # 设定标签值，防止计算产生的精度误差
            if np.any(DA_data):
                DA_data = np.append(DA_data, da_spe, axis=0)
            else:
                DA_data = da_spe
    DA_data = np.append(DA_data, data_train, axis=0) # 相当于原始数据的5倍，1+4
    # DA_arr = np.array(DA_data)
    return DA_data

def DA_smooth(data_train, times):
    # seed_dict = {'1000':0.116,'1001':0.117, '1002':0.118, '1003':0.119, '1004':0.120, 
    #             '1005':0.121, '1006':0.122, '1007':0.123,'1008':0.124,'1009':0.125}
    # second_para = seed_dict[str(seed_i)]
    data_train_spe = data_train[:, :-1]
    data_label = np.expand_dims(data_train[:, -1], axis=-1)
    DA_data = np.array([])
    for i in range(times):
        second_para = np.random.randint(116, 126)/1000
        b, a = signal.butter(8, second_para)
        data_spe_smooth = signal.filtfilt(b, a, data_train_spe, axis=-1,padlen=30)
        data_smooth_i = np.concatenate((data_spe_smooth, data_label), axis=1)
        if np.any(DA_data):
            DA_data = np.append(DA_data, data_smooth_i, axis=0)
        else:
            DA_data = data_smooth_i
    DA_data = np.append(DA_data, data_train, axis=0)
    return DA_data


def DA_noise(data_train, times):
    DA_data = np.array([])
    for i in range(times):
        # 生成±0.01的随机噪声
        noise_ = np.random.randint(-100, 100,size=data_train.shape)/10000
        noise_[:, -1] = 0 # 最后一列的标签不处理
        noise_data_i = data_train + noise_
        if np.any(DA_data):
            DA_data = np.append(DA_data, noise_data_i, axis=0)
        else:
            DA_data = noise_data_i
    DA_data = np.append(DA_data, data_train, axis=0)
    return DA_data


def choose_da(seed_i, pinzhong_i, data_train, eb, ti=None, da_method=None, times=None):
    if da_method == 'DA_xianyan':
        DA_data = DA_big_smal(data_train, ti, plus_xianyan=True, times=times)
    elif da_method == 'DA':
        DA_data = DA_big_smal(data_train, ti, plus_xianyan=False, times=times)
    elif da_method == 'noise':
        DA_data = DA_noise(data_train, times=times)
    elif da_method == 'None':
        DA_data = data_train
    elif da_method == 'TPW':
        DA_data = DA_tpw(data_train, times=times)
    elif da_method == 'smooth':
        DA_data = DA_smooth(data_train, times=times)
    elif da_method == 'Erase':
        DA_data = DArandom_erase(data_train, times=times)
    elif da_method == 'ACGAN':
        DA_data = load_acgan_data(seed_i, pinzhong_i,ti, times)
    elif da_method == 'DAACGAN':
        DA_data = load_DAacgan_data(seed_i, pinzhong_i,ti, times)
    else:
        raise ValueError
    
    # da_df = pd.DataFrame(DA_data)
    # da_df.to_csv('D:/research3/Train_model/GAN_data/'+ pinzhong_i+da_method+ str(ti)+ 'dax'+str(times+1)+'.csv',index=False)
    return DA_data

def choose_eb(eb, DA_data):
    if eb.any():
        # print('执行eb！')
        DA_eb = DA_data[:, eb]
        DA_label = np.expand_dims(DA_data[:, -1], axis=-1)
        eb_data = np.concatenate((DA_eb, DA_label), axis=-1)
    else:
        eb_data = DA_data
    return eb_data

class SE_Module(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(SE_Module, self).__init__()
        # self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.AdaptiveAvgPool1d(1) # 144
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid(),)
    def forward(self, x):
        b, c, _, = x.size()
        y = self.squeeze(x).view(b, c)       # 32,96
        z = self.excitation(y)
        z = z.view(b, c, 1,)
        return x * z.expand_as(x)

from math import log

class ECA_Module(nn.Module):
    def __init__(self, channel,gamma=2, b=1):
        super(ECA_Module, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _,  = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1,-2))
        y = y.transpose(-1,-2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CNN1D(nn.Module):
    def __init__(self, in_dim, num_bands,out_classes):
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.out_classes = out_classes
        # in_channels, out_channels, kernel_size
        self.conv01 = nn.Sequential(
            # nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=1, groups=in_dim),
            nn.Conv1d(self.in_dim, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))
        # self.conv02 = SE_Module(channel=)
        self.conv02 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))
        self.pool01 = nn.Sequential(
            # nn.MaxPool1d(2, stride=2))
            nn.Conv1d(32, 64, 3, padding=1,stride=2,bias=False),)
        self.conv03 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.conv04 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.conv05 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.conv06 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.conv07 = nn.Sequential(
            nn.Conv1d(256+64, 512, 3, padding=1,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.pool02 = nn.Sequential(
            # nn.MaxPool1d(2, stride=2))
            nn.Conv1d(512, 512, 3, padding=1,stride=2,bias=False),)
        self.out_layer = nn.Sequential(
            # nn.Conv1d(256, 512, 3, padding=1,stride=2,bias=False),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            nn.Flatten(),
            # nn.Conv1d(32, self.out_classes, 3, padding=1, bias=False),
            nn.Linear(512 * int(np.ceil(self.num_bands/4)),self.out_classes, bias=False))

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)
        x1 = self.pool01(x)
        x1 = self.conv03(x1)
        x2 = self.conv04(x1)
        x_cat01 = torch.cat([x1,x2], dim=-2)
        x_conv05 = self.conv05(x_cat01)
        x_conv06 = self.conv06(x_conv05)
        x_cat02 = torch.cat([x_conv06,x2], dim=-2)
        x_conv07 = self.conv07(x_cat02)
        pool2 = self.pool02(x_conv07)
        output = self.out_layer(pool2)
        return output

class NNReshape(nn.Module):

    def __init__(self, *new_shape):
        super(NNReshape, self).__init__()
        self._new_shape = new_shape

    def forward(self, x):
        new_shape = (x.size(i) if self._new_shape[i] == 0 else self._new_shape[i] for i in range(len(self._new_shape)))
        return x.view(*new_shape)


class ONE_D_CNN(nn.Module):
    # x_dim=输入波段, c_dim=输出类别,
    def __init__(self, x_dim, c_dim, dim=96, ):
        super(ONE_D_CNN, self).__init__()
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                        nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding),
                        nn.BatchNorm1d(out_dim),
                        # nn.LayerNorm(out_dim),
                        nn.LeakyReLU(0.1),
                        # nn.GELU(),
                        )

        self.ls01 = nn.Sequential(  # (N, 1, 288)-->N, dim=96, 288
            conv_norm_lrelu(x_dim, dim),)
        self.ls02 = nn.Sequential(
            ECA_Module(dim),
            # conv_norm_lrelu(dim, dim),
            conv_norm_lrelu(dim, dim, stride=2),  # (N, 96 , 144)
            conv_norm_lrelu(dim, dim, stride=2),  # (N, 96 , 72) # 根据64的数据加一层池化

            conv_norm_lrelu(dim, dim * 2),
            conv_norm_lrelu(dim * 2, dim * 2, stride=2),# (N, dim*2, 36) 3次步长2，变成36
            conv_norm_lrelu(dim * 2, dim * 2, stride=2, padding=1),# (N, dim*2, 36) 3次步长2，变成36
            conv_norm_lrelu(dim * 2, dim * 3, stride=1, padding=1),)# out=288,18
            
        self.ls03 = nn.Sequential(
            conv_norm_lrelu(dim * 3, dim * 3, kernel_size=3, stride=2, padding=1),# out=288,9
            conv_norm_lrelu(dim * 3, dim * 3, kernel_size=3, stride=2, padding=1),  # out=288,7
            # conv_norm_lrelu(dim * 3, dim * 3, kernel_size=3, stride=1, padding=0),  # out=288,5
            conv_norm_lrelu(dim * 3, dim * 3, kernel_size=1, stride=1, padding=0),# out=288,5
            ECA_Module(dim * 3),)  # out=288,5
        self.ls04 = nn.Sequential(
            nn.AvgPool1d(kernel_size=5), # (N, 288,1)
            # nn.Conv1d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=0),
            )  
        self.ls05 = nn.Sequential(
            NNReshape(-1, dim * 3),)  # (N, dim*2)

        self.l_c_logit = nn.Linear(dim * 3, c_dim)  # (N, c_dim)

    def forward(self, x):
        # x: (N, x_dim, 32, 32)
        ls01 = self.ls01(x)
        ls02 = self.ls02(ls01)
        ls03 = self.ls03(ls02)
        ls04 = self.ls04(ls03)
        feat = self.ls05(ls04)
        # feat = self.ls02(ls04)
        l_c_logit = self.l_c_logit(feat)
        return l_c_logit

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_bands = 128
    # net = CNN1D(in_dim=1,num_bands= num_bands,out_classes=3).to(device)
    # # summary(net, input_size=(10,1,num_bands,))
    # input = torch.randn(10,  1,256,)
    # out = net(input)
    # num_bands = 128
    # net = CNN1D(in_dim=1,num_bands= num_bands,out_classes=3).to(device)
    # input = torch.randn(5,  1,128,).to(device)
    # out = net(input)
    # print( '\n', out.shape)
    # summary(net, input_size=(1,num_bands,))

    Classify_N = ONE_D_CNN(1,4).to(device)
    input = torch.randn(32,  1,288,).to(device)
    out = Classify_N(input)
    # print(Classify_N)
    torch.save(Classify_N.state_dict(), './aaaaaaa.pth')
    print( '\n', out.shape)



class trainDataset(Dataset):
    def __init__(self, train_spe, train_label):
        self.train_x = train_spe
        self.train_y = train_label

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        # 输入shape (1, 128)
        train_x = np.expand_dims(self.train_x[idx, :], axis=0)
        train_y = self.train_y[idx]
        return train_x, train_y

class valDataset(Dataset):
    def __init__(self, val_spe, val_label):
        self.val_x = val_spe
        self.val_y = val_label

    def __len__(self):
        return len(self.val_y)

    def __getitem__(self, idx):
        # 输入shape (1, 128)
        val_x = np.expand_dims(self.val_x[idx, :], axis=0)
        val_y = self.val_y[idx]
        return val_x, val_y


def choose_da_img(seed_i, pinzhong_i, data_train_, eb, ti, da_method, times):
    DA_data = np.array([])
    # data_train_.shape=[样本，路径/标签/是否增强] 2000,2
    da_zero = np.zeros((data_train_.shape[0], 1))
    da_one = np.ones((data_train_.shape[0], 1))*1
    da_two = np.ones((data_train_.shape[0], 1))*2
    da_three = np.ones((data_train_.shape[0], 1))*3
    if da_method == 'None':
        DA_data = np.concatenate((data_train_, da_zero), axis=-1)
    else:
        for i in range(times):
            if i ==0:
                DA_data_i = np.concatenate((data_train_, da_one), axis=-1)
            elif i ==1:
                DA_data_i = np.concatenate((data_train_, da_two), axis=-1)
            elif i ==2:
                DA_data_i = np.concatenate((data_train_, da_three), axis=-1)
            
            if np.any(DA_data):
                # DA_data_i = np.concatenate((data_train_, da_one), axis=-1)
                DA_data = np.append(DA_data, DA_data_i, axis=0)
            else:
                DA_data = DA_data_i
        yuanshi = np.concatenate((data_train_, da_zero), axis=-1)
        DA_data = np.append(yuanshi, DA_data, axis=0)
    return DA_data

# 原始
class T_or_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_or_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

# 噪声
class T_noise_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # 生成噪声
        noise_i = np.random.randint(-100, 100, size=mat_img.shape)/10000
        if self.bool_da[idx]:
            mat_img = mat_img + noise_i
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            mat_img[mask==0] = 0
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_noise_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # 生成噪声
        noise_i = np.random.randint(-100, 100, size=mat_img.shape)/10000
        if self.bool_da[idx]:
            mat_img = mat_img + noise_i
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            mat_img[mask==0] = 0
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

# 旋转2
class T_rotate2_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        if self.bool_da[idx] : # 逆时针旋转90度
            resized_img = np.rot90(resized_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_rotate2_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        if self.bool_da[idx] : # 逆时针旋转90度
            resized_img = np.rot90(resized_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label


# 旋转4
class T_rotate4_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        if self.bool_da[idx]==1: # 逆时针旋转90度
            resized_img = np.rot90(resized_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
        elif self.bool_da[idx]==2:
            resized_img = resized_img[::-1,:,:].copy()# 左右=水平, 带-1索引的必须加.copy()
        elif self.bool_da[idx]==3:
            resized_img = resized_img[:,::-1,:].copy()# 上下=垂直
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_rotate4_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        if self.bool_da[idx]==1: # 逆时针旋转90度
            resized_img = np.rot90(resized_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
        elif self.bool_da[idx]==2:
            resized_img = resized_img[::-1,:,:].copy()# 左右=水平, 带-1索引的必须加.copy()
        elif self.bool_da[idx]==3:
            resized_img = resized_img[:,::-1,:].copy()# 上下=垂直
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

# 擦除
class T_erase_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # 随机擦除
        erase_x = np.random.randint(0, mat_img.shape[0]-5)
        erase_y = np.random.randint(0, mat_img.shape[1]-5)
        if self.bool_da[idx]: # 逆时针旋转90度
            mat_img[erase_x:erase_x+5, erase_y:erase_y+5] = 0
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            mat_img[mask==0] = 0
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)

        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_erase_Dataset(Dataset):
    def __init__(self, data_train_shuf, image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        # 随机擦除
        erase_x = np.random.randint(0, mat_img.shape[0]-5)
        erase_y = np.random.randint(0, mat_img.shape[1]-5)
        if self.bool_da[idx]: # 逆时针旋转90度
            mat_img[erase_x:erase_x+5, erase_y:erase_y+5] = 0
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            mat_img[mask==0] = 0
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)

        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

# DA
class T_DA_Dataset(Dataset):
    def __init__(self, data_train_shuf, sdDA_arr, start, end,image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64
        self.start=start
        self.end = end
        self.sdDA_arr = sdDA_arr # 单类别sd

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        if self.bool_da[idx]: 
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            random_1 = np.random.randint(self.start, self.end)/1000
            SD = self.sdDA_arr[int(self.data_label[idx]), :288] # 提取出sd的值，去除类别列
            mat_img = mat_img + SD*random_1
            mat_img[mask==0] = 0
            if self.bool_da[idx]==1: # 逆时针旋转90度
                mat_img = np.rot90(mat_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
            elif self.bool_da[idx]==2:
                mat_img = mat_img[::-1,:,:].copy()# 左右=水平, 带-1索引的必须加.copy()
            elif self.bool_da[idx]==3:
                mat_img = mat_img[:,::-1,:].copy()# 上下=垂直
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label

class V_DA_Dataset(Dataset):
    def __init__(self, data_train_shuf, sdDA_arr, start, end,image_size=(64, 64)):
        self.data_path = data_train_shuf[:, 0]
        self.data_label = data_train_shuf[:, 1]
        self.bool_da = data_train_shuf[:, -1] # 是否增强
        self.image_size = image_size # 64
        self.start=start
        self.end = end
        self.sdDA_arr = sdDA_arr # 单类别sd

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        # 读取光谱
        hys_pyth = self.data_path[idx]
        load_mat = sio.loadmat(hys_pyth)
        mat_img = load_mat['image']
        if self.bool_da[idx]: 
            mask = np.bool_(mat_img[:, :, 174]) # start = -1100; end = 1100
            random_1 = np.random.randint(self.start, self.end)/1000
            SD = self.sdDA_arr[int(self.data_label[idx]), :288] # 提取出sd的值，去除类别列
            mat_img = mat_img + SD*random_1
            mat_img[mask==0] = 0
            if self.bool_da[idx]==1: # 逆时针旋转90度
                mat_img = np.rot90(mat_img, -1).copy() # 函数是逆时针旋转，-1变成顺时针
            elif self.bool_da[idx]==2:
                mat_img = mat_img[::-1,:,:].copy()# 左右=水平, 带-1索引的必须加.copy()
            elif self.bool_da[idx]==3:
                mat_img = mat_img[:,::-1,:].copy()# 上下=垂直
        # padding to 104*104
        row = np.round((self.image_size[0] - mat_img.shape[0])/2).astype(np.int16)
        col = np.round((self.image_size[1] - mat_img.shape[1])/2).astype(np.int16)
        resized_img = np.pad(mat_img, ((row, self.image_size[0]-row-mat_img.shape[0]), 
                            (col, self.image_size[1]-col-mat_img.shape[1]), (0, 0)), 
                            'constant', constant_values=0)
        image = resized_img.transpose((2, 0, 1))
        # 标签
        label = np.int64(self.data_label[idx])
        return image, label


def genDA_big_smal(data_train, ti=None, plus_xianyan=False, times=None):
    sdDA_ = []
    class_num = int(np.max(data_train[:, -1]) + 1)
    for label_i in range(class_num): # 遍历类别
        # 绘制子图
        # 1.分别计算每个梯度的类内光谱差
        # 1.1计算每类的平均光谱
        class_i = data_train[data_train[:, -1]==label_i] # t1
        # 1.2按1901nm处的波段排序, 第175个波段
        train_t1_sort = class_i[class_i[:,174].argsort()]
        # 1.2 计算类内sd, 不包括标签
        big_sample = train_t1_sort[ : train_t1_sort.shape[0]//2, :]
        small_sample = train_t1_sort[train_t1_sort.shape[0]//2 : , :]
        class_mean_t1_big = np.mean(big_sample, axis=0)
        class_mean_t1_smal = np.mean(small_sample, axis=0)

        sd_t1 = np.abs(class_mean_t1_big-class_mean_t1_smal)[:-1]
        sd_t1_label = np.append(sd_t1, label_i)
        sdDA_.append(sd_t1_label)
    sdDA_arr = np.array(sdDA_)
    sdDA_df = pd.DataFrame(sdDA_arr)
    sdDA_df.to_excel(excel_writer='./sdDA.xlsx', )
    return sdDA_arr


def choose_dataset(train_data, val_data, sdDA_arr, da_i, dax_i,t123):
    if da_i == 'None':
        trainDataset = T_or_Dataset(train_data)
        valDataset = V_or_Dataset(val_data)
    elif da_i == 'Erase':
        trainDataset = T_erase_Dataset(train_data)
        valDataset = V_erase_Dataset(val_data)
    elif da_i == 'noise':
        trainDataset = T_noise_Dataset(train_data)
        valDataset = V_noise_Dataset(val_data)
    elif da_i == 'rotate':
        if dax_i == 1:
            trainDataset = T_rotate2_Dataset(train_data)
            valDataset = V_rotate2_Dataset(val_data)
        else:
            trainDataset = T_rotate4_Dataset(train_data)
            valDataset = V_rotate4_Dataset(val_data)
    elif da_i == 'DA':
        if t123 == 't12':
            trainDataset = T_DA_Dataset(train_data, sdDA_arr, start=-200, end=800)
            valDataset = V_DA_Dataset(val_data, sdDA_arr, start=-200, end=800)
        else:
            trainDataset = T_DA_Dataset(train_data, sdDA_arr, start=-500, end=500)
            valDataset = V_DA_Dataset(val_data, sdDA_arr, start=-500, end=500)
    return trainDataset, valDataset


