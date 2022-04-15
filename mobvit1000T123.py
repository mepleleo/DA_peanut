# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np 
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import accuracy_score, precision_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorboardX
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from methods import load_data, load_ti_data, load_t123_data, choose_da_img, choose_eb, \
    CNN1D, ONE_D_CNN, choose_dataset, T_or_Dataset
from mobile_vit import mobilevit_xs


def cal_mean(Accuracy, F1, Kappa, Precision, Recall):
    Accuracy = np.sort(np.array(Accuracy))#[1:-1]
    F1 = np.sort(np.array(F1))#[1:-1]
    Kappa = np.sort(np.array(Kappa))#[1:-1]
    Precision = np.sort(np.array(Precision))#[1:-1]
    Recall = np.sort(np.array(Recall))#[1:-1]
    # 计算均值
    Accuracy_mean = np.round(np.sum(Accuracy)/(Accuracy.shape[0]), decimals=4)
    F1_mean = np.round(np.sum(F1)/(F1.shape[0]), decimals=4)
    Kappa_mean = np.round(np.sum(Kappa)/(Kappa.shape[0]), decimals=4)
    Precision_mean = np.round(np.sum(Precision)/(Precision.shape[0]), decimals=4)
    Recall_mean = np.round(np.sum(Recall)/(Recall.shape[0]), decimals=4)
    return Accuracy_mean, F1_mean, Kappa_mean, Precision_mean, Recall_mean

def mob_vit_test(data_test_shuf, net, ):
    # print('Start mob_vit_train test!')
    # 测试
    # net.load_state_dict(torch.load(weight_path))
    # print(net)
    net.eval()
    test_dataset = T_or_Dataset(data_test_shuf)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=0, 
                            pin_memory=True,drop_last=False)
    result_all = np.array([])
    label_all = np.array([])
    with torch.no_grad():
        for n_iter, (spe_test, label) in enumerate(test_loader):
            spe_test = spe_test.to(device).type(torch.cuda.FloatTensor)
            label = label.to(device)
            output = net(spe_test)
            result = torch.max(output, -1)[1]# 返回最大值的索引
            result_batch = result.cpu().numpy()
            result_all = np.append(result_all,result_batch)
            label_batch = label.cpu().numpy()
            label_all = np.append(label_all,label_batch)
            # print('ss')
    label_all = label_all.astype(np.int8)
    result_all = result_all.astype(np.int8)
    return result_all, label_all


def mob_vit_train(data_train_shuf, data_test_shuf, SD_pinzhong, net, weight_path, da_i, ):
    start = weight_path.rindex('/'); weight_dir = weight_path[:start+1]
    weight_list = os.listdir(weight_dir)
    if len(weight_list)==0:
        start_ep = 0
    else:
        weight_list.sort(key=lambda x:os.path.getmtime((weight_dir+"\\"+x)))
        newest_file = os.path.join(weight_dir, weight_list[-1])
        print('newst file:',newest_file) # 绝对路径
        net.load_state_dict(torch.load(newest_file))
        start_ep = int(newest_file.split('_')[-2])
        global lr
        lr = lr*(0.2**(start_ep//20))
        # acc_tmp = float((newest_file.split('_')[-1]).split('.')[0])
        # y_pred, label_all = mob_vit_test(data_test_shuf, net)
        # acc_tmp, F1_tmp, Kappa_tmp, Pre_tmp, Recall_tmp = gen_acc(method_i, da_i, pinzhong_i,y_pred, label_all, ti=t123)
    print('Start training!', weight_path)
    # 训练
    train_data = data_train_shuf[:int(data_train_shuf.shape[0]*0.8), :]
    val_data = data_train_shuf[int(data_train_shuf.shape[0]*0.8):, :]
    train_dataset, val_dataset = choose_dataset(train_data, val_data,SD_pinzhong, da_i, dax_i, t123)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    total_episodes = len(train_loader) # epoch_num * len(train_loader)
    # 加载优化器、损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, ) # weight_decay=1e-5, weight_decay=5e-4
    optimizer_scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40,], gamma=0.2)
    os.makedirs('D:/research3/Train_model/model/mobile_vit_xs_log/'+weight_path.split('/')[-1], exist_ok=True)
    writer = tensorboardX.SummaryWriter('D:/research3/Train_model/model/mobile_vit_xs_log/'+weight_path.split('/')[-1])
    loss_function = nn.CrossEntropyLoss()
    Accuracy_list, F1_list, Kappa_list, Precision_list, Recall_list = [], [], [], [], []
    for epoch in range(start_ep, epoch_num):
        running_loss = 0.0
        accuracy = 0.0
        for batch_index, (spectrals, labels) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_index + 1
            inputs = spectrals.to(device).type(torch.cuda.FloatTensor)
            # labels = labels.type(torch.cuda.LongTensor)
            labels = labels.to(device)
            net.train()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss', loss.item(), global_step=step)
            # print(epoch, batch_index, loss.item())
            # 统计数据,loss,accuracy
            running_loss += loss.item()
            if batch_index % (total_episodes) == 0: # 每xx个batch验证一次
                correct = 0
                total = 0
                net.eval()
                for inputs, labels in val_loader:
                    inputs = inputs.to(device).type(torch.cuda.FloatTensor)
                    # labels = labels.type(torch.cuda.LongTensor)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    _, prediction = torch.max(outputs, 1)
                    correct += ((prediction == labels).sum()).item()
                    total += labels.size(0)
                accuracy = correct / total
                print_txt = '[{}-{}; {}] running loss = {:.6f} acc = {:.6f}'.format(
                            epoch + 1,epoch_num, total_episodes, running_loss / 20, accuracy,)
                print(print_txt, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),)
                writer.add_scalar('Train Accuracy', accuracy, global_step=step)
                running_loss = 0.0
        optimizer_scheduler.step()
        if epoch>=(epoch_num-10):
            y_pred, label_all = mob_vit_test(data_test_shuf, net, )
            Accuracy_Test, F1, Kappa, Precision, Recall = gen_acc(method_i, da_i, pinzhong_i,y_pred, label_all, ti=t123)
            Accuracy_list.append(Accuracy_Test); F1_list.append(F1); Kappa_list.append(Kappa); 
            Precision_list.append(Precision); Recall_list.append(Recall)
            writer.add_scalar('Test Accuracy', Accuracy_Test, global_step=step)
            writer.add_scalar('Test F1', F1, global_step=step)
            writer.add_scalar('Test Kappa', Kappa, global_step=step)
            writer.add_scalar('Test Precision', Precision, global_step=step)
        if (epoch==(epoch_num-10)) or (epoch==(epoch_num-1)) :
            weight_path_out = weight_path + '_'+str(epoch+1)+'_'+str(np.round(Accuracy_Test, decimals=6))+'.pth'
            print('save :%s weight', epoch+1, method_i, da_i, pinzhong_i, t123)
            torch.save(net.state_dict(), weight_path_out)
    print('Training finish!')
    Accuracy_mean, F1_mean, Kappa_mean, Precision_mean, Recall_mean = cal_mean(Accuracy_list, 
                    F1_list, Kappa_list, Precision_list, Recall_list)
    writer.close()
    return Accuracy_mean, F1_mean, Kappa_mean, Precision_mean, Recall_mean


def gen_acc(methods_i, da_method_i, pinzhong_i, y_pred, y_label, ti=123):
    # 预测和标签都是0123...
    # y_label = np.argmax(np.array(gt_test), axis=1)
    # # 混淆矩阵
    # CM = confusion_matrix(y_label, y_pred)
    # print('Confusion_matrix',CM)
    # gen_jingdu(y_label, y_pred)
    Accuracy = np.round(accuracy_score(y_label, y_pred), decimals=4)
    Precision = np.round(precision_score(y_label, y_pred,  average='weighted'), decimals=4)
    Recall = np.round(recall_score(y_label, y_pred,  average='weighted'), decimals=4)
    F1 = np.round(f1_score(y_label, y_pred, average='macro'), decimals=4)  # sklearn计算的是每个类别F1的均值
    # plsda_f1 = np.round(calculate_f1(plsda_pre,plsda_recall), decimals=4) # 自定义代码用所有类别的pre和recall计算F1
    Kappa = np.round(cohen_kappa_score(y_label, y_pred,), decimals=4)
    out2txt = [seed_i,methods_i+'+'+da_method_i+'+'+pinzhong_i, ti,  
                'ACC-->', Accuracy,'<--Pre:', Precision, 'Recall:',Recall, 'F1:',F1, 'Kappa:',Kappa, 
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ]
    print(out2txt,) # '\n'    # 种子，分类方法，da方法，品种，训练数据，特征波段，da次数
    return Accuracy, F1, Kappa, Precision, Recall


def choose_method(data_train_shuf, data_test_shuf,SD_pinzhong, method_i, da_i, weight_path, pinzhong_i, c_dim):
    if method_i == 'mobile_vit_xs':
        # num_bands = 288 # train_spectral.shape[1] # 提取波段数量
        net = mobilevit_xs(img_size, num_classes=c_dim).to(device)
        Acc, F1, Kappa, Pre, Recall = mob_vit_train(data_train_shuf, data_test_shuf,SD_pinzhong, net, weight_path,da_i, )
    return Acc, F1, Kappa, Pre, Recall


def gen_weight_path(seed_i, methods_i, da_method_i, pinzhong_i, ti, dax_i):
    info = str(seed_i)+methods_i+ da_method_i+pinzhong_i+'t'+str(ti)+ '_dax'+str(dax_i)
    os.makedirs('D:/research3/Train_model/model/'+ methods_i +'/'+info +'/', exist_ok=True)
    weight_path = 'D:/research3/Train_model/model/'+ methods_i +'/'+info +'/'+ info
    return weight_path


def all_data_train12(da_method, methods, pinzhong, train_all_data, test_all_data, eb, da_times,seed_i):
    # 多品种, t123训练t123测试
    for da_method_i in da_method: # da方法
        for methods_i in methods: # 分类方法
            T12_all_data = np.array([])
            for idx in range(len(pinzhong)): # 5个品种
                pinzhong_i = pinzhong[idx]
                data_train_arrt1, data_test_arrt1 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=1)
                data_train_arrt2, data_test_arrt2 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=2)
                pinzhong_t12 = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
                if np.any(T12_all_data):
                    T12_all_data = np.append(T12_all_data, pinzhong_t12, axis=0)
                else:
                    T12_all_data = pinzhong_t12
            
            data_train_shuf = shuffle(T12_all_data, random_state=seed_i)
            data_test_shuf = shuffle(test_all_data.iloc[:, 1:], random_state=0)
            data_train_shuf = np.array(data_train_shuf)
            data_test_shuf = np.array(data_test_shuf)
            # 选取20%的数据训练 80002000
            data_num =  data_train_shuf.shape[0]  # 2000  
            data_train_ = data_train_shuf[: data_num, :]
            # DA
            data_train_DA = choose_da(seed_i, 'all_pinzhong', data_train_, eb, 
                                        ti='T12', da_method=da_method_i, times=da_times)
            # data_train_DA = choose_da(pinzhong_i='all_pinzhong', data_train=data_train_, eb=eb, ti=None, 
            #                             da_method=da_method_i, times=da_times)
            data_train_ti = shuffle(data_train_DA, random_state=0)
            data_test_ti = shuffle(data_test_shuf, random_state=0)
            data_train_ti = choose_eb(eb, data_train_ti)
            data_test_ti = choose_eb(eb, data_test_ti)
            print('train.shape', data_train_ti.shape, 'test.shape', data_test_ti.shape, 
                    da_method_i, 'all_data_train12', methods_i, 't12', data_num)
            train_spectral = data_train_ti[:, :-1]
            train_label = np.int64(data_train_ti[:, -1])
            test_spectral = data_test_ti[:, :-1]
            test_label = np.int64(data_test_ti[:, -1])
            weight_path = gen_weight_path(methods_i, da_method_i, pinzhong_i='all_pinzhong', ti=12)
            y_pred = choose_method(train_spectral, train_label, test_spectral, test_label, methods_i,
                                    weight_path=weight_path, pinzhong_i='all_pinzhong')
            ACC = gen_acc(methods_i, da_method_i, 'all_data_train12',y_pred, test_label, ti=12)
    return

def all_data_train123(da_method, methods, pinzhong, train_all_data, test_all_data, eb, da_times,seed_i):
    # 多品种, t123训练t123测试
    for da_method_i in da_method: # da方法
        for methods_i in methods: # 分类方法
            data_train_shuf = shuffle(train_all_data.iloc[:, 1:], random_state=seed_i)
            data_test_shuf = shuffle(test_all_data.iloc[:, 1:], random_state=0)
            data_train_shuf = np.array(data_train_shuf)
            data_test_shuf = np.array(data_test_shuf)
            # 选取20%的数据训练 80002000
            data_num =  data_train_shuf.shape[0]  # 2000  
            data_train_ = data_train_shuf[: data_num, :]
            # DA
            data_train_DA = choose_da(seed_i, 'all_pinzhong', data_train_, eb, 
                                        ti='T123', da_method=da_method_i, times=da_times)
            data_train_DA = choose_da(pinzhong_i='all_pinzhong', data_train=data_train_, eb=eb, ti=None, 
                                        da_method=da_method_i, times=da_times)
            data_train_ti = shuffle(data_train_DA, random_state=0)
            data_test_ti = shuffle(data_test_shuf, random_state=0)
            data_train_ti = choose_eb(eb, data_train_ti)
            data_test_ti = choose_eb(eb, data_test_ti)
            print('train.shape', data_train_ti.shape, 'test.shape', data_test_ti.shape, 
                    da_method_i, 'all_data_train123', methods_i, 't12', data_num)
            train_spectral = data_train_ti[:, :-1]
            train_label = np.int64(data_train_ti[:, -1])
            test_spectral = data_test_ti[:, :-1]
            test_label = np.int64(data_test_ti[:, -1])
            weight_path = gen_weight_path(methods_i, da_method_i, pinzhong_i='all_pinzhong', ti=123)
            y_pred = choose_method(train_spectral, train_label, test_spectral, test_label, methods_i,
                                    weight_path=weight_path, pinzhong_i='all_pinzhong')
            ACC = gen_acc(methods_i, da_method_i, 'all_data_train123',y_pred, test_label, ti=123)
    return



# if __name__ == '__main__':
eb = 288
if eb == 8:
    # eb = np.array([38, 72, 86, 126, 175, 195, 220, 249,])-1
    eb = np.array([1, 47, 65, 92, 179, 217, 249, 287]) # 多品种 dt2000样本 seed100
elif eb ==288:
    eb = np.array([])
elif eb ==10:
    eb = np.array([1, 25, 47, 71, 92, 118, 178, 217, 249, 286]) # 多品种 dt2000样本 seed34+50




print('程序运行时间: ',time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 'plsda', 'svm', 'knn', ONE_D_CNN, 1dcnn, mobile_vit_xs
methods = ['mobile_vit_xs', ]
lr = 0.001; batch_sizes=[0, 128, 0, 128]
epoch_num = 50; 
img_size = 64
seed_all = [2000, 2001, 2002, 2003, 2004, 2005, ] # 1010,1011, 2006, 2007,2008,2009, 
train_all_data, test_all_data = load_data()
da_method = ['DA',  'rotate','Erase',] #'None', 'noise',  
pinzhong_tmp = ['black-', 'dbs-','qc-', 'slh-', 'xbs-']
pinzhong = ['black', 'dbs-','qc', 'slh', 'xbs'] # black, 'dbs','qc', 'slh', 'xbs','all_pinzhong'
t123 = 't123'
for seed_i in seed_all: # 遍历种子
    np.random.seed(seed_i)
    svm_acc, svm_F1, svm_kap, svm_pre, svm_rec = [], [], [], [], [], 
    for method_i in methods:  # 分类方法
        dax2_acc, dax2_F1, dax2_kap, dax2_pre, dax2_rec = [], [], [], [], [], 
        for dax_i in [1, ]:  # da次数 1,3,
            batch_size = batch_sizes[dax_i]
            da_ALL_acc, da_ALL_F1, da_ALL_kap, da_ALL_pre, da_ALL_rec = [], [], [], [], [], 
            for da_i in da_method:  # da方法
                Accuracy_all, F1_all, Kappa_all, Precision_all, Recall_all = [], [], [], [], [], 
                for idx in range(len(pinzhong)): # 遍历品种
                    pinzhong_i = pinzhong[idx]
                    # 梯度12
                    if pinzhong_i == 'all_pinzhong':
                        c_dim = 20; 
                        T12_all_data = np.array([])
                        for idx_a in range(5): # 5个品种
                            pinzhong_ia = pinzhong[idx_a]
                            data_train_arrt1, data_test_arrt1 = load_ti_data(train_all_data, test_all_data, 
                                    pinzhong_ia, idx_a, ti=1, train_spectral=False)# 去掉光谱提取路径
                            data_train_arrt2, data_test_arrt2 = load_ti_data(train_all_data, test_all_data, 
                                    pinzhong_ia, idx_a, ti=2, train_spectral=False)
                            pinzhong_t12 = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
                            if np.any(T12_all_data):
                                T12_all_data = np.append(T12_all_data, pinzhong_t12, axis=0)
                            else:
                                T12_all_data = pinzhong_t12
                        data_train_shuf = shuffle(T12_all_data, random_state=0)
                        data_test_shuf = shuffle(test_all_data.iloc[:, 1:], random_state=0)
                        data_train_shuf = np.array(data_train_shuf)
                        data_test_shuf = np.array(data_test_shuf)
                        # 读取sd
                        SD_all = np.array(pd.read_excel('G:/BNU_second_time_data/sdDA.xlsx'))
                        SD_pinzhong = SD_all
                    elif pinzhong_i in pinzhong_tmp:
                        c_dim = 4; 
                        data_train_arrt1, data_test_arrt1 = load_ti_data(train_all_data, test_all_data, 
                                    pinzhong_i, idx, ti=1, train_spectral=False) # 去掉光谱提取路径
                        data_train_arrt2, data_test_arrt2 = load_ti_data(train_all_data, test_all_data, 
                                    pinzhong_i, idx, ti=2, train_spectral=False)
                        data_train_arrt3, data_test_arrt3 = load_ti_data(train_all_data, test_all_data, 
                                    pinzhong_i, idx, ti=3, train_spectral=False)
                        train_data12 = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
                        train_data = np.concatenate((train_data12, data_train_arrt3), axis=0)
                        test_data = data_test_arrt1
                        # 改标签
                        train_data[:, -1] = train_data[:, -1] - idx*4  # 标签改为0-3
                        test_data[:, -1] = test_data[:, -1] - idx*4  # 标签改为0-3
                        data_train_shuf = shuffle(train_data, random_state=seed_i)
                        data_test_shuf = shuffle(test_data, random_state=seed_i)
                        # 读取sd
                        SD_all = np.array(pd.read_excel('G:/BNU_second_time_data/sdDA.xlsx'))
                        SD_pinzhong = SD_all[idx*4:idx*4+4, :]
                        SD_pinzhong[:, -1] = SD_pinzhong[:, -1] - idx*4  # 标签改为0-3
                    
                        # 选取20%的数据训练 80002000
                        # data_num =  data_train_shuf.shape[0]  # 2000  
                        if t123 == 't123':
                            data_train_ = data_train_shuf[:train_data12.shape[0], :] # [: data_num, :]
                        else:
                            data_train_ = data_train_shuf
                        # DA 路径
                        data_train_DA = choose_da_img(seed_i, pinzhong_i, data_train_, eb, ti=t123, da_method=da_i, times=dax_i)
                        data_train_ti = shuffle(data_train_DA, random_state=seed_i)
                        data_test_ti = shuffle(data_test_shuf, random_state=seed_i)
                        print(data_train_ti.shape, data_test_ti.shape)
                        weight_path = gen_weight_path(seed_i, method_i, da_i, pinzhong_i, ti=t123, dax_i = dax_i)
                        start = weight_path.rindex('/'); weight_dir = weight_path[:start+1]
                        weight_list = os.listdir(weight_dir)
                        weight_list.sort(key=lambda x:os.path.getmtime((weight_dir+"\\"+x)))
                        newest_file = os.path.join(weight_dir, weight_list[-1])
                        print('newst file:',newest_file) # 绝对路径
                        net = mobilevit_xs(img_size, num_classes=c_dim).to(device)
                        net.load_state_dict(torch.load(newest_file))
                        y_pred, label_all = mob_vit_test(data_test_shuf, net, )
                        Accuracy, F1, Kappa, Precision, Recall = gen_acc(method_i, da_i, pinzhong_i,y_pred, label_all, ti=t123)
                        # Accuracy, F1, Kappa, Precision, Recall = choose_method(data_train_ti, data_test_ti, SD_pinzhong,
                        #                         method_i, da_i, 
                        #                         weight_path, pinzhong_i, c_dim)
                        # 5+1 个品种的结果，ACCF1kappa
                        Accuracy_all.append(Accuracy); F1_all.append(F1); Kappa_all.append(Kappa); 
                        Precision_all.append(Precision); Recall_all.append(Recall)
                        #暂停2分钟
                        # time.sleep(20)
                    else:
                        Accuracy, F1, Kappa, Precision, Recall = 0,0,0,0,0
                        # 5+1 个品种的结果，ACCF1kappa
                        Accuracy_all.append(Accuracy); F1_all.append(F1); Kappa_all.append(Kappa); 
                        Precision_all.append(Precision); Recall_all.append(Recall)
                Accuracy_da_i = np.array(Accuracy_all).reshape((1, len(pinzhong)))
                F1_da_i = np.array(F1_all).reshape((1, len(pinzhong)))
                Kappa_da_i = np.array(Kappa_all).reshape((1, len(pinzhong)))
                Precision_da_i = np.array(Precision_all).reshape((1, len(pinzhong)))
                Recall_da_i = np.array(Recall_all).reshape((1, len(pinzhong)))
                # 遍历一遍增强方法
                da_ALL_acc.append(Accuracy_da_i)
                da_ALL_F1.append(F1_da_i)
                da_ALL_kap.append(Kappa_da_i)
                da_ALL_pre.append(Precision_da_i)
                da_ALL_rec.append(Recall_da_i)
            #暂停2分钟
            # time.sleep(60*2)
            da_ALL_acc_arr = np.array(da_ALL_acc)
            da_ALL_F1_arr = np.array(da_ALL_F1)
            da_ALL_kap_arr = np.array(da_ALL_kap)
            da_ALL_pre_arr = np.array(da_ALL_pre)
            da_ALL_rec_arr = np.array(da_ALL_rec)
            dax2_acc.append(da_ALL_acc_arr)
            dax2_F1.append(da_ALL_F1_arr)
            dax2_kap.append(da_ALL_kap_arr)
            dax2_pre.append(da_ALL_pre_arr)
            dax2_rec.append(da_ALL_rec_arr)
        dax2_acc_arr = np.array(dax2_acc)
        dax2_F1_arr = np.array(dax2_F1)
        dax2_kap_arr = np.array(dax2_kap)
        dax2_pre_arr = np.array(dax2_pre)
        dax2_rec_arr = np.array(dax2_rec)
        svm_acc.append(dax2_acc_arr)
        svm_F1.append(dax2_F1_arr)
        svm_kap.append(dax2_kap_arr)
        svm_pre.append(dax2_pre_arr)
        svm_rec.append(dax2_rec_arr)
    svm_acc_arr = np.array(svm_acc).reshape((-1, len(pinzhong)))
    svm_F1_arr = np.array(svm_F1).reshape((-1, len(pinzhong)))
    svm_kap_arr = np.array(svm_kap).reshape((-1, len(pinzhong)))
    svm_pre_arr = np.array(svm_pre).reshape((-1, len(pinzhong)))
    svm_rec_arr = np.array(svm_rec).reshape((-1, len(pinzhong)))
    os.makedirs('./result/oneD_cnn_result_dbs/', exist_ok=True)
    out_excel = ('./result/oneD_cnn_result_dbs/' +method_i+'_' +str(seed_i)+ '_' + str(t123) + '_statistic.xlsx')
    if os.path.exists(out_excel):
        print('文件存在! 删除', out_excel)
        os.remove(out_excel)
    col_lie = ['黑花生','大白沙','七彩','四粒红','小白沙','多品种']
    index_hang = pd.Series(['1_x2', '2_x2', '3_x2', '4_x2', '5_x2', 
            '6_x4', '7_x4', '8_x4', '9_x4', '10_x4', ])
    writer = pd.ExcelWriter(out_excel,engine='openpyxl')# pylint: disable=abstract-class-instantiated
    svm_acc_df = pd.DataFrame(svm_acc_arr, columns = col_lie[:len(pinzhong)], index = index_hang[:svm_acc_arr.shape[0]])
    svm_F1_df = pd.DataFrame(svm_F1_arr, columns = col_lie[:len(pinzhong)], index = index_hang[:svm_acc_arr.shape[0]])
    svm_kap_df = pd.DataFrame(svm_kap_arr, columns = col_lie[:len(pinzhong)], index = index_hang[:svm_acc_arr.shape[0]])
    svm_pre_df = pd.DataFrame(svm_pre_arr, columns = col_lie[:len(pinzhong)], index = index_hang[:svm_acc_arr.shape[0]])
    svm_rec_df = pd.DataFrame(svm_rec_arr, columns = col_lie[:len(pinzhong)], index = index_hang[:svm_acc_arr.shape[0]])
    svm_acc_df.to_excel(writer, sheet_name='Accuracy',)
    svm_F1_df.to_excel(writer, sheet_name='F1',)
    svm_kap_df.to_excel(writer, sheet_name='Kappa',)
    svm_pre_df.to_excel(writer, sheet_name='Precision',)
    svm_rec_df.to_excel(writer, sheet_name='Recall',)
    writer.close()

