# -*- coding: utf-8 -*-
import os
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from methods import load_data, load_ti_data, load_t123_data, choose_da, choose_eb, \
    trainDataset, valDataset, CNN1D, ONE_D_CNN


def cnn1d_test(test_spectral, test_label, net, weight_path):
    print('Start cnn1d test!')
    # 测试
    # net.load_state_dict(torch.load(weight_path))
    # print(net)
    net.eval()
    test_dataset = trainDataset(test_spectral, test_label)
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
    return result_all


def cnn1d_train(train_spectral, train_label, test_spectral, test_label, net, weight_path):
    print('Start cnn1d training!')
    # 训练
    train_spe, val_spe,train_lab,  val_lab = train_test_split(
            train_spectral, train_label, test_size=0.2, random_state=0)
    train_dataset = trainDataset(train_spe, train_lab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = valDataset(val_spe, val_lab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    total_episodes = len(train_loader) # epoch_num * len(train_loader)
    # 加载优化器、损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, ) # weight_decay=1e-5
    optimizer_scheduler = StepLR(optimizer, step_size=int(total_episodes), gamma=0.99)
    loss_function = nn.CrossEntropyLoss()
    os.makedirs('D:/research3/Train_model/model/log/', exist_ok=True)
    # runing_record = open('D:/research3/Train_model/model/log/'+info+'runing_record.txt', 'w')
    acc_tmp = 0
    for epoch in range(epoch_num):
        running_loss = 0.0
        accuracy = 0.0
        for batch_index, (spectrals, labels) in enumerate(train_loader):
            inputs = spectrals.to(device).type(torch.cuda.FloatTensor)
            # labels = labels.type(torch.cuda.LongTensor)
            labels = labels.to(device)
            net.train()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
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
                print_txt = '[{}, {}, {}] running loss = {:.6f} acc = {:.6f}'.format(
                            epoch + 1,epoch_num, total_episodes, running_loss / 20, accuracy)
                print(print_txt)
                # runing_record.write(print_txt+ '\n')
                running_loss = 0.0
        optimizer_scheduler.step()
        if epoch>=(epoch_num-5):
            y_pred = cnn1d_test(test_spectral, test_label, net, weight_path)
            Accuracy, F1, Kappa, Precision, Recall = gen_acc(method_i, da_i, pinzhong_i,y_pred, label_all, ti=t123)
            if Accuracy > acc_tmp:
                acc_tmp = Accuracy; F1_tmp=F1; Kappa_tmp=Kappa; Pre_tmp=Precision; Recall_tmp=Recall
                weight_path_out = weight_path + '_'+str(epoch+1)+'_'+str(np.round(Accuracy, decimals=6))+'.pth'
                print('save epoch:%s weight', epoch+1, method_i, da_i, pinzhong_i, t123)
                torch.save(net.state_dict(), weight_path_out)
            if (epoch+1)==epoch_num:
                weight_path_out = weight_path + '_'+str(epoch+1)+'_'+str(np.round(Accuracy, decimals=6))+'.pth'
                print('save epoch:%s weight', epoch+1, method_i, da_i, pinzhong_i, t123)
                torch.save(net.state_dict(), weight_path_out)
    return y_pred


def ONE_D_CNN_test(test_spectral, test_label, net, weight_path):
    print('Start ONE_D_CNN test!')
    # 测试
    net.load_state_dict(torch.load(weight_path))
    # print(net)
    net.eval()
    test_dataset = trainDataset(test_spectral, test_label)
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
    return result_all


def ONE_D_CNN_train(train_spectral, train_label, net, weight_path):
    print('Start ONE_D_CNN training!')
    # 训练
    train_spe, val_spe,train_lab,  val_lab = train_test_split(
            train_spectral, train_label, test_size=0.2, random_state=0)
    train_dataset = trainDataset(train_spe, train_lab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = valDataset(val_spe, val_lab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    total_episodes = len(train_loader) # epoch_num * len(train_loader)
    # 加载优化器、损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, )# weight_decay=1e-5
    optimizer_scheduler = StepLR(optimizer, step_size=int(total_episodes), gamma=0.99)
    # loss_function = torch.nn.functional.cross_entropy()
    os.makedirs('D:/research3/Train_model/model/log/', exist_ok=True)
    # writer = tensorboardX.SummaryWriter('D:/research3/Train_model/model/log/')
    for epoch in range(epoch_num):
        running_loss = 0.0
        accuracy = 0.0
        for batch_index, (spectrals, labels) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_index + 1
            inputs = spectrals.to(device).type(torch.cuda.FloatTensor)
            # labels = labels.type(torch.cuda.LongTensor)
            labels = labels.to(device)
            net.train()
            optimizer.zero_grad()
            outputs = net(inputs) # shape=N, 1, 288
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('loss', loss.data.cpu().numpy(), global_step=step)
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
                # writer.add_scalar('accuracy', accuracy, global_step=step)
                print_txt = '[{}, {}] running loss = {:.6f} acc = {:.6f}'.format(epoch + 1,epoch_num, running_loss / 20, accuracy)
                print(print_txt)
                running_loss = 0.0
        optimizer_scheduler.step()
    print('\n train finish! save model weight')
    torch.save(net.state_dict(), weight_path)
    return


def svm_method(img_train, gt_train, img_test,pinzhong_i):
    if pinzhong_i == 'all_pinzhong':
        # # 确定svm的最佳超参数
        # params = [{'C': np.logspace(0 ,9 ,11 ,base =1.8), 'gamma':np.logspace(0 ,9 ,11 ,base =1.8),}]
        # model = GridSearchCV(SVC(), params, n_jobs=4, return_train_score=True,)
        # model.fit(img_train, gt_train)
        # svm_func = model.best_estimator_
        # print('best_estimator_:',model.best_estimator_, )     # GridSearchCV的属性
        svm_func = SVC(C=198, gamma=1, kernel='rbf').fit(img_train, gt_train)
    else:
        if pinzhong_i == 'black-':
            C = 198; gamma = 40
        elif pinzhong_i == 'dbs-':
            C = 116; gamma = 14
        elif pinzhong_i == 'qc-':
            C = 116; gamma = 14
        elif pinzhong_i == 'slh-':
            C = 40; gamma = 68
        elif pinzhong_i == 'xbs-':
            C = 68; gamma = 68
        svm_func = SVC(C=C, gamma=gamma, kernel='rbf').fit(img_train, gt_train)
    y_predict = svm_func.predict(img_test)
    return y_predict

def knn_method(img_train, gt_train, img_test):
    # 确定svm的最佳超参数2,3,4,5,6,7,8,9,10
    params = [{'n_neighbors': [5,10,15,20]}]
    model = GridSearchCV(KNN(), params, n_jobs=4, return_train_score=True,)
    model.fit(img_train, gt_train)
    knn_func = model.best_estimator_
    print('best_estimator_:',model.best_estimator_, )     # GridSearchCV的属性
    # if pinzhong_i == 'all_pinzhong':
    #     raise ValueError
    #     knn_func = KNN(n_neighbors=5).fit(img_train, gt_train)
    # else:
    #     if pinzhong_i == 'black-':
    #         knn_func = KNN(n_neighbors=3).fit(img_train, gt_train)
    #     elif pinzhong_i == 'dbs-':
    #         knn_func = KNN(n_neighbors=3).fit(img_train, gt_train)
    #     elif pinzhong_i == 'qc-':
    #         knn_func = KNN(n_neighbors=3).fit(img_train, gt_train)
    #     elif pinzhong_i == 'slh-':
    #         knn_func = KNN(n_neighbors=4).fit(img_train, gt_train)
    #     elif pinzhong_i == 'xbs-':
    #         knn_func = KNN(n_neighbors=1).fit(img_train, gt_train)
    # knn_func = KNN(n_neighbors=5).fit(img_train, gt_train)
    y_predict = knn_func.predict(img_test)
    return y_predict

def dt_method(img_train, gt_train,img_test):
    dt_func = DT(criterion="entropy").fit(img_train, gt_train)
    y_predict = dt_func.predict(img_test)
    return y_predict

def plsda_test(img_train, train_label,img_test):
    # 所有标签转为独热编码
    train_label_onehot = pd.get_dummies(train_label)
    # test_label_onehot = pd.get_dummies(test_label)
    if img_train.shape[1] == 288:
        # # 确定plsda的最佳超参数np.arange左闭右开
        # params = [{'n_components': list(np.arange(1,img_train.shape[1]+1)),}]
        # model = GridSearchCV(PLSRegression(max_iter=2000), params, return_train_score=True,)
        # model.fit(img_train, train_label_onehot)
        # plsda_func = model.best_estimator_
        # print('best_estimator_:',model.best_estimator_, )     # GridSearchCV的属性
        plsda_func = PLSRegression(n_components=78).fit(img_train, train_label_onehot) # 2k个样本全波段选出78个成分
    else:
        plsda_func = PLSRegression(n_components=img_train.shape[1]).fit(img_train, train_label_onehot)
    y_predict = plsda_func.predict(img_test)
    # 评估
    y_pred = np.array([np.argmax(i) for i in y_predict])
    return y_pred

def plsda_test2(img_train, gt_train,img_test,):
    # for i in range(1, img_train.shape[1]+1):
    model = None
    # model = PLSRegression(n_components=i)
    model = PLSRegression(n_components=75)
    model.fit(img_train, gt_train)
    y_predict = model.predict(img_test)
    # 评估
    y_pred = np.array([np.argmax(i) for i in y_predict])
    return y_pred

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
    out2txt = [seed_i,methods_i+'+'+da_method_i+'+'+pinzhong_i,'梯度:',ti,  
                'ACC:', Accuracy,'Pre:', Precision, 'Recall:',Recall, 'F1:',F1, 'Kappa:',Kappa, 
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ]
    print(out2txt,'\n')     # 种子，分类方法，da方法，品种，训练数据，特征波段，da次数
    # txt_path = 'D:/research3/Train_model/'+ methods[0] + da_method_i +pinzhong_i+'_T'+ str(ti)+'T123_' \
    #             +str(eb)+'_dax'+str(da_times+1)+'.txt'
    # with open(txt_path,"a") as f:
    #     f.write(str(out2txt)+ '\n')
    # # print('plsda128结束时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return Accuracy, F1, Kappa, Precision, Recall

def choose_method(train_spectral, train_label, test_spectral,test_label, method,weight_path,pinzhong_i, c_dim):
    if method == 'plsda':
        y_pred = plsda_test(train_spectral, train_label, test_spectral,)
    elif method == 'svm':
        y_pred = svm_method(train_spectral, train_label, test_spectral,pinzhong_i)
    elif method == 'dt':
        y_pred = dt_method(train_spectral, train_label, test_spectral)
    elif method == 'knn':
        y_pred = knn_method(train_spectral, train_label, test_spectral)
    elif method == '1dcnn':
        num_bands = train_spectral.shape[1] # 提取波段数量
        net = CNN1D(in_dim=1, num_bands=num_bands, out_classes=c_dim).to(device)
        y_pred = cnn1d_train(train_spectral, train_label,test_spectral, test_label, net,weight_path)
        # y_pred = cnn1d_test(test_spectral, test_label, net, weight_path)
    elif method == 'ONE_D_CNN':
        net = ONE_D_CNN(x_dim=1, c_dim=c_dim,).to(device)
        if not os.path.exists(weight_path):
            ONE_D_CNN_train(train_spectral, train_label, net,weight_path)
            y_pred = ONE_D_CNN_test(test_spectral, test_label, net, weight_path)
        else:
            print('检查点存在, 删除'); os.remove(weight_path)
            ONE_D_CNN_train(train_spectral, train_label, net,weight_path)
            y_pred = ONE_D_CNN_test(test_spectral, test_label, net, weight_path)
    elif method == 'shufflenetv2':
        net = ONE_D_CNN(x_dim=1, c_dim=c_dim,).to(device)
        if not os.path.exists(weight_path):
            ONE_D_CNN_train(train_spectral, train_label, net,weight_path)
            y_pred = ONE_D_CNN_test(test_spectral, test_label, net, weight_path)
        else:
            print('检查点存在, 删除'); os.remove(weight_path)
            ONE_D_CNN_train(train_spectral, train_label, net,weight_path)
            y_pred = ONE_D_CNN_test(test_spectral, test_label, net, weight_path)
    return y_pred


def gen_weight_path(seed_i, methods_i, da_method_i, pinzhong_i, ti, dax_i):
    if da_method_i == 'ONE_D_CNN':
        info = str(seed_i)+methods_i+ da_method_i+pinzhong_i+'t'+str(ti)+ '_dax'+str(dax_i)
        os.makedirs('D:/research3/Train_model/model/'+ methods_i +'/'+info +'/', exist_ok=True)
        weight_path = 'D:/research3/Train_model/model/'+ methods_i +'/'+info +'/'+ info
    else:
        weight_path=None
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
# 'plsda', 'svm', 'knn', ONE_D_CNN, 1dcnn
methods = ['knn', ] # 'svm', knn
epoch_num = 50
# 3000,3001,3002,3003,3004,3005,3006,3007,3008,3009, 1005, 1006, 1007,1008,1009,
seed_all = [1000,1001, 1002, 1003, 1004,] 
train_all_data, test_all_data = load_data()
da_method = ['None',] # 'Erase', 'noise', 'TPW','DA'
pinzhong_tmp = ['black-', 'dbs-','qc-', 'slh-', 'xbs-']
pinzhong = ['black-',  'dbs-','qc-', 'slh-', 'xbs-',] # black-, 'dbs-','qc-', 'slh-', 'xbs-','all_pinzhong'
t123 = 't123'
for seed_i in seed_all: # 遍历种子
    np.random.seed(seed_i)
    svm_acc, svm_F1, svm_kap, svm_pre, svm_rec = [], [], [], [], [], 
    for method_i in methods:  # 分类方法
        dax2_acc, dax2_F1, dax2_kap, dax2_pre, dax2_rec = [], [], [], [], [], 
        for dax_i in [1, 3]:  # da次数
            da_ALL_acc, da_ALL_F1, da_ALL_kap, da_ALL_pre, da_ALL_rec = [], [], [], [], [], 
            for da_i in da_method:  # da方法
                Accuracy_all, F1_all, Kappa_all, Precision_all, Recall_all = [], [], [], [], [], 
                for idx in range(len(pinzhong)): # 遍历品种
                    pinzhong_i = pinzhong[idx]
                    # 梯度12
                    if pinzhong_i == 'all_pinzhong':
                        c_dim = 20; batch_size = 128
                        data_train_shuf = shuffle(train_all_data.iloc[:, 1:], random_state=0)
                        data_test_shuf = shuffle(test_all_data.iloc[:, 1:], random_state=0)
                        data_train_shuf = np.array(data_train_shuf)
                        data_test_shuf = np.array(data_test_shuf)
                    elif pinzhong_i in pinzhong_tmp:
                        c_dim = 4; batch_size = 128
                        data_train_arrt1, data_test_arrt1 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=1)
                        data_train_arrt2, data_test_arrt2 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=2)
                        data_train_arrt3, data_test_arrt3 = load_ti_data(train_all_data, test_all_data, pinzhong_i, idx, ti=3)
                        if t123 == 't123':
                            train_data12 = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
                            train_data123 = np.concatenate((train_data12, data_train_arrt3), axis=0)
                        else:
                            train_data12 = np.concatenate((data_train_arrt1, data_train_arrt2), axis=0)
                            train_data123 = train_data12
                        test_data = data_test_arrt1
                        # 改标签
                        train_data123[:, -1] = train_data123[:, -1] - idx*4  # 标签改为0-3
                        test_data[:, -1] = test_data[:, -1] - idx*4  # 标签改为0-3
                        data_train_shuf = shuffle(train_data123, random_state=seed_i-1000)
                        data_test_shuf = shuffle(test_data, random_state=seed_i-1000)
                    
                        # 选取20%的数据训练 80002000
                        # data_num =  data_train_shuf.shape[0]  # 2000  
                        data_train_ = data_train_shuf[:train_data12.shape[0], :] # [: data_num, :]
                        # DA
                        data_train_DA = choose_da(seed_i, pinzhong_i, data_train_, eb, ti='T123', da_method=da_i, times=dax_i)
                        data_train_ti = shuffle(data_train_DA, random_state=seed_i)
                        data_test_ti = shuffle(data_test_shuf, random_state=seed_i)
                        data_train_ti = choose_eb(eb, data_train_ti)
                        data_test_ti = choose_eb(eb, data_test_ti)
                        print(data_train_ti.shape, data_test_ti.shape, method_i, da_i, pinzhong_i, seed_i, t123 +'_dax'+ str(dax_i))
                        train_spectral = data_train_ti[:, :-1]
                        train_label = np.int64(data_train_ti[:, -1])
                        test_spectral = data_test_ti[:, :-1]
                        test_label = np.int64(data_test_ti[:, -1])
                        weight_path = gen_weight_path(seed_i, method_i, da_i, pinzhong_i, ti=12, dax_i = dax_i)
                        y_pred = choose_method(train_spectral, train_label, test_spectral, test_label, method_i,
                                                weight_path, pinzhong_i, c_dim)
                        Accuracy, F1, Kappa, Precision, Recall = gen_acc(method_i, da_i, pinzhong_i,y_pred, test_label, ti=t123)
                        # 5+1 个品种的结果，ACCF1kappa
                        Accuracy_all.append(Accuracy); F1_all.append(F1); Kappa_all.append(Kappa); 
                        Precision_all.append(Precision); Recall_all.append(Recall)
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
            # #暂停2分钟
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
    os.makedirs('./result/KNN_result2/', exist_ok=True)
    out_excel = ('./result/KNN_result2/' +method_i+'_' +str(seed_i)+ '_' + str(t123) + '_statistic.xlsx')
    if os.path.exists(out_excel):
        print('文件存在! 删除', out_excel)
        os.remove(out_excel)
    col_lie = ['黑花生','大白沙','七彩','四粒红','小白沙','多品种']
    index_hang = pd.Series(['1_None_x2', '2_Erase_x2', '3_Noise_x2', '4_TPW_x2', '5_DSM_x2', 
            '6_None_x4', '7_Erase_x4', '8_Noise_x4', '9_TPW_x4', '10_DSM_x4', ])
    writer = pd.ExcelWriter(out_excel,engine='openpyxl')# pylint: disable=abstract-class-instantiated
    svm_acc_df = pd.DataFrame(svm_acc_arr, columns = col_lie[:len(pinzhong)], index = index_hang)
    svm_F1_df = pd.DataFrame(svm_F1_arr, columns = col_lie[:len(pinzhong)], index = index_hang)
    svm_kap_df = pd.DataFrame(svm_kap_arr, columns = col_lie[:len(pinzhong)], index = index_hang)
    svm_pre_df = pd.DataFrame(svm_pre_arr, columns = col_lie[:len(pinzhong)], index = index_hang)
    svm_rec_df = pd.DataFrame(svm_rec_arr, columns = col_lie[:len(pinzhong)], index = index_hang)
    svm_acc_df.to_excel(writer, sheet_name='Accuracy',)
    svm_F1_df.to_excel(writer, sheet_name='F1',)
    svm_kap_df.to_excel(writer, sheet_name='Kappa',)
    svm_pre_df.to_excel(writer, sheet_name='Precision',)
    svm_rec_df.to_excel(writer, sheet_name='Recall',)
    writer.close()

