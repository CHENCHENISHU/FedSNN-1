#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.nn as nn

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_non_iid, mnist_dvs_iid, mnist_dvs_non_iid, nmnist_iid, nmnist_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedLearn
from models.Fed import model_deviation
from models.test import test_img
import models.vgg as ann_models
import models.resnet as resnet_models
import models.vgg_spiking_bntt as snn_models_bntt

import tables
import yaml
import glob
import json

from PIL import Image

from pysnn.datasets import nmnist_train_test

if __name__ == '__main__':
    # parse args
    # 传入参数转换为args
    # python
    # main_fed.py - -snn - 用snn训练 -dataset CIFAR10 数据集为CIFAR10
    # - -num_classes 10 类别数
    # - -model VGG9 - 用VGG9训练 - 模型为SNN的VGG9
    # - -optimizer SGD - 用SGD优化器 - 优化器为SGD
    # - -bs 32 全局批量大小
    # - -local_bs 32 本地训练的批量大小
    # - -lr 0.1 学习率0.1
    # - -lr_reduce 5   学习率衰减的间隔周期，5epoch减少一次
    # - -epochs 100 全局训练的轮数
    # - -local_ep 2  客户端本地训练轮数为2
    # - -eval_every  1 每个全局轮评估一次模型
    # - -num_users   10  用户总数为10
    # - -frac 0.2 每轮参加训练的客户端比例
    # - -iid - 使用IID数据分布
    # -gpu 0 ：使用gpu0
    # - -timesteps 20 SNN的时间步长为20
    # - -result_dir test 结果保存的目录为test
    args = args_parser()
    # 设置PyTorch，numpy的随机数生成器的种子。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 这确保了整个联邦学习训练过程（包括服务器聚合和客户端本地训练）都在GPU0上进行
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.device != 'cpu': # 如果使用GPU（在您的命令中是cuda:0）
        torch.backends.cudnn.deterministic = True # 确保CUDA操作确定性 如果没有确定性设置，相同数据可能在不同运行中得到不同结果
        torch.backends.cudnn.benchmark = False # 防止CuDNN自动寻找最优卷积算法
    # torch.set_default_tensor_type('torch.cuda.FloatTensor') 这行代码会将PyTorch的默认张量类型设置为CUDA浮点张量，意味着之后创建的所有张量都会自动放在GPU上。

    dataset_keys = None  # 用于存储数据集键名
    h5fs = None # 用于存储HDF5文件对象
    # load dataset and split users
    if args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 将PIL图像转为Tensor，数值范围[0,1]，归一化到[-1，1]
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar) # 训练集：50,000张图像
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar) # 测试集：10,000张图像
        if args.iid:
            # cifar_iid 函数会将50,000个训练样本平均分给10个用户
            # 每个用户获得：50,000 / 10 = 5,000个样本
            # 每个用户的数据都包含所有10个类别的均匀分布
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == 'CIFAR100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == 'N-MNIST':
        dataset_train, dataset_test = nmnist_train_test("nmnist/data")
        if args.iid:
            dict_users = nmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = nmnist_non_iid(dataset_train, args.num_classes, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # 创建模型
    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg': # True，因为'VGG9'的前三个字符是'VGG'
        if args.snn: # 因为您使用了--snn参数
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps} # model_args = {'num_cls': 10, 'timesteps': 20}输入命令为这个
            # 创建并初始化SNN全局模型，并将其移动到GPU上
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).cuda()
    elif args.model[0:6].lower() == 'resnet':
        if args.snn:
            pass
        else:
            model_args = {'num_cls': args.num_classes}
            net_glob = resnet_models.Network(**model_args).cuda()
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # copy weights
    # 加载预训练模型权重到全局模型中如果有的话
    if args.pretrained_model:
        net_glob.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    # 将模型包装为数据并行模式，使其能够在多个GPU上并行训练。
    net_glob = nn.DataParallel(net_glob)
    # training
    loss_train_list = [] # 记录训练损失历史
    cv_loss, cv_acc = [], [] # 记录交叉验证损失和准确率
    val_loss_pre, counter = 0, 0 # 用于早停机制的变量
    net_best = None # 保存最佳模型状态
    best_loss = None # 记录最佳损失值
    val_acc_list, net_list = [], [] # 记录验证准确率和模型历史

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], [] # 记录训练损失历史
    ms_acc_test_list, ms_loss_test_list = [], [] # 记录交叉验证损失和准确率
    # 每轮参与的客户端数量 通信成本指标（模型聚合时计算通信成本）
    ms_num_client_list, ms_tot_comm_cost_list, ms_avg_comm_cost_list, ms_max_comm_cost_list = [], [], [], []
    # 梯度稀疏性指标 梯度向量中只有少数位置有非零值，那么我们就说这个梯度是稀疏的
    ms_tot_nz_grad_list, ms_avg_nz_grad_list, ms_max_nz_grad_list = [], [], []
    # 模型一致性指标 测量客户端模型与全局模型的差异
    ms_model_deviation = []

    # testing 模型评估阶段
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Initial Training accuracy: {:.2f}".format(acc_train))
    # print("Initial Testing accuracy: {:.2f}".format(acc_test))
    acc_train, loss_train = 0, 0 # 初始化为0
    acc_test, loss_test = 0, 0 # 初始化为0
    # Add metrics to store
    ms_acc_train_list.append(acc_train) # 记录训练准确率（当前为0）
    ms_acc_test_list.append(acc_test)  # 记录测试准确率（当前为0）
    ms_loss_train_list.append(loss_train) # 记录训练损失（当前为0）
    ms_loss_test_list.append(loss_test) # 记录测试损失（当前为0）

    # Define LR Schedule
    values = args.lr_interval.split() # 分割学习率调整间隔字符串
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs)) # 转换为实际轮次

    # Define Fed Learn object
    fl = FedLearn(args)

    for iter in range(args.epochs):
        net_glob.train() # 设置全局模型为训练模式
        w_locals_selected, loss_locals_selected = [], [] # 选中客户端的权重和损失
        w_locals_all, loss_locals_all = [], [] # 所有客户端的权重和损失
        # 选择参与本轮训练的客户端
        m = max(int(args.frac * args.num_users), 1) # 计算选择的客户端数量
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)
        # for idx in idxs_users:
        # Do local update in all the clients # Not required (local updates in only the selected clients is enough) for normal experiments but neeeded for model deviation analysis
        for idx in range(args.num_users): # 遍历所有10个客户端
            # 创建本地训练任务
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client

            model_copy = type(net_glob.module)(**model_args) #  创建新模型实例
            model_copy = nn.DataParallel(model_copy) # 数据并行包装
            model_copy.load_state_dict(net_glob.state_dict()) # 复制权重
            # 本地训练
            w, loss = local.train(net=model_copy.to(args.device))
            # 存储所有客户端的结果
            w_locals_all.append(copy.deepcopy(w))
            loss_locals_all.append(copy.deepcopy(loss))
            # 如果是选中客户端，额外存储用于聚合
            if idx in idxs_users:
                w_locals_selected.append(copy.deepcopy(w))
                loss_locals_selected.append(copy.deepcopy(loss))
        # 模型偏差分析
        model_dev_list = model_deviation(w_locals_all, net_glob.state_dict())
        ms_model_deviation.append(model_dev_list)
        # 全局模型更新
        w_glob = fl.FedAvg(w_locals_selected, w_init = net_glob.state_dict())
        
        w_init = net_glob.state_dict() # 保存聚合前的全局模型权重
        delta_w_locals_selected = [] # 存储选中客户端的权重变化量
        for i in range(0, len(w_locals_selected)):
            delta_w = {}
            for k in w_init.keys(): # 遍历所有参数键（如'conv1.weight', 'bn1.bias'等）
                delta_w[k] = w_locals_selected[i][k] - w_init[k] # 计算参数变化量
            delta_w_locals_selected.append(delta_w)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob) # 将联邦平均后的权重加载到全局模型
 
        # print loss
        # 本地损失记录和平均
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
        # 定期模型评估
        if iter % args.eval_every == 0: # 每1轮评估一次
            # testing
            net_glob.eval()  # 训练模式
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(iter, acc_train))
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))
 
            # Add metrics to store
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)

        if iter in lr_interval:
            args.lr = args.lr/args.lr_reduce

    Path('./{}'.format(args.result_dir)).mkdir(parents=True, exist_ok=True)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    plt.savefig('./{}/fed_loss_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir,args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    print("Final Training accuracy: {:.2f}".format(acc_train))
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Final Testing accuracy: {:.2f}".format(acc_test))

    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(ms_acc_train_list)), ms_acc_train_list)
    plt.plot(range(len(ms_acc_test_list)), ms_acc_test_list)
    plt.plot()
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Accuracy')
    plt.legend(['Training acc', 'Testing acc'])
    plt.savefig('./{}/fed_acc_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid))

    # Write metric store into a CSV
    metrics_df = pd.DataFrame(
        {
            'Train acc': ms_acc_train_list,
            'Test acc': ms_acc_test_list,
            'Train loss': ms_loss_train_list,
            'Test loss': ms_loss_test_list
        })
    metrics_df.to_csv('./{}/fed_stats_{}_{}_{}_C{}_iid{}.csv'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid), sep='\t')

    torch.save(net_glob.module.state_dict(), './{}/saved_model'.format(args.result_dir))

    fn = './{}/model_deviation_{}_{}_{}_C{}_iid{}.json'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid)
    with open(fn, 'w') as f:
        json.dump(ms_model_deviation, f)