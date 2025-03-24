# -*- coding: UTF-8 -*-
import torch
import numpy as np
import os
import xlrd
from PIL import Image
import copy
import os
import pickle
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import sklearn.metrics
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime
import torch
from torchvision import models
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils
import torch.autograd as autograd
import argparse



def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch PrimarySchool Fundus photography RA-Regression')
    # Params
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default= -1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=24, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='Mixed',
                        help='choose the optimizer function')
    
    # Dataset
    parser.add_argument('--data_root', type=str, default='../data/JPG256SIZE-95')
    parser.add_argument('--filename', type=str, default='std')
    parser.add_argument('--feature_list', type=int, nargs='+', default=[8, 9], help='')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--eye', type=str, default='R',
                        help='choose the right or left eye')
    parser.add_argument('--center', type=str, default='Maculae',
                        help='choose the center type pf the picture,input(Maculae)(Disc)(Maculae&Disc) ')
    
    # Model
    parser.add_argument('--load_dir', type=str, default='/home')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--works', type=int, default=2)
    
    # Visulization
    parser.add_argument('--is_plot', type=int, default=0)
    parser.add_argument('--plot_sigmoid', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=100)
    
    
    #parser.add_argument('--margin', type=float, default=0.5,
                        #help='set the error margin')
    
    parser.add_argument('--label', type=str, default='RA',
                        help='choose the label we want to do research')
    
    # Train set
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr2', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr3', type=float, default=0.001,
                        help='lr3 for ESPCN')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='')
    parser.add_argument('--lr_controler', type=int, default=30, )
    parser.add_argument('--wd', type=float, default=0.0001, help='')
    parser.add_argument('--wd2', type=float, default=0.0001, help='')
    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument('--is_ESPCN', type=int, default=0,
                        help='whether load model ESPCN')
    parser.add_argument('--dw_midch', type=int, default=1024,
                        help='choose the mid-channel type of ESPCN')
    parser.add_argument('--RNN_hidden', type=int, nargs='+', default=[256], help='')
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--dp_n', type=float, default=0.5)
    parser.add_argument('--wcl', type=float, default=0.01,
                        help='control the clamp parameter of Dnet, weight_cliping_limit')
    parser.add_argument('--isplus', type=int, default=0)
    parser.add_argument('--final_tanh', type=int, default=0)
    parser.add_argument('--train_ori_data', type=int, default=0)
    parser.add_argument('--pred_future', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1,
                        help='parameter of loss image')
    parser.add_argument('--beta', type=float, default=1,
                        help='parameter of fake loss RA prediction')
    parser.add_argument('--beta1', type=float, default=1,
                        help='parameter of fake loss RA prediction')
    parser.add_argument('--gamma', type=float, default=1,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--delta1', type=float, default=0.001,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--delta', type=float, default=0.001,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--sequence', type=str, default='5-separate-1',
                        help='save the check point of the model and optim')
    parser.add_argument('--D_epoch', type=int, default=4)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--xlsx_name', type=str, default='../xl_results/xlsx_1')

    #parser.add_argument('--is_endwith6', type=int, default=0,
                        #help='if make the dataset end with 6')
    # parser.add_argument('--dataset_type', type=str, default='baseline')
    # parser.add_argument('--load_model_1', type=str, default='nbase_2')
    # parser.add_argument('--load_model_pre', type=str, default='baseline')
    parser.add_argument('--G_net_type', type=str, default='G_net')
    # parser.add_argument('--dw_type', type=str, default='conv',
    # help='choose the downsample type of ESPCN')
    
    #parser.add_argument('--critic_iter', type=int, default=5,
                        #help='control the critic iterations for updating Gnet')

    parser.add_argument('--noise_size', type=int, default=516,
                        help='control the size of the input of Gnet')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='save the check point of the model and optim')
    
    #parser.add_argument('--betas1', type=float, default='0.5',
                        #help='hyper-parameter for Adam')
    #parser.add_argument('--betas2', type=float, default='0.9',
                        #help='hyper-parameter for Adam')
    #parser.add_argument('--LAMBDA', type=float, default='10',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--bi_linear', type=int, default=0,
                        #help='whether use the bilinear mode in Unet')
    # sijie add for debugging biggan
    #parser.add_argument('--gantype', type=str, default='Big',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--discritype', type=str, default='normal',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--SCR', type=int, default=0,
                        #help='hyper-parameter for gradient panelty')
    # sijie for generate test
    
    
    #parser.add_argument('--scale_factor', type=int, default=2,
                        #help='choose the mid-channel type of ESPCN')
    
    args = parser.parse_args()
    args.final_tanh = 0
    time_stamp = datetime.datetime.now()
    save_dir = '../results/Ecur_%s/%s_%s_%f_%f_size_%s_ep_%d_%d_%s_%s/' % (
        args.sequence,
        time_stamp.strftime(
            '%Y-%m-%d-%H-%M-%S'),
        args.optimizer,
        args.lr,
        args.lr_controler,
        args.image_size,
        args.epochs,
        args.wcl, args.eye,
        args.center)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = get_logger(args)
    
    return args

def load_pytorch_model(model, path):
    from collections import OrderedDict
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict,  strict=False)###parameter strict: do not load the part dont match
    return model

def get_logger(opt, fold=0):
    import logging
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training_{}.log'.format(opt.save_dir, fold))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def draw_features(width, height, feature, savepath, image_sequence, train_loader, a, batchsize, dpi):
    str = a
    batch_size = batchsize
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)


    num = image_sequence
    x = feature
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]
        plt.imshow(img)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = os.path.join(savepath, '%s' % (str) + os.path.basename(
                        train_loader.dataset.image_path_all_1[num]))
    fig.savefig(savename, dpi=dpi)
    fig.clf()
    plt.close()


def init_metric(args):
    args.best_test_acc = -1
    args.best_test_auc = -1
    args.best_val_acc = -1
    args.best_val_auc = -1

    args.best_test_acc_1 = -1
    args.best_test_auc_1 = -1
    args.best_val_acc_1 = -1
    args.best_val_auc_1 = -1
    args.best_val_auc_1_epoch = 0
    args.best_val_acc_1_epoch = 0

    args.best_test_acc_2 = -1
    args.best_test_auc_2 = -1
    args.best_val_acc_2 = -1
    args.best_val_auc_2 = -1
    args.best_val_auc_2_epoch = 0
    args.best_val_acc_2_epoch = 0

    args.best_test_acc_3 = -1
    args.best_test_auc_3 = -1
    args.best_val_acc_3 = -1
    args.best_val_auc_3 = -1
    args.best_val_auc_3_epoch = 0
    args.best_val_acc_3_epoch = 0

    args.best_test_acc_4 = -1
    args.best_test_auc_4 = -1
    args.best_val_acc_4 = -1
    args.best_val_auc_4 = -1
    args.best_val_auc_4_epoch = 0
    args.best_val_acc_4_epoch = 0

    args.best_test_acc_5 = -1
    args.best_test_auc_5 = -1
    args.best_val_acc_5 = -1
    args.best_val_auc_5 = -1
    args.best_val_auc_5_epoch = 0
    args.best_val_acc_5_epoch = 0

    args.best_test_acc_list = [args.best_test_acc_1, args.best_test_acc_2, args.best_test_acc_3, args.best_test_acc_4,
                               args.best_test_acc_5]
    args.best_test_auc_list = [args.best_test_auc_1, args.best_test_auc_2, args.best_test_auc_3, args.best_test_auc_4,
                               args.best_test_auc_5]

    args.best_val_acc_list = [args.best_val_acc_1, args.best_val_acc_2, args.best_val_acc_3, args.best_val_acc_4,
                               args.best_val_acc_5]
    args.best_val_auc_list = [args.best_val_auc_1, args.best_val_auc_2, args.best_val_auc_3, args.best_val_auc_4,
                               args.best_val_auc_5]

    args.best_val_auc_epoch_list = [args.best_val_auc_1_epoch, args.best_val_auc_2_epoch, args.best_val_auc_3_epoch,
                                      args.best_val_auc_4_epoch, args.best_val_auc_5_epoch]
    args.best_val_acc_epoch_list = [args.best_val_acc_1_epoch, args.best_val_acc_2_epoch, args.best_val_acc_3_epoch,
                                    args.best_val_acc_4_epoch, args.best_val_acc_5_epoch]
    return args







def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', base_dir='./'):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename), os.path.join(base_dir, 'model_best.pth.tar'))



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_controler))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import xlwt
import xlrd
from xlutils.copy import copy as x_copy


def save_results_as_xlsx(root, args=None):
    if not os.path.exists(os.path.dirname(args.xlsx_name)):
        os.makedirs(os.path.dirname(args.xlsx_name))
        pass
    
    if not os.path.exists(os.path.join(root, args.xlsx_name)):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('Sheet1')
        worksheet.write(0, 0, 'exp_index')
        worksheet.write(0, 1, 'lr')
        worksheet.write(0, 2, 'wd')
        worksheet.write(0, 3, 'epochs')
        #worksheet.write(0, 4, 'loss')
        worksheet.write(0, 5, 'lr_decay')
        worksheet.write(0, 6, 'lr_controler')
        worksheet.write(0, 7, 'alpha')
        worksheet.write(0, 8, 'lambda_1')
        worksheet.write(0, 9, 'lambda_2')
        worksheet.write(0, 10, 'theta')
        worksheet.write(0, 11, 'seed')
        worksheet.write(0, 12, 'val_acc')
        worksheet.write(0, 13, 'val_auc')
        worksheet.write(0, 14, 'test_acc')
        worksheet.write(0, 15, 'test_auc')
        worksheet.write(0, 16, 'results_root')
        worksheet.write(0, 17, 'optimizer')
        worksheet.write(0, 18, 'dropout')
        worksheet.write(0, 19, 'dp')


        idx = 1
        worksheet.write(idx, 0, 1)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.wd)
        worksheet.write(idx, 3, args.epochs)
        #worksheet.write(idx, 4, args.loss)
        worksheet.write(idx, 5, args.lr_decay)
        worksheet.write(idx, 6, args.lr_controler)
        worksheet.write(idx, 7, args.alpha)
        worksheet.write(idx, 8, args.gamma)
        worksheet.write(idx, 9, args.lambda2)
        worksheet.write(idx, 10, args.delta1)
        worksheet.write(idx, 11, args.seed)
        worksheet.write(idx, 12, args.best_val_acc)
        worksheet.write(idx, 13, args.best_val_auc)
        worksheet.write(idx, 14, args.best_test_acc)
        worksheet.write(idx, 15, args.best_test_auc)
        worksheet.write(idx, 16, args.save_dir)
        worksheet.write(idx, 17, args.optimizer)
        #worksheet.write(idx, 18, args.dropout)
        #worksheet.write(idx, 19, args.dp)

        workbook.save(os.path.join(root, args.xlsx_name))
    else:
        rb = xlrd.open_workbook(os.path.join(root, args.xlsx_name))
        wb = x_copy(rb)
        worksheet = wb.get_sheet(0)
        idx = len(worksheet.get_rows())

        worksheet.write(idx, 0, 1)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.wd)
        worksheet.write(idx, 3, args.epochs)
        #worksheet.write(idx, 4, args.loss)
        worksheet.write(idx, 5, args.lr_decay)
        worksheet.write(idx, 6, args.lr_controler)
        worksheet.write(idx, 7, args.alpha)
        worksheet.write(idx, 8, args.gamma)
        worksheet.write(idx, 9, args.lambda2)
        worksheet.write(idx, 10, args.delta1)
        worksheet.write(idx, 11, args.seed)
        worksheet.write(idx, 12, args.best_val_acc)
        worksheet.write(idx, 13, args.best_val_auc)
        worksheet.write(idx, 14, args.best_test_acc)
        worksheet.write(idx, 15, args.best_test_auc)
        worksheet.write(idx, 16, args.save_dir)
        worksheet.write(idx, 17, args.optimizer)

        wb.save(os.path.join(root, args.xlsx_name))

import numpy as np
import torch
from torch.autograd import Variable
import logging

def print_model_parm_nums(model, logger = None):
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('test')
    total = sum([param.nelement() for param in model.parameters()])
    logger.info('  + Number of params: %.2fM' % (total / 1e6))

def print_model_parm_flops(model, logger = None):

    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('test')
    # prods = {}
    # def save_prods(self, input, output):
        # print 'flops:{}'.format(self.__class__.__name__)
        # print 'input:{}'.format(input)
        # print '_dim:{}'.format(input[0].dim())
        # print 'input_shape:{}'.format(np.prod(input[0].shape))
        # grads.append(np.prod(input[0].shape))

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width, input_depth = input[0].size()
        output_channels, output_height, output_width, output_depth = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * output_depth

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width, input_depth = input[0].size()
        output_channels, output_height, output_width, output_depth = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * output_depth

        list_pooling.append(flops)



    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose3d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool3d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
                foo(c)

    foo(model)
    input = Variable(torch.rand(3,1,64,64,64), requires_grad = True)
    out = model(input.cuda())


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    logger.info('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))