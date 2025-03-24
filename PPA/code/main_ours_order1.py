from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from utils import *
import time
import pickle
import copy
import datetime
import xlrd
import torch.utils.data as data


class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 label=0,
                 filename='std',
                 feature_list=[8, 9],
                 ):
        super(Clas_ppa_train, self).__init__()
        self.root = root  # excel path
        self.transform = transform
        self.eye = eye
        self.center = center
        self.label = label
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        
        self.base_grade_num = []
        self.base_grade_num_2 = []
        
        self.feature_mask_1 = np.zeros(72, ).astype('bool')
        self.feature_mask_2 = np.zeros(72, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True
            self.feature_mask_2[ids + 32] = True
        

        if filename == 'std':
            workbook1 = xlrd.open_workbook(
                r"./ppa-order1-std.xls")

        
        sheet1 = workbook1.sheet_by_index(0)
        
        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[5] == fold:
                if sheet1.row_values(rows)[3] in self.eye:
                    if str(sheet1.row_values(rows)[4]) in self.center:
                        '''
                        if is_endwith6 == 1:
                            if int(sheet1.row_values(rows)[2].split('/')[0][5]) == 6:
                                self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                                self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                                self.target_ppa.append(sheet1.row_values(rows)[6])
                                # print(np.array(sheet1.row_values(rows))[self.feature_mask_1])
                                self.feature_all_1.append(
                                    np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                                self.feature_all_2.append(
                                    np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                                self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                                self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))
                        else:
                            '''
                        self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                        self.target_ppa.append(sheet1.row_values(rows)[6])
                        # print(np.array(sheet1.row_values(rows))[self.feature_mask_1])
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))
    
    def __getitem__(self, index):
        
        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        target_ppa = self.target_ppa[index]
        
        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        base_target = [-1, -1]
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num[index] == 1:
            base_target[0] = 0
        
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num_2[index] == 6:
            base_target[1] = target_ppa
        
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        
        return img_1, \
               img_2, \
               torch.from_numpy(np.array(target_ppa).astype('int')), \
               torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
               torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
               torch.from_numpy(np.array(self.base_grade_num_2[index]).astype('int'))
    
    def __len__(self):
        return len(self.image_path_all_1)


def get_all_dataloader(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        Clas_ppa_train(args.data_root, fold='train', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           # transforms.RandomHorizontalFlip(p=0.5),
                           # transforms.RandomVerticalFlip(p=0.5),
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),
            
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test2', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val2', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test3', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val3', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test4', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val4', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test5', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val5', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    return train_loader, val_loader_list, test_loader_list


def train(args,
          model_1,
          model_2_generate,
          model_2_res,
          G_net,
          D_net,
          train_loader,
          optimizer_M_2_generate,
          optimizer_M_2_res,
          optimizer_G,
          optimizer_D,
          epoch):
    model_1.eval()
    train_loss_D = AverageMeter()
    train_loss_G = AverageMeter()
    train_loss_M2_reg_mono = AverageMeter()
    train_loss_M2_res = AverageMeter()
    train_loss_M2 = AverageMeter()
    train_loss_M2_gen_cls = AverageMeter()
    eps = 1e-5
    
    image_sequence = 0
    for batch_idx, (data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
        
        if batch_idx % 5 < args.D_epoch:
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for p in model_2_res.parameters():
                p.requires_grad = True
            for p in G_net.parameters():
                p.requires_grad = False
            for p in D_net.parameters():
                p.requires_grad = True
            for p in model_2_generate.parameters():
                p.requires_grad = True
            # for p in model_ESPCN.parameters():
            # p.requires_grad = False
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all = 1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                          all_generated_feature[0]], dim=0)
            
            real_loss_1 = D_net(featuremap_2, z1).mean(0).view(1)
            fake_loss_1 = D_net(generate_feature_1, z1).mean(0).view(1)
            loss_D = (real_loss_1 - fake_loss_1)
            
            optimizer_D.zero_grad()
            for p in D_net.parameters():
                p.data.clamp_(-args.wcl, args.wcl)
            loss_D.backward(retain_graph=False)
            optimizer_D.step()
            
            loss_res = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            gen_count = 0
            loss_res_count = 0
            reg_mono_count = 0
            
            res_feature_1 = generate_feature_1 - featuremap_1
            P_current_1_i = torch.softmax(model_2_generate(featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_current_1_j = torch.softmax(model_2_generate(featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            pred_gen_1 = model_2_generate(generate_feature_1, None)
            
            for i in range(data_1.size(0)):
                if int(grad_1[i].detach().cpu().numpy()) == 6 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    print('it should not be appear')
                    exit(1)
                if int(grad_1[i].detach().cpu().numpy()) == 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    loss_res_count += 1
                    reg_mono_count += 1
                    gen_count += 1
                    loss_res = torch.add(loss_res,
                                         F.nll_loss(torch.log(eps + P_residue_1_i[i].unsqueeze(0)), target_ppa[i]))
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                
                elif int(grad_1[i].detach().cpu().numpy()) != 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    loss_res_count += 1
                    reg_mono_count += 1
                    gen_count += 1
                    P_yj_1 = P_current_1_i[i][1] + P_current_1_i[i][0] * P_residue_1_i[i][1]
                    loss_res = torch.add(loss_res,
                                         F.nll_loss(
                                             torch.log(eps + torch.cat([1 - P_yj_1.unsqueeze(0), P_yj_1.unsqueeze(0)],
                                                                       dim=0).unsqueeze(0)), target_ppa[i]))
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                
                else:
                    reg_mono_count += 1
                    gen_count += 2
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
            
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_res_count != 0:
                loss_res = loss_res / loss_res_count
            if gen_count != 0:
                loss_gen = loss_gen / gen_count
            
            loss = args.lambda2 * loss_res + args.alpha * reg_mono + args.gamma * loss_gen
            
            if loss.item() != 0:
                optimizer_M_2_generate.zero_grad()
                optimizer_M_2_res.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_2_generate.step()
                optimizer_M_2_res.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_res_count > 0:
                train_loss_M2_res.update(loss_res.item(), loss_res_count)
                train_loss_M2.update(loss.item(), loss_res_count)
            if gen_count > 0:
                train_loss_M2_gen_cls.update(args.gamma * loss_gen.item(), gen_count)
            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
        
        if batch_idx % 5 >= args.D_epoch:
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for p in model_2_res.parameters():
                p.requires_grad = True
            for p in G_net.parameters():
                p.requires_grad = True
            for p in D_net.parameters():
                p.requires_grad = False
            for p in model_2_generate.parameters():
                p.requires_grad = True
            # for p in model_ESPCN.parameters():
            # p.requires_grad = True
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            
            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all=1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                    all_generated_feature[0]], dim=0)
            # generate_feature_2 = model_ESPCN(generate_feature_2)
            
            loss_G = D_net(generate_feature_1, z1).mean(0).view(1)
            
            loss_res = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            gen_count = 0
            loss_res_count = 0
            reg_mono_count = 0
            res_feature_1 = generate_feature_1 - featuremap_1
            
            P_current_1_i = torch.softmax(model_2_generate(featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_current_1_j = torch.softmax(model_2_generate(featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            pred_gen_1 = model_2_generate(generate_feature_1, None)
            if args.train_ori_data:
                pred_cur_i = model_2_generate(featuremap_1)
                pred_cur_j = model_2_generate(featuremap_2)
            
            for i in range(data_1.size(0)):
                if int(grad_1[i].detach().cpu().numpy()) == 6 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    print('it should not be appear')
                    exit(1)
                if int(grad_1[i].detach().cpu().numpy()) == 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    loss_res_count += 1
                    reg_mono_count += 1
                    gen_count += 1
                    loss_res = torch.add(loss_res,
                                         F.nll_loss(torch.log(eps + P_residue_1_i[i].unsqueeze(0)), target_ppa[i]))
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                    if args.train_ori_data:
                        if target_ppa[i] == 0:
                            gen_count += 2
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_i[i].unsqueeze(0), target_ppa[i]))
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                        else:
                            gen_count += 2
                            loss_gen = torch.add(loss_gen,
                                                 F.cross_entropy(pred_cur_i[i].unsqueeze(0), 1 - target_ppa[i]))
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                
                elif int(grad_1[i].detach().cpu().numpy()) != 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                    loss_res_count += 1
                    reg_mono_count += 1
                    gen_count += 1
                    P_yj_1 = P_current_1_i[i][1] + P_current_1_i[i][0] * P_residue_1_i[i][1]
                    loss_res = torch.add(loss_res,
                                         F.nll_loss(
                                             torch.log(eps + torch.cat([1 - P_yj_1.unsqueeze(0), P_yj_1.unsqueeze(0)],
                                                                       dim=0).unsqueeze(0)), target_ppa[i]))
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                    
                    if args.train_ori_data:
                        if target_ppa[i] == 0:
                            gen_count += 2
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_i[i].unsqueeze(0), target_ppa[i]))
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                        else:
                            gen_count += 1
                            loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                else:
                    reg_mono_count += 1
                    gen_count += 1
                    reg_mono = torch.add(reg_mono, torch.max(P_current_1_i[i][1] - P_current_1_j[i][1] + args.delta1,
                                                             torch.FloatTensor([0.0]).cuda()))
                    loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
            
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_res_count != 0:
                loss_res = loss_res / loss_res_count
            if gen_count != 0:
                loss_gen = loss_gen / gen_count
            
            loss_temp = args.lambda2 * loss_res + args.alpha * reg_mono + args.gamma * loss_gen
            loss_G_mse = args.beta * F.mse_loss(generate_feature_1, featuremap_2)
            loss = loss_G + loss_temp + loss_G_mse
            
            optimizer_M_2_generate.zero_grad()
            optimizer_M_2_res.zero_grad()
            optimizer_G.zero_grad()
            # optimizer_ESPCN.zero_grad()
            loss.backward(retain_graph=False)
            # optimizer_ESPCN.step()
            optimizer_G.step()
            optimizer_M_2_generate.step()
            optimizer_M_2_res.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_res_count > 0:
                train_loss_M2_res.update(loss_res.item(), loss_res_count)
                train_loss_M2.update(loss.item(), loss_res_count)
            if gen_count > 0:
                train_loss_M2_gen_cls.update(args.gamma * loss_gen.item(), gen_count)
            train_loss_G.update(loss_G.item(), data_1.size(0))
        
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] loss_res: {:.6f}, '
                         'reg_mono: {:.6f}, gen_cls: {:.6f}, overall: {:.6f}'.format(
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss_M2_res.avg,
            train_loss_M2_reg_mono.avg, train_loss_M2_gen_cls.avg, train_loss_M2.avg))
        
        args.logger.info('loss_D is real pred RA loss: {}'.format(train_loss_D.avg))
    
    loss = {
        'loss_M_current_reg': train_loss_M2_reg_mono.avg,
        'loss_M_minus': train_loss_M2_res.avg,
        'loss_M_gen_cls': train_loss_M2_gen_cls.avg,
        'loss': train_loss_M2.avg,
        'loss_D': train_loss_D.avg,
        'loss_G': train_loss_G.avg,
    }
    return loss


def evaluate(args,
             model_1,
             model_2_generate,
             model_2_res,
             G_net,
             # model_ESPCN,
             test_loader,
             epoch):
    model_1.eval()
    model_2_generate.eval()
    model_2_res.eval()
    G_net.eval()
    # model_ESPCN.eval()
    
    pred_result_current = np.zeros((len(test_loader.dataset), args.class_num))
    pred_result_minus = np.zeros((len(test_loader.dataset), args.class_num))
    correct_generate = 0
    correct_minus = 0
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    pred_label_minus = np.zeros((len(test_loader.dataset), 1))
    name = []
    test_loss_generate = 0
    test_loss_minus = 0
    
    pred_result_cur_res = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_cur_res = np.zeros((len(test_loader.dataset), 1))
    
    pred_result_average_all = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_average_all = np.zeros((len(test_loader.dataset), 1))
    
    pred_result_gen = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_gen = np.zeros((len(test_loader.dataset), 1))
    with torch.no_grad():
        batch_begin = 0
        
        for batch_idx, (data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(test_loader):
            data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), target_ppa.cuda(), feature_1.cuda(), feature_2.cuda(), \
                grad_1.cuda(), grad_2.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            
            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all=1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                    all_generated_feature[0]], dim=0)
            # generate_feature_1 = model_ESPCN(generate_feature_1)
            res_feature_1 = generate_feature_1 - featuremap_1
            
            P_current_1_i = torch.softmax(model_2_generate(featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            gen_pred = torch.softmax(model_2_generate(generate_feature_1, None), dim=1)
            
            pred_result_gen[batch_begin:batch_begin + data_1.size(0), :] = gen_pred.detach().cpu().numpy()
            pred_label_gen[batch_begin:batch_begin + data_1.size(0)] = gen_pred.argmax(dim=1,
                                                                                       keepdim=True).detach().cpu().numpy()
            pred_current = P_current_1_i.argmax(dim=1, keepdim=True)
            pred_minus = P_residue_1_i.argmax(dim=1, keepdim=True)
            correct_generate += pred_current.eq(target_ppa.view_as(pred_current)).sum().item()
            correct_minus += pred_minus.eq(target_ppa.view_as(pred_minus)).sum().item()
            
            pred_result_current[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_current_1_i,
                                                                                         dim=1).detach().cpu().numpy()
            pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_residue_1_i,
                                                                                       dim=1).detach().cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_current.detach().cpu().numpy()
            pred_label_minus[batch_begin:batch_begin + data_1.size(0)] = pred_minus.detach().cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            
            output_cur_res_1 = P_current_1_i[:, 1] + P_current_1_i[:, 0] * P_residue_1_i[:, 1]
            output_cur_res = torch.cat([1 - output_cur_res_1.unsqueeze(1), output_cur_res_1.unsqueeze(1)], dim=1)
            
            pred_cur_res = output_cur_res.argmax(dim=1, keepdim=True)
            pred_result_cur_res[batch_begin:batch_begin + data_1.size(0), :] = output_cur_res.detach().cpu().numpy()
            pred_label_cur_res[batch_begin:batch_begin + data_1.size(0)] = pred_cur_res.detach().cpu().numpy()
            
            output_average_temp = (P_current_1_i[:, 1] + P_current_1_i[:, 0] * P_residue_1_i[:, 1] + gen_pred[:,
                                                                                                     1]) / 2.
            output_average_all = torch.cat([1 - output_average_temp.unsqueeze(1), output_average_temp.unsqueeze(1)],
                                           dim=1)
            
            pred_average_all = output_average_all.argmax(dim=1, keepdim=True)
            pred_result_average_all[batch_begin:batch_begin + data_1.size(0),
            :] = output_average_all.detach().cpu().numpy()
            pred_label_average_all[batch_begin:batch_begin + data_1.size(0)] = pred_average_all.detach().cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
            
            if args.is_plot == 1:
                print('draw featuremap {} / {}'.format(batch_begin, len(test_loader.dataset)))
                for i in range(data_1.size(0)):
                    image_sequence = batch_begin - data_1.size(0) + i  ##range the image and path_name
                    if target_ppa[i] == 1:
                        
                        save_dir = '../Feature_map/Ecur_%s-Label1/' % (args.sequence,)
                        
                        unet_feature1_1 = generate_feature_1[i, :, :, :]  # add the unnomarlization
                        unet_feature1_1 = torch.unsqueeze(unet_feature1_1, 0)
                        if args.plot_sigmoid:
                            unet_feature1_1 = torch.sigmoid(unet_feature1_1)
                        unet_feature1_1 = unet_feature1_1.cpu().detach().numpy()
                        save_dir1 = (save_dir + 'unet/')
                        draw_features(16, 8, unet_feature1_1, save_dir1, image_sequence, test_loader, 'unet_',
                                      args.batch_size, args.dpi)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi)
                    
                    
                    elif target_ppa[i] == 0:
                        # image_sequence = batch_begin - data_1.size(0) + i
                        save_dir = '../Feature_map/Ecur_%s-Label0/' % (args.sequence,)
                        
                        unet_feature1_1 = generate_feature_1[i, :, :, :]  # add the unnomarlization
                        unet_feature1_1 = torch.unsqueeze(unet_feature1_1, 0)
                        if args.plot_sigmoid:
                            unet_feature1_1 = torch.sigmoid(unet_feature1_1)
                        unet_feature1_1 = unet_feature1_1.cpu().detach().numpy()
                        save_dir1 = (save_dir + 'unet/')
                        draw_features(16, 8, unet_feature1_1, save_dir1, image_sequence, test_loader, 'unet_',
                                      args.batch_size, args.dpi)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi)
    
    AUC_minus = sklearn.metrics.roc_auc_score(target, pred_result_minus[:, 1])
    acc_minus = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_minus, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_minus, axis=1))
    sensitivity_minus = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    specificity_minus = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])

    # AUC_gen = sklearn.metrics.roc_auc_score(target, pred_result_gen[:, 1])
    # acc_gen = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_gen, axis=1))
    # cm_gen = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_gen, axis=1))
    # sensitivity_gen = cm_gen[0, 0] / (cm_gen[0, 0] + cm_gen[0, 1])
    # specificity_gen = cm_gen[1, 1] / (cm_gen[1, 0] + cm_gen[1, 1])

    AUC_cur_res = sklearn.metrics.roc_auc_score(target, pred_result_cur_res[:, 1])
    acc_cur_res = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_cur_res, axis=1))
    cm_cur_res = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_cur_res, axis=1))
    sensitivity_cur_res = cm_cur_res[0, 0] / (cm_cur_res[0, 0] + cm_cur_res[0, 1])
    specificity_cur_res = cm_cur_res[1, 1] / (cm_cur_res[1, 0] + cm_cur_res[1, 1])
    
    AUC_average_all = sklearn.metrics.roc_auc_score(target, pred_result_average_all[:, 1])
    acc_average_all = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_average_all, axis=1))
    
    AUC_gen = sklearn.metrics.roc_auc_score(target, pred_result_gen[:, 1])
    acc_gen = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_gen, axis=1))
    
    args.logger.info('In epoch {} for generate, AUC is {}, acc is {}.'.format(epoch, AUC_gen, acc_gen))
    args.logger.info(
        'In epoch {} for minus, AUC is {}, acc is {}, loss is {}'.format(epoch, AUC_minus, acc_minus, test_loss_minus))
    args.logger.info('In epoch {} for cur_res, AUC is {}, acc is {}'.format(epoch, AUC_cur_res, acc_cur_res))
    args.logger.info(
        'In epoch {} for average all with gen, AUC is {}, acc is {}'.format(epoch, AUC_average_all, acc_average_all))
    args.logger.info('      ')
    
    results = {
        'AUC_minus': AUC_minus,
        'acc_minus': acc_minus,
        'sensitivity_minus': sensitivity_minus,
        'specificity_minus': specificity_minus,
        'pred_result_minus': pred_result_minus,
        'pred_label_minus': pred_label_minus,
        
        'AUC_cur_res': AUC_cur_res,
        'acc_cur_res': acc_cur_res,
        'sensitivity_cur_res': sensitivity_cur_res,
        'specificity_cur_res': specificity_cur_res,
        'pred_result_cur_res': pred_result_cur_res,
        'pred_label_cur_res': pred_label_cur_res,
        
        'AUC_average_all': AUC_average_all,
        'acc_average_all': acc_average_all,
        'pred_result_average_all': pred_result_average_all,
        'pred_label_average_all': pred_label_average_all,
        
        'pred_result_gen': pred_result_gen,
        'AUC_gen': AUC_gen,
        'acc_gen': acc_gen,
        
        'target': target,
        'image_path': name,
        
    }
    return results


def save_results(args,
                 model_1,
                 model_2_generate,
                 model_2_res,
                 G_net,
                 D_net,
                 train_results,
                 val_results_list,
                 test_results_list,
                 full_results,
                 optimizer_M_2_generate,
                 optimizer_M_2_res,
                 optimizer_G,
                 optimizer_D,
                 epoch):
    val_auc_average = (val_results_list[0]['AUC_average_all'] + val_results_list[1]['AUC_average_all'] +
                       val_results_list[2]['AUC_average_all'] + val_results_list[3]['AUC_average_all'] +
                       val_results_list[4]['AUC_average_all']) / 5
    val_acc_average = (val_results_list[0]['acc_average_all'] + val_results_list[1]['acc_average_all'] +
                       val_results_list[2]['acc_average_all'] + val_results_list[3]['acc_average_all'] +
                       val_results_list[4]['acc_average_all']) / 5
    test_auc_average = (test_results_list[0]['AUC_average_all'] + test_results_list[1]['AUC_average_all'] +
                        test_results_list[2]['AUC_average_all'] + test_results_list[3]['AUC_average_all'] +
                        test_results_list[4]['AUC_average_all']) / 5
    test_acc_average = (test_results_list[0]['acc_average_all'] + test_results_list[1]['acc_average_all'] +
                        test_results_list[2]['acc_average_all'] + test_results_list[3]['acc_average_all'] +
                        test_results_list[4]['acc_average_all']) / 5
    
    if args.best_test_acc < test_acc_average:
        args.best_test_acc = copy.deepcopy(test_acc_average)
        args.best_test_acc_epoch = copy.deepcopy(epoch)
    
    if args.best_test_auc < test_auc_average:
        args.best_test_auc = copy.deepcopy(test_auc_average)
        args.best_test_auc_epoch = copy.deepcopy(epoch)
    
    if args.best_val_acc < val_acc_average:
        args.best_val_acc = copy.deepcopy(val_acc_average)
        args.best_val_acc_epoch = copy.deepcopy(epoch)
    
    if args.best_val_auc < val_auc_average:
        args.best_val_auc = copy.deepcopy(val_auc_average)
        args.best_val_auc_epoch = copy.deepcopy(epoch)
    
    if epoch == args.best_test_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_G_net.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net.pt'))
    
    args.logger.info(
        'Utill now the best test acc epoch is : {},  acc is {}'.format(args.best_test_acc_epoch, args.best_test_acc))
    args.logger.info(
        'Utill now the best test AUC epoch is : {}, AUC is {}'.format(args.best_test_auc_epoch, args.best_test_auc))
    full_results[epoch] = {
        'train_results': copy.deepcopy(train_results),
        'test_results_list': copy.deepcopy(test_results_list),
        'val_results_list': copy.deepcopy(val_results_list),
    }
    pickle.dump(full_results, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))

    strs = 'minus'
    args.logger.info('results with %s'%strs)
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))

    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))

    strs = 'cur_res'
    args.logger.info('results with %s' % strs)
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    strs = 'gen'
    args.logger.info('results with %s' % strs)
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    strs = 'average_all'
    args.logger.info('results with %s' % strs)
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))

    if epoch == args.epochs:
        save_results_as_xlsx(root='', args=args)
        pass

    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_1': model_1.state_dict(),
            'model_2_generate': model_2_generate.state_dict(),
            'model_2_minus': model_2_res.state_dict(),
            'G_net': G_net.state_dict(),
            'D_net': D_net.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_M_2_generate': optimizer_M_2_generate.state_dict(),
            'optimizer_M_2_res': optimizer_M_2_res.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, is_best, base_dir=args.save_dir)
        # torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        # torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'Final_model_2_generate.pt'))
        # torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'Final_model_2_res.pt'))
        # torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'Final_G_net.pt'))
        # torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'Final_D_net.pt'))


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    train_loader, val_loader_list, test_loader_list = get_all_dataloader(args)
    
    model_1 = RN18_front(final_tanh=args.final_tanh).cuda()
    model_2_res = RN18_last_attr_ind(num_classes=args.class_num,
                                     feature_dim=len(args.feature_list),
                                     concat_num=1,
                                     task_num=5).cuda()
    model_2_generate = RN18_last_attr(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list) + 1,
                                      dropout=args.dropout, dp=args.dp).cuda()
    

    load_pytorch_model(model_1,
                       r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_1.pt')
    #load_pytorch_model(model_2_generate,
                       #r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_2_generate.pt')


    if args.G_net_type == 'G_net':
        G_net = Generator_LSTM_1_ind(feature_num=len(args.feature_list) + 1,
                                    final_tanh=args.final_tanh,
                                    in_channel=128,
                                    RNN_hidden=args.RNN_hidden).cuda()
    elif args.G_net_type == 'U_net':
        G_net = UNet(n_channels=128,
                     n_classes=128,
                     bilinear=args.bi_linear,
                     feature_num=len(args.feature_list) + 1,
                     final_tanh=args.final_tanh,
                     is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                     dw_type=args.dw_type).cuda()
    
    D_net = Discriminator(128).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_res, logger=args.logger)
    print_model_parm_nums(model_2_generate, logger=args.logger)
    print_model_parm_nums(G_net, logger=args.logger)
    print_model_parm_nums(D_net, logger=args.logger)
    print('-' * 20)
    print('-' * 20)

    optimizer_M_2_res = optim.SGD([{'params': model_2_res.parameters(), 'lr': args.lr2,
                                    'weight_decay': args.wd2, 'momentum': args.momentum}
                                   ])
    optimizer_M_2_generate = optim.SGD(
        [{'params': model_2_generate.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])
    optimizer_G = optim.RMSprop([{'params': G_net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
    optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)

    
    full_results = {}
    args = init_metric(args)
    
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train(args, model_1, model_2_generate, model_2_res, G_net, D_net,
                                  train_loader, optimizer_M_2_generate, optimizer_M_2_res, optimizer_G, optimizer_D,
                                  epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate(args, model_1, model_2_generate, model_2_res, G_net,
                             test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate(args, model_1, model_2_generate, model_2_res, G_net,
                             val_loader_list[ss], epoch))
            
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)
            adjust_learning_rate(optimizer_M_2_res, epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results(args,
                         model_1,
                         model_2_generate,
                         model_2_res,
                         G_net,
                         D_net,
                         train_results,
                         val_results_list,
                         test_results_list,
                         full_results,
                         optimizer_M_2_generate,
                         optimizer_M_2_res,
                         optimizer_G,
                         optimizer_D,
                         epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
