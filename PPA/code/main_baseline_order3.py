from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import xlrd
from models import *
from utils import *
import time
import pickle
import copy
import datetime


class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 label=0,
                 filename='std',
                 feature_list=[7, 8, 9],
                 order=2):
        super(Clas_ppa_train, self).__init__()
        self.root = root  # excel path
        self.transform = transform
        self.eye = eye
        self.center = center
        self.label = label
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.image_path_all_3 = []
        self.image_path_all_4 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.feature_all_3 = []
        self.feature_all_4 = []
        
        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.base_grade_num_3 = []
        self.base_grade_num_4 = []
        
        self.feature_mask_1 = np.zeros(137, ).astype('bool')
        self.feature_mask_2 = np.zeros(137, ).astype('bool')
        self.feature_mask_3 = np.zeros(137, ).astype('bool')
        self.feature_mask_4 = np.zeros(137, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids + 1] = True
            self.feature_mask_2[ids + 1 + 32] = True
            self.feature_mask_3[ids + 1 + 64] = True
            self.feature_mask_4[ids + 1 + 96] = True
        
        if filename == 'std_all':
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"./ppa_06-29-order3-std-8.xls")
        
        sheet1 = workbook1.sheet_by_index(0)
        
        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[7] == fold:
                if sheet1.row_values(rows)[5] in self.eye:
                    if str(sheet1.row_values(rows)[6]) in self.center:
                        self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                        self.image_path_all_3.append(os.path.join(self.root, sheet1.row_values(rows)[3]))
                        self.image_path_all_4.append(os.path.join(self.root, sheet1.row_values(rows)[4]))
                        
                        self.target_ppa.append(sheet1.row_values(rows)[8])
                        
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.feature_all_3.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_3].astype('float32'))
                        self.feature_all_4.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_4].astype('float32'))
                        
                        self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))
                        self.base_grade_num_3.append(int(sheet1.row_values(rows)[3].split('/')[0][-1]))
                        self.base_grade_num_4.append(int(sheet1.row_values(rows)[4].split('/')[0][-1]))
    
    def __getitem__(self, index):
        
        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        img_path_3, img_path_4 = self.image_path_all_3[index], self.image_path_all_4[index]
        target_ppa = self.target_ppa[index]
        
        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        img_3 = Image.open(img_path_3)
        img_4 = Image.open(img_path_4)
        
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
            img_4 = self.transform(img_4)
        
        return img_1, \
               img_2, \
               img_3, \
               img_4, \
               torch.from_numpy(np.array(target_ppa).astype('int')), \
               torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_3[index]).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_4[index]).astype('float32')), \
               torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
               torch.from_numpy(np.array(self.base_grade_num_2[index]).astype('int')), \
               torch.from_numpy(np.array(self.base_grade_num_3[index]).astype('int')), \
               torch.from_numpy(np.array(self.base_grade_num_4[index]).astype('int'))
    
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
    
    return train_loader, val_loader_list, test_loader_list


def train_baseline(args,
                   model_1,
                   model_2_generate,
                   train_loader,
                   optimizer_M_1,
                   optimizer_M_2_generate,
                   epoch):
    model_1.train()
    model_2_generate.train()
    
    train_loss_M2_generate = AverageMeter()
    
    image_sequence = 0
    correct_generate = 0
    for batch_idx, (data_1, data_2, data_3, data_4, target_ppa, feature_1, feature_2, feature_3, feature_4,
                    grad_1, grad_2, grad_3, grad_4) in enumerate(train_loader):
        data_1, data_2, data_3, data_4, target_ppa, feature_1, feature_2, feature_3, feature_4, \
        grad_1, grad_2, grad_3, grad_4 = \
            data_1.cuda(), data_2.cuda(), data_3.cuda(), data_4.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), feature_3.cuda(), feature_4.cuda(), grad_1.cuda(), grad_2.cuda(), \
            grad_3.cuda(), grad_4.cuda()
        
        loss_generate_sum = torch.FloatTensor([0.0]).cuda()
        image_sequence += data_1.size(0)
        for p in model_1.parameters():
            p.requires_grad = True
        for p in model_2_generate.parameters():
            p.requires_grad = True
        
        featuremap_1 = model_1(data_1)
        featuremap_2 = model_1(data_2)
        featuremap_3 = model_1(data_3)
        if args.model == 'MM_F':
            output_generate_1 = model_2_generate(featuremap_1, torch.cat([feature_1,
                                                            (grad_4 - grad_1).float().view(data_2.size(0), -1)], 1))
            output_generate_2 = model_2_generate(featuremap_2, torch.cat([feature_2,
                                                            (grad_4 - grad_2).float().view(data_2.size(0), -1)], 1))
            output_generate_3 = model_2_generate(featuremap_3, torch.cat([feature_3,
                                                                          (grad_4 - grad_3).float().view(data_2.size(0),
                                                                                                         -1)], 1))
        else:
            output_generate_1 = model_2_generate(featuremap_1)
            output_generate_2 = model_2_generate(featuremap_2)
            output_generate_3 = model_2_generate(featuremap_3)
        
        if args.pred_future:
            loss_M_generate = F.cross_entropy((output_generate_1 + output_generate_2 + output_generate_3)/3, target_ppa)
        
        else:
            sample_num = 0
            loss_M_generate = torch.FloatTensor([0.0]).cuda()
            
            for ii in range(data_1.size(0)):
                if target_ppa[ii].detach().cpu().numpy() == 0:
                    sample_num += 2
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(0), torch.LongTensor([0]).cuda())
                        + F.cross_entropy(output_generate_2[ii].unsqueeze(0), torch.LongTensor([0]).cuda()))
                elif int(grad_1[ii].detach().cpu().numpy()) == 1:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(0),
                                        torch.LongTensor([0]).cuda()))  # torch.LongTensor([0]).cuda()))
                elif int(grad_1[ii].detach().cpu().numpy()) == 6:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))
                else:
                    pass
                
                if target_ppa[ii].detach().cpu().numpy() == 1 and int(grad_2[ii].detach().cpu().numpy()) == 6:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_2[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))
            
            if sample_num == 0:
                continue
            loss_M_generate = loss_M_generate / sample_num
        
        res = F.softmax(output_generate_1, dim=1)[:, 1] - F.softmax(output_generate_2, dim=1)[:, 1] + args.delta1 + \
              F.softmax(output_generate_2, dim=1)[:, 1] - F.softmax(output_generate_3, dim=1)[:, 1] + args.delta1
        res[res < 0] = 0
        reg_loss = torch.mean(res)
        loss_generate_sum += loss_M_generate + args.alpha * reg_loss
        
        target_ppa = target_ppa.reshape(-1, 1)
        pred_generate_1 = output_generate_1.argmax(dim=1, keepdim=True)
        correct_generate += pred_generate_1.eq(target_ppa.view_as(pred_generate_1)).sum().item()
        if not loss_generate_sum == 0:
            optimizer_M_1.zero_grad()
            optimizer_M_2_generate.zero_grad()
            loss_generate_sum.backward(retain_graph=False)
            optimizer_M_1.step()
            optimizer_M_2_generate.step()
            
            train_loss_M2_generate.update(loss_generate_sum.item(), data_1.size(0))  ####add loss_MG
            args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(
                epoch, batch_idx * len(data_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss_M2_generate.avg))
    args.logger.info('In epoch {}, acc is : {}'.format(epoch, correct_generate / len(train_loader.dataset)))
    
    loss = {
        'loss_M_generate': train_loss_M2_generate.avg,
    }
    return loss


def evaluate_baseline(args,
                      model_1,
                      model_2_generate,
                      test_loader,
                      epoch):
    model_1.eval()
    model_2_generate.eval()
    
    pred_result_generate = np.zeros((len(test_loader.dataset), 2))
    correct_generate = 0
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    name = []
    test_loss_generate = 0
    with torch.no_grad():
        batch_begin = 0
        for batch_idx, (data_1, data_2, data_3, data_4, target_ppa, feature_1, feature_2, feature_3, feature_4,
                        grad_1, grad_2, grad_3, grad_4) in enumerate(test_loader):
            data_1, data_2, data_3, data_4, target_ppa, feature_1, feature_2, feature_3, feature_4, \
            grad_1, grad_2, grad_3, grad_4 = \
                data_1.cuda(), data_2.cuda(), data_3.cuda(), data_4.cuda(), target_ppa.cuda(), feature_1.cuda(), \
                feature_2.cuda(), feature_3.cuda(), feature_4.cuda(), grad_1.cuda(), grad_2.cuda(), \
                grad_3.cuda(), grad_4.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            featuremap_3 = model_1(data_3)
            if args.model == 'MM_F':
                output_generate_1 = model_2_generate(featuremap_1, torch.cat([feature_1,
                                                                              (grad_4 - grad_1).float().view(
                                                                                  data_2.size(0), -1)], 1))
                output_generate_1 = output_generate_1 + model_2_generate(featuremap_2, torch.cat([feature_2,
                                                                              (grad_4 - grad_2).float().view(
                                                                                  data_2.size(0), -1)], 1))
                output_generate_1 = output_generate_1 / 3 + model_2_generate(featuremap_3,
                                                                             torch.cat([feature_3,
                                                                                        (grad_4 - grad_3).float().
                                                                                       view(data_2.size(0), -1)],1)) / 3
            else:
                output_generate_1 = model_2_generate(featuremap_1) / 3 + \
                                    model_2_generate(featuremap_2) / 3 + \
                                    model_2_generate(featuremap_3) / 3
            
            output_generate_1 = F.softmax(output_generate_1, dim=1)
            pred_generate = output_generate_1.argmax(dim=1, keepdim=True)
            correct_generate += pred_generate.eq(target_ppa.view_as(pred_generate)).sum().item()
            test_loss_generate += F.cross_entropy(output_generate_1, target_ppa, reduction='sum').item()
            pred_result_generate[batch_begin:batch_begin + data_1.size(0), :] = output_generate_1.cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_generate.cpu().numpy()
            
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
    
    test_loss_generate /= len(test_loader.dataset)
    
    AUC_generate = sklearn.metrics.roc_auc_score(target,
                                                 pred_result_generate[:, 1])  # pred - output, labelall-target
    acc_generate = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_generate, axis=1))
    cm_generate = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_generate, axis=1))
    sensitivity_generate = cm_generate[0, 0] / (cm_generate[0, 0] + cm_generate[0, 1])
    specificity_generate = cm_generate[1, 1] / (cm_generate[1, 0] + cm_generate[1, 1])
    
    args.logger.info(
        'In epoch {} for generate, AUC is {}, acc is {}. loss is {}'.format(epoch, AUC_generate, acc_generate,
                                                                            test_loss_generate))
    
    results = {
        'AUC_average_all': AUC_generate,
        'acc_average_all': acc_generate,
        'sensitivity': sensitivity_generate,
        'specificity': specificity_generate,
        'pred_result': pred_result_generate,
        'pred_label': pred_label_generate,
        'loss': test_loss_generate,
        'target': target,
        'image_path': name,
        
    }
    return results


def save_results_baseline(args,
                          model_1,
                          model_2_list,
                          train_results,
                          val_results_list,
                          test_results_list,
                          full_results,
                          optimizer_M_1,
                          optimizer_M_2_list,
                          epoch):
    val_auc_average = (val_results_list[0]['AUC_average_all'] + val_results_list[1]['AUC_average_all'] +
                       val_results_list[2]['AUC_average_all']) / 3
    val_acc_average = (val_results_list[0]['acc_average_all'] + val_results_list[1]['acc_average_all'] +
                       val_results_list[2]['acc_average_all']) / 3
    test_auc_average = (test_results_list[0]['AUC_average_all'] + test_results_list[1]['AUC_average_all'] +
                        test_results_list[2]['AUC_average_all']) / 3
    test_acc_average = (test_results_list[0]['acc_average_all'] + test_results_list[1]['acc_average_all'] +
                        test_results_list[2]['acc_average_all']) / 3
    
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
    
    model_2_list_state_dict = model_2_list.state_dict()
    if epoch == args.best_test_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_1.pt'))
        torch.save(model_2_list_state_dict, os.path.join(args.save_dir, 'best_test_acc' + '_model_2_generate.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_1.pt'))
        torch.save(model_2_list_state_dict, os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_1.pt'))
        torch.save(model_2_list_state_dict, os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_1.pt'))
        torch.save(model_2_list_state_dict, os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
    
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
    
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_average_all']
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_average_all']))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_average_all']
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_average_all']))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    if epoch == args.epochs:
        save_results_as_xlsx(root='', args=args)
        pass
    
    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_1': model_1.state_dict(),
            'model_2_generate': model_2_list_state_dict,
            'best_test_acc': args.best_test_acc,
        }, is_best, base_dir=args.save_dir)
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        torch.save(model_2_list_state_dict, os.path.join(args.save_dir, 'Final_model_2_generate.pt'))


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    train_loader, val_loader_list, test_loader_list = get_all_dataloader(args)
    
    model_1 = RN18_front(final_tanh=args.final_tanh).cuda()
    if args.model == 'MM_F':
        model_2_generate = RN18_last_attr_e(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list)+1).cuda()
    else:
        model_2_generate = RN18_last_e(num_classes=args.class_num).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_generate, logger=args.logger)
    print('-' * 20)
    print('-' * 20)
    
    optimizer_M_1 = optim.SGD([{'params': model_1.parameters(), 'lr': args.lr2,
                                'weight_decay': args.wd2, 'momentum': args.momentum}])
    optimizer_M_2_generate = optim.SGD(
        [{'params': model_2_generate.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])
    
    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train_baseline(args, model_1, model_2_generate, train_loader,
                                           optimizer_M_1, optimizer_M_2_generate, epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate_baseline(args, model_1, model_2_generate, test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate_baseline(args, model_1, model_2_generate, val_loader_list[ss], epoch))
            
            adjust_learning_rate(optimizer_M_1, epoch, args)
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline(
                args,
                model_1,
                model_2_generate,
                train_results,
                val_results_list,
                test_results_list,
                full_results,
                optimizer_M_1,
                optimizer_M_2_generate,
                epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)

if __name__ == '__main__':
    main()
