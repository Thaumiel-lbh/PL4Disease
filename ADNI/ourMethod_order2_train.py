# coding=utf-8
import argparse
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import Order2Data
from model import *
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler


# clip_gradient(optimizer, opt.clip)

def train(opt):
    # 获取当前的日期和时间
    now = datetime.datetime.now()
    # 格式化日期和时间
    formatted_now = now.strftime("%Y%m%d_%H_%M_%S")
    
    # tensorboard
    tensorboard_path = os.path.join(opt.savepath, "tensorboard")
    loss_D_writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, "summary_loss_D"))
    loss_G_writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, "summary_loss_G"))
    loss_res_writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, "summary_loss_res"))
    loss_gen_writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, "summary_loss_gen"))

    # Datasets
    dataset_train = Order2Data(data_index='./data/dataset_order2.xls', data_root=opt.data_root, mode='train')
    dataset_test = Order2Data(data_index='./data/dataset_order2.xls', data_root=opt.data_root, mode='test')
    
    dataloader_train = DataLoader(dataset_train, sampler=ImbalancedDatasetSampler(dataset_train), 
                                  batch_size=opt.batchsize, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchsize, shuffle=False, num_workers=8)
    data_size = opt.data_size
    
    # Model
    ## extractor
    extractor = RN18_extrator().cuda()
    extractor.eval()
    extractor.load_state_dict(torch.load(opt.extractor_path)['extractor_state_dict'])
    ## generator
    G_net = Generator_LSTM_2(in_channel=64).cuda()
    ## discriminator
    D_net = Discriminator(in_channels=64, data_size=data_size).cuda()
    ## classifer
    C_net = RN18_Classifer().cuda()
    ## classifer_res
    C_Res_net_1 = RN18_Res_Classifer(num_classes=2).cuda() # P(T=2|t2=1), P(T=1|t2=1)
    C_Res_net_0 = RN18_Res_Classifer(num_classes=3).cuda() # P(T=2|t2=0), P(T=1|t2=0), P(T=0|t2=0)
    ## classifer_future
    C_future_net = RN18_Classifer_future().cuda()

    # optimizer
    ## for C_net
    optimizer_C = torch.optim.SGD(params=C_net.parameters(),
                                  lr=opt.lr2,
                                  weight_decay=opt.wd2,
                                  momentum=opt.momentum)
    scheduler_C = torch.optim.lr_scheduler.StepLR(optimizer_C, step_size=23000, gamma=0.5)
    ## for C_future_net
    optimizer_C_Future = torch.optim.SGD(params=C_net.parameters(),
                                  lr=opt.lr2,
                                  weight_decay=opt.wd2,
                                  momentum=opt.momentum)
    scheduler_C_Future = torch.optim.lr_scheduler.StepLR(optimizer_C, step_size=23000, gamma=0.5)
    ## for C_Res_net
    optimizer_C_Res_0 = torch.optim.SGD(params=C_Res_net_0.parameters(),
                                      lr=opt.lr2,
                                      weight_decay=opt.wd2,
                                      momentum=opt.momentum)
    scheduler_C_Res_0 = torch.optim.lr_scheduler.StepLR(optimizer_C_Res_0, step_size=23000, gamma=0.5)
    
    optimizer_C_Res_1 = torch.optim.SGD(params=C_Res_net_1.parameters(),
                                      lr=opt.lr2,
                                      weight_decay=opt.wd2,
                                      momentum=opt.momentum)
    scheduler_C_Res_1 = torch.optim.lr_scheduler.StepLR(optimizer_C_Res_1, step_size=23000, gamma=0.5)    
    
    ## for G_net
    optimizer_G = torch.optim.RMSprop(params=G_net.parameters(), lr=opt.lr_G, weight_decay=opt.wd)
    ## for D_net
    optimizer_D = torch.optim.RMSprop(params=D_net.parameters(), lr=opt.lr_D, weight_decay=opt.wd)

    # train
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)
    global_step = 0
    best_auc = 0
    best_acc = 0
    best_epoch = 0

    for epoch in range(int(opt.epoch)):
        
        # train
        C_net.train()
        C_future_net.train()
        C_Res_net_0.train()
        C_Res_net_1.train()

        total_step = len(dataloader_train)

        for step, (data_3D_t1, data_3D_t2, data_3D_T, 
                   label_t1, label_t2, label_T, 
                   grad1, grad2, grad3) in enumerate(dataloader_train):
            batch_size = data_3D_t1.size(0)
            data_3D_t1, data_3D_t2, data_3D_T, label_t2, label_T, grad1, grad2, grad3 = data_3D_t1.cuda().float(), \
                                                                                      data_3D_t2.cuda().float(), \
                                                                                      data_3D_T.cuda().float(), \
                                                                                      label_t2.cuda().float(), \
                                                                                      label_T.cuda().float(), \
                                                                                      grad1.cuda(), \
                                                                                      grad2.cuda(), \
                                                                                      grad3.cuda()

            data_3D_t1 = F.interpolate(data_3D_t1, size=(data_size, data_size, data_size), mode='trilinear')
            data_3D_t2 = F.interpolate(data_3D_t2, size=(data_size, data_size, data_size), mode='trilinear')
            data_3D_T = F.interpolate(data_3D_T, size=(data_size, data_size, data_size), mode='trilinear')

            featuremap_t1 = extractor(data_3D_t1)
            featuremap_t2 = extractor(data_3D_t2)
            featuremap_T = extractor(data_3D_T)

            # generate feature
            z1 = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            z2 = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            generated_feature_2_by_1, generated_feature_3_by_12, = G_net(featuremap_t1, featuremap_t2, 
                                                                            z1, z2, 
                                                                            grad1, grad2, grad3)
            # compute res feature
            res_feature_12 = generated_feature_2_by_1 - featuremap_t1
            res_feature_23 = generated_feature_3_by_12 - featuremap_t2

            
            for p in extractor.parameters():
                p.requires_grad = False
            for p in C_Res_net_0.parameters():
                p.requires_grad = True
            for p in C_Res_net_1.parameters():
                p.requires_grad = True
            for p in C_future_net.parameters():
                p.requires_grad = True
            for p in C_net.parameters():
                p.requires_grad = True
                
            if epoch % opt.D_epoch != 0:
                for p in G_net.parameters():
                    p.requires_grad = False
                for p in D_net.parameters():  # 训练四次D
                    p.requires_grad = True
                D_net.train()
                G_net.eval()
                real_loss_2_by_1 = D_net(featuremap_t2).mean(0).view(1)
                real_loss_3_by_12 = D_net(featuremap_T).mean(0).view(1)
                fake_loss_2_by_1 = D_net(generated_feature_2_by_1).mean(0).view(1)
                fake_loss_3_by_12 = D_net(generated_feature_3_by_12).mean(0).view(1)
                loss_D = (real_loss_2_by_1 - fake_loss_2_by_1 + real_loss_3_by_12 - fake_loss_3_by_12) / 2

                loss_D_writer.add_scalar("loss_D", loss_D.detach(), epoch)

                optimizer_D.zero_grad()
                for p in D_net.parameters():
                    p.data.clamp_(-opt.wcl, opt.wcl)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
            else:
                for p in G_net.parameters():
                    p.requires_grad = True
                for p in D_net.parameters():  # 训练四次D
                    p.requires_grad = False

                D_net.eval()
                G_net.train()

            # pred
            # 为了训练分类器
            pred_t1 = torch.softmax(C_net(featuremap_t1), dim=1)
            pred_t2 = torch.softmax(C_net(featuremap_t2), dim=1)
            pred_T = torch.softmax(C_net(featuremap_T), dim=1)
            # 为了训练未来分类器
            pred_t1_to_t2 = torch.softmax(C_future_net(featuremap_t1, grad1, grad2), dim=1)
            pred_t1_to_T = torch.softmax(C_future_net(featuremap_t1, grad1, grad3), dim=1)
            pred_t2_to_T = torch.softmax(C_future_net(featuremap_t2, grad2, grad3), dim=1)
            # 为了训练Res分类器
            pred_res_23_0 = torch.softmax(C_Res_net_0(res_feature_23), dim=1) # P(T=0|t2=0), P(T=1|t2=0), P(T=2|t2=0)
            pred_res_23_1 = torch.softmax(C_Res_net_1(res_feature_23), dim=1) # P(T=1|t2=1), P(T=2|t2=1)
            # 为了训练生成器
            pred_gen_2_by_1 = torch.softmax(C_net(generated_feature_2_by_1), dim=1)
            pred_gen_3_by_12 = torch.softmax(C_net(generated_feature_3_by_12), dim=1)

            # t2时刻为0的概率
            pred_t2_is_0 = (pred_t2[::, 0] + pred_t1_to_t2[::, 0]) / 2
            # t2时刻为1的概率
            pred_t2_is_1 = (pred_t2[::, 1] + pred_t1_to_t2[::, 1]) / 2
            # t2时刻为2的概率
            pred_t2_is_2 = (pred_t2[::, 2] + pred_t1_to_t2[::, 2]) / 2
            # P(T=1) = P(t2=1) * P(T=1|t2=1) + P(t2=0) * P(T=1|t2=0)
            pred1 = pred_t2_is_1 * pred_res_23_1[::, 0] + pred_t2_is_0 * pred_res_23_0[::, 1]
            # P(T=2) = P(t2=2) + P(t2=1) * P(T=2|t2=1) + P(t2=0) * P(T=1|t2=0)
            pred2 = pred_t2_is_2 + pred_t2_is_1 * pred_res_23_1[::, 1] + pred_t2_is_0 * pred_res_23_0[::, 2]
            # 变为3维的预测向量
            pred = torch.cat([(1 - pred1 - pred2).unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1)], dim=1)

            # loss
            ## main_loss
            loss_res = F.cross_entropy(pred, label_T)
            ## loss for C_net_future
            loss_future = F.cross_entropy(pred_t1_to_t2, label_t2)
            ## loss for G_net
            loss_gen_2_by_1 = F.cross_entropy(pred_gen_2_by_1, label_t2)
            loss_gen_3_by_12 = F.cross_entropy(pred_gen_3_by_12, label_T)
            loss_gen = (loss_gen_3_by_12 + loss_gen_2_by_1) / 2
            ## regularizer
            P_t1 = F.one_hot(torch.argmax(pred_t1, dim=1), num_classes=3)
            P_t2 = F.one_hot(torch.argmax(pred_t2, dim=1), num_classes=3)
            P_T = F.one_hot(torch.argmax(pred_T, dim=1), num_classes=3)
            reg_12 = torch.where(P_t1 == P_t2, pred_t1 - pred_t2 + opt.margin, torch.zeros_like(P_t1).float().cuda())
            reg_23 = torch.where(P_t2 == P_T, pred_t2 - pred_T + opt.margin, torch.zeros_like(P_t2).float().cuda())
            reg_mono = (torch.clamp(reg_12, 0) + torch.clamp(reg_23, 0)) / 2
            reg_mono = torch.sum(reg_mono) / batch_size
            ## all loss and backward
            if epoch % opt.D_epoch != 0:
                # train D
                # loss
                loss = opt.lw_lambda * loss_res + opt.lw_alpha * reg_mono + opt.lw_gamma * loss_gen + opt.lw_theta * loss_future
                loss_gen_writer.add_scalar("loss_gen", loss_gen.detach(), epoch)
                loss_res_writer.add_scalar("loss_res", loss_res.detach(), epoch)
                # backward
                optimizer_C.zero_grad()
                optimizer_C_Future.zero_grad()
                optimizer_C_Res_0.zero_grad()
                optimizer_C_Res_1.zero_grad()
                
                loss.backward(retain_graph=False)
                
                optimizer_C.step()
                optimizer_C_Future.step()
                optimizer_C_Res_0.step()
                optimizer_C_Res_1.step()
                
                scheduler_C.step()
                scheduler_C_Future.step()
                scheduler_C_Res_0.step()
                scheduler_C_Res_1.step()

            else:
                # train G
                # loss
                loss_G = (D_net(generated_feature_2_by_1).mean(0).view(1) +
                          D_net(generated_feature_3_by_12).mean(0).view(1)) / 2
                loss = opt.lw_lambda * loss_res + opt.lw_alpha * reg_mono + opt.lw_gamma * loss_gen + opt.lw_theta * loss_future

                loss_gen_writer.add_scalar("loss_gen", loss_gen.detach(), epoch)
                loss_res_writer.add_scalar("loss_res", loss_res.detach(), epoch)

                loss_G_mse = (F.mse_loss(generated_feature_2_by_1, featuremap_t2) +
                              F.mse_loss(generated_feature_3_by_12, featuremap_T)) / 2

                loss_G_writer.add_scalar("loss_G", loss_G.detach() + loss_G_mse.detach(), epoch)

                loss = loss_G + loss + loss_G_mse
                # backward
                optimizer_C.zero_grad()
                optimizer_C_Future.zero_grad()
                optimizer_C_Res_0.zero_grad()
                optimizer_C_Res_1.zero_grad()
                optimizer_G.zero_grad()
                
                loss.backward(retain_graph=False)
                
                optimizer_C.step()
                optimizer_C_Future.step()
                optimizer_C_Res_0.step()
                optimizer_C_Res_1.step()
                optimizer_G.step()
                
                scheduler_C.step()
                scheduler_C_Future.step()
                scheduler_C_Res_0.step()
                scheduler_C_Res_1.step()

            global_step += 1
            if step % 10 == 0 or step == total_step - 1:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f ' %
                      (datetime.datetime.now(), global_step, epoch, opt.epoch,
                       optimizer_C.param_groups[0]['lr'], loss.item()))

        # eval
        G_net.eval()
        C_Res_net_0.eval()
        C_Res_net_1.eval()
        C_future_net.eval()
        C_net.eval()
        D_net.eval()

        pred_all = []
        pred_gen_all = []
        pred_average_all = []
        label_all = []

        for step, (data_3D_t1, data_3D_t2, data_3D_T, 
                   label_t1, label_t2, label_T, 
                   grad1, grad2, grad3) in enumerate(dataloader_test):
            batch_size = data_3D_t1.size(0)
            data_3D_t1, data_3D_t2, data_3D_T, label_t2, label_T, grad1, grad2, grad3 = data_3D_t1.cuda().float(), \
                                                                                        data_3D_t2.cuda().float(), \
                                                                                        data_3D_T.cuda().float(), \
                                                                                        label_t2.cuda().float(), \
                                                                                        label_T.cuda().float(), \
                                                                                        grad1.cuda(), \
                                                                                        grad2.cuda(), \
                                                                                        grad3.cuda()

            data_3D_t1 = F.interpolate(data_3D_t1, size=(data_size, data_size, data_size), mode='trilinear')
            data_3D_t2 = F.interpolate(data_3D_t2, size=(data_size, data_size, data_size), mode='trilinear')

            # inference
            featuremap_t1 = extractor(data_3D_t1)
            featuremap_t2 = extractor(data_3D_t2)
            ## generate feature
            z1 = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            z2 = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            generated_feature_2_by_1, generated_feature_3_by_12 = G_net(featuremap_t1, featuremap_t2, 
                                                                           z1, z2, 
                                                                           grad1, grad2, grad3)
            ## res
            res_feature = generated_feature_3_by_12 - featuremap_t2
            ## pred
            ### cur
            pred_t2 = torch.softmax(C_net(featuremap_t2), dim=1)
            ### future
            pred_t1_to_t2 = torch.softmax(C_future_net(featuremap_t1, grad1, grad2), dim=1)
            ### res
            pred_res_23_0 = torch.softmax(C_Res_net_0(res_feature), dim=1) # P(T=0|t2=0), P(T=1|t2=0), P(T=2|t2=0)
            pred_res_23_1 = torch.softmax(C_Res_net_1(res_feature), dim=1) # P(T=1|t2=1), P(T=2|t2=1)
            # t2时刻为0的概率
            pred_t2_is_0 = (pred_t2[::, 0] + pred_t1_to_t2[::, 0]) / 2
            # t2时刻为1的概率
            pred_t2_is_1 = (pred_t2[::, 1] + pred_t1_to_t2[::, 1]) / 2
            # t2时刻为2的概率
            pred_t2_is_2 = (pred_t2[::, 2] + pred_t1_to_t2[::, 2]) / 2
            # P(T=1) = P(t2=1) * P(T=1|t2=1) + P(t2=0) * P(T=1|t2=0)
            pred1 = pred_t2_is_1 * pred_res_23_1[::, 0] + pred_t2_is_0 * pred_res_23_0[::, 1]
            # P(T=2) = P(t2=2) + P(t2=1) * P(T=2|t2=1) + P(t2=0) * P(T=1|t2=0)
            pred2 = pred_t2_is_2 + pred_t2_is_1 * pred_res_23_1[::, 1] + pred_t2_is_0 * pred_res_23_0[::, 2]
            # 变为3维的预测向量
            pred = torch.cat([(1 - pred1 - pred2).unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1)], dim=1)
            pred_gen = torch.softmax(C_net(generated_feature_3_by_12), dim=1)
            pred_average = (pred_gen + pred) / 2

            pred = pred.detach().cpu().numpy()
            pred_all.extend(pred)

            pred_gen = pred_gen.detach().cpu().numpy()
            pred_gen_all.extend(pred_gen)

            pred_average = pred_average.detach().cpu().numpy()
            pred_average_all.extend(pred_average)

            label_T = label_T.cpu().numpy()
            label_all.extend(label_T)

        # compute metric
        auc = roc_auc_score(label_all, pred_all, multi_class='ovr')
        auc_gen = roc_auc_score(label_all, pred_gen_all, multi_class='ovr')
        auc_average = roc_auc_score(label_all, pred_average_all, multi_class='ovr')

        acc = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_all, axis=1))
        acc_gen = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_gen_all, axis=1))
        acc_average = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_average_all, axis=1))

        if auc > best_auc:
            states = {
                'metric_auc': auc,
                'metric_acc': acc,
                'C_Net_dict': C_net.state_dict(),
                'D_Net_dict': D_net.state_dict(),
                'G_Net_dict': G_net.state_dict(),
                'C_Res_Net_0_dict': C_Res_net_0.state_dict(),
                'C_Res_Net_1_dict': C_Res_net_1.state_dict(),
                'C_Future_Net_dict': C_future_net.state_dict()
            }
            torch.save(states, os.path.join(opt.savepath,
                                            'model-ourMethod-order2-bestauc-%s.pth' % formatted_now))
            best_auc = auc
            best_epoch = epoch
            
        if acc > best_acc:
            states = {
                'metric_auc': auc,
                'metric_acc': acc,
                'C_Net_dict': C_net.state_dict(),
                'D_Net_dict': D_net.state_dict(),
                'G_Net_dict': G_net.state_dict(),
                'C_Res_Net_0_dict': C_Res_net_0.state_dict(),
                'C_Res_Net_1_dict': C_Res_net_1.state_dict(),
                'C_Future_Net_dict': C_future_net.state_dict()
            }
            torch.save(states, os.path.join(opt.savepath,
                                            'model-ourMethod-order2-bestacc-%s.pth' % formatted_now))
            best_acc = acc

        print("auc:{:.4f} | best_auc:{:.4f} | acc:{:.4f} | best_acc:{:.4f} | best_epoch:{}" \
            .format(auc, best_auc, acc, best_acc, best_epoch))

    loss_D_writer.close()
    loss_G_writer.close()
    loss_res_writer.close()
    loss_gen_writer.close()


def main():
    parser = argparse.ArgumentParser()

    # Optimizer config
    parser.add_argument('--epoch', type=int,
                        default=5e3, help='epoch number')
    parser.add_argument('--D_epoch', type=int,
                        default=3, help='epoch number for D_net')
    parser.add_argument('--lr_D', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--lr_G', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--lr2', type=float,
                        default=5e-5, help='learning rate for classifer')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='SGD momentum rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--wd', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--wd2', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--wcl', type=float,
                        default=0.01, help='weight cliping limit for D_net')

    # Checkpoints config
    parser.add_argument('--savepath', type=str, default='./checkpoint/ourmethod/order2')
    parser.add_argument('--extractor_path', type=str,
                        default='./checkpoint/extractor/model-ds128-AUC8346.pth')
    # model-newAUC0.8629-scale=1.pth
    # model-best-scale=0.5.pth

    # Datasets config
    parser.add_argument('--data_root', type=str,
                        default='./dataset',
                        help='path to box train dataset')

    # Hyper-parameter
    parser.add_argument('--margin', type=float,
                        default=0, help='margin h-p for reg_mono')
    parser.add_argument('--lw_gamma', type=float,
                        default=1, help='loss weight for loss_gen')
    parser.add_argument('--lw_alpha', type=float,
                        default=1, help='loss weight for reg_mono')
    parser.add_argument('--lw_lambda', type=float,
                        default=1, help='loss weight for loss_res')
    parser.add_argument('--lw_beta', type=float,
                        default=1, help='loss weight for loss_G_mse')
    parser.add_argument('--lw_theta', type=float,
                        default=1, help='loss weight for loss_future')

    # Model config
    parser.add_argument('--data_size', type=int, default=128, help='3D ADdata resize')
    
    opt = parser.parse_args()
    train(opt)


if __name__ == '__main__':
    main()
