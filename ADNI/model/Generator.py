import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, dp=0.0):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dp)
        k_size = 3
        self.Gates = nn.Conv3d(input_size + hidden_size, 4 * hidden_size, kernel_size=k_size, padding=k_size // 2)

    def forward(self, input_, pre_state=None):
        # pre_state is previous LSTM state

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty pre_state, if None is provided
        if pre_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden, prev_cell = (
                torch.zeros(state_size).cuda(),
                torch.zeros(state_size).cuda()
            )
        else:
            prev_hidden, prev_cell = pre_state

        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return self.dropout(hidden), cell


class Generator_LSTM_1(nn.Module):
    def __init__(self,
                 final_tanh=0,
                 in_channel=64,
                 mid_feature_channel=512,
                 mid_z_channel=256,
                 RNN_hidden=None,
                 drop_rate=0.0):
        super(Generator_LSTM_1, self).__init__()
        if RNN_hidden is None:
            RNN_hidden = [512]
        self.final_tanh = final_tanh
        self.in_channel = in_channel
        self.mid_feature_channel=mid_feature_channel
        self.mid_z_channel=mid_z_channel
        self.RNN_hidden = RNN_hidden
        self.dropout = nn.Dropout(drop_rate)
        # for feature_map
        self.conv_blocks_feature = nn.Sequential(
            # 128*64*64 - 256*32*32
            nn.Conv3d(in_channels=self.in_channel, out_channels=self.mid_feature_channel//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=self.mid_feature_channel//2),
            nn.ReLU(inplace=True),

            # 256*32*32 - 512*32*32
            nn.Conv3d(in_channels=self.mid_feature_channel//2, out_channels=self.mid_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=self.mid_feature_channel),
            nn.ReLU(inplace=True),
        )
        # for z1
        self.conv_blocks1 = nn.Sequential(
            # batch_size*100*1*1 to batch_size*512*32*32

            # ConvTranspose3d tips: up2:4-2-1 up4:4-4-0 or 8-4-2
            # batch_size*100*1*1 to batch_size*1024*4*4
            nn.ConvTranspose3d(in_channels=100, out_channels=self.mid_z_channel*4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=self.mid_z_channel*4),
            nn.ReLU(inplace=True),

            # batch_size*1024*4*4 to batch_size*512*8*8
            nn.ConvTranspose3d(in_channels=self.mid_z_channel*4, out_channels=self.mid_z_channel*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=self.mid_z_channel*2),
            nn.ReLU(inplace=True),

            # batch_size*1024*8*8 to batch_size*512*16*16
            nn.ConvTranspose3d(in_channels=self.mid_z_channel*2, out_channels=self.mid_z_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=self.mid_z_channel),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=self.mid_z_channel, out_channels=self.mid_z_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(num_features=self.mid_z_channel),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            # up2:4-2-1 up4:4-4-0 8-4-2
            # 32*32 -> 64*64
            nn.ConvTranspose3d(in_channels=self.RNN_hidden[-1], out_channels=256, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 64*64 -> 64*64
            nn.ConvTranspose3d(in_channels=256, out_channels=self.in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=self.in_channel),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        # for grad
        self.LSTM_cell = []
        input_dim = self.mid_z_channel+self.mid_feature_channel
        for hidden_num in self.RNN_hidden:
            self.LSTM_cell.append(ConvLSTMCell(input_dim, hidden_num))
            input_dim = hidden_num
        self.LSTM_cell = nn.ModuleList(self.LSTM_cell)

    def forward(self, feature1, z1, grad_1, grad_2):
        # 这里原代码中G的输入是batch中的一个sample(因为不同sample的grad不一样)，但是新代码将sample-wise操作改到生成器中了。
        batch_size = feature1.size(0)
        mid_feature_img_1 = self.conv_blocks_feature(feature1)
        # print(mid_feature_img_1.shape) # B, mid_feature_channel, input_H // 2, input_W // 2, input_D // 2
        mid_feature_z_1 = self.conv_blocks1(z1)
        mid_feature_z_1 = F.interpolate(mid_feature_z_1,
                                        size=(mid_feature_img_1.size(2),
                                              mid_feature_img_1.size(3),
                                              mid_feature_img_1.size(4)))
        # print(mid_feature_z_1.shape) # B, mid_z_channel, input_H // 2, input_W // 2, input_D // 2
        x1 = torch.cat((mid_feature_z_1, mid_feature_img_1), 1)
        # print(x1.shape) # B, mid_feature_channel + mid_z_channel, input_H // 2, input_W // 2, input_D // 2

        hidden = None
        hidden_0 = None

        for idx in range(batch_size):
            # 将batch进行sample-wise的LSTM操作
            batch = x1[idx].unsqueeze(0)
            for delta in range(int(grad_2[idx]) - int(grad_1[idx])):
                # 这里进行时序不同次数的LSTM
                if len(self.RNN_hidden) == 1:
                    # One ConvLSTM layer
                    hidden = self.LSTM_cell[0](batch, hidden)
                elif len(self.RNN_hidden) == 2:
                    # Two ConvLSTM layer
                    hidden_0 = self.LSTM_cell[0](batch, hidden_0)
                    hidden = self.LSTM_cell[1](hidden_0[0], hidden)
            
            # 这里是把得到的每个特征图聚合成batch
            if idx == 0:
                featuremap = self.conv_blocks2(hidden[0])
            else:
                featuremap = torch.cat((featuremap, self.conv_blocks2(hidden[0])), dim=0)
        
        if self.final_tanh:
            featuremap = torch.tanh(featuremap)
        return featuremap


class Generator_LSTM_2(nn.Module):
    def __init__(self, final_tanh=0, in_channel=64, RNN_hidden=None, drop_rate=0.0):
        super(Generator_LSTM_2, self).__init__()
        if RNN_hidden is None:
            RNN_hidden = [512]
        self.final_tanh = final_tanh
        self.in_channel = in_channel
        self.RNN_hidden = RNN_hidden
        self.dropout = nn.Dropout(drop_rate)
        # for feature_map
        self.conv_blocks_feature = nn.Sequential(
            # 128*64*64 - 5256*32*32
            nn.Conv3d(in_channels=self.in_channel, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),

            # 256*32*32 - 512*32*32
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
        )
        # for z1
        self.conv_blocks1 = nn.Sequential(
            # batch_size*100*1*1 to batch_size*512*32*32

            # ConvTranspose3d tips: up2:4-2-1 up4:4-4-0 or 8-4-2
            # batch_size*100*1*1 to batch_size*1024*4*4
            nn.ConvTranspose3d(in_channels=100, out_channels=2048, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=2048),
            nn.ReLU(inplace=True),

            # batch_size*1024*4*4 to batch_size*512*8*8
            nn.ConvTranspose3d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=1024),
            nn.ReLU(inplace=True),

            # batch_size*1024*8*8 to batch_size*512*16*16
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks2_2_by_1 = nn.Sequential(
            # up2:4-2-1 up4:4-4-0 8-4-2
            # 32*32 -> 64*64
            nn.ConvTranspose3d(in_channels=self.RNN_hidden[-1], out_channels=256, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 64*64 -> 64*64
            nn.ConvTranspose3d(in_channels=256, out_channels=self.in_channel, kernel_size=3, stride=1, padding=1),
        )
        self.conv_blocks2_3_by_12 = nn.Sequential(
            # up2:4-2-1 up4:4-4-0 8-4-2
            # 32*32 -> 64*64
            nn.ConvTranspose3d(in_channels=self.RNN_hidden[-1], out_channels=256, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 64*64 -> 64*64
            nn.ConvTranspose3d(in_channels=256, out_channels=self.in_channel, kernel_size=3, stride=1, padding=1),
        )
        # for grad
        self.LSTM_cell = []
        input_dim = 1024
        for hidden_num in self.RNN_hidden:
            self.LSTM_cell.append(ConvLSTMCell(input_dim, hidden_num))
            input_dim = hidden_num
        self.LSTM_cell = nn.ModuleList(self.LSTM_cell)

    def forward(self, feature1, feature2, z1, z2, grad_1, grad_2, grad_3):
        # 这里原代码中G的输入是batch中的一个sample，但是新代码将sample-wise操作改到生成器中了。
        # grad_1 = order1的time; grad_2 = order2的time; grad_3 = orderT的time
        batch_size = feature1.size(0)
        # 处理order1的数据
        mid_feature_img_1 = self.conv_blocks_feature(feature1)
        mid_feature_z_1 = self.conv_blocks1(z1)
        mid_feature_z_1 = F.interpolate(mid_feature_z_1,
                                        size=(mid_feature_img_1.size(2),
                                              mid_feature_img_1.size(3),
                                              mid_feature_img_1.size(4)))
        x1 = torch.cat((mid_feature_z_1, mid_feature_img_1), 1)
        # 处理order2的数据
        mid_feature_img_2 = self.conv_blocks_feature(feature2)
        mid_feature_z_2 = self.conv_blocks1(z2)
        mid_feature_z_2 = F.interpolate(mid_feature_z_2,
                                        size=(mid_feature_img_2.size(2),
                                              mid_feature_img_2.size(3),
                                              mid_feature_img_2.size(4)))
        x2 = torch.cat((mid_feature_z_2, mid_feature_img_2), 1)
        # 初始化hidden状态
        hidden = None
        hidden_0 = None

        for idx in range(batch_size):
            # 将batch进行sample-wise的LSTM操作
            sample_order1 = x1[idx].unsqueeze(0)
            sample_order2 = x2[idx].unsqueeze(0)
            
            for delta in range(int(grad_2[idx]) - int(grad_1[idx])):
                # 这里进行时序1到时序2的LSTM
                if len(self.RNN_hidden) == 1:
                    # One ConvLSTM layer
                    hidden = self.LSTM_cell[0](sample_order1, hidden)
                elif len(self.RNN_hidden) == 2:
                    # Two ConvLSTM layer
                    hidden_0 = self.LSTM_cell[0](sample_order1, hidden_0)
                    hidden = self.LSTM_cell[1](hidden_0[0], hidden)
            # 这里是把得到的每个由order1生成order2的特征图聚合成batch
            if idx == 0:
                featuremap_2_by_1 = self.conv_blocks2_2_by_1(hidden[0])
            else:
                featuremap_2_by_1 = torch.cat((featuremap_2_by_1, self.conv_blocks2_3_by_12(hidden[0])), dim=0)
                
            for delta in range(int(grad_3[idx]) - int(grad_2[idx])):
                # 这里进行时序2到时序3的LSTM
                if len(self.RNN_hidden) == 1:
                    # One ConvLSTM layer
                    hidden = self.LSTM_cell[0](sample_order2, hidden)
                elif len(self.RNN_hidden) == 2:
                    # Two ConvLSTM layer
                    hidden_0 = self.LSTM_cell[0](sample_order2, hidden_0)
                    hidden = self.LSTM_cell[1](hidden_0[0], hidden)
            # 这里是把得到的每个由order1，order2生成order3的特征图聚合成batch
            if idx == 0:
                featuremap_3_by_12 = self.conv_blocks2_3_by_12(hidden[0])
            else:
                featuremap_3_by_12 = torch.cat((featuremap_3_by_12, self.conv_blocks2_3_by_12(hidden[0])), dim=0)


        if self.final_tanh:
            featuremap_2_by_1 = torch.tanh(featuremap_2_by_1)
            featuremap_3_by_12 = torch.tanh(featuremap_3_by_12)
        return featuremap_2_by_1, featuremap_3_by_12

if __name__ == '__main__':
    grad_1 = torch.tensor([0, 0, 0, 0]).cuda()
    grad_2 = torch.tensor([2, 1, 2, 3]).cuda()
    grad_3 = torch.tensor([4, 2, 4, 6]).cuda()
    
    
    feature1 = torch.rand(4, 64, 32, 32, 32).cuda()
    z1 = torch.randn(4, 100, 1, 1, 1).cuda()
    G_oreder1 = Generator_LSTM_1().cuda()
    out_order1 = G_oreder1(feature1, z1, grad_1, grad_2)
    
    
    feature1 = torch.rand(4, 64, 23, 27, 23).cuda()
    feature2 = torch.rand(4, 64, 23, 27, 23).cuda()
    z2 = torch.randn(4, 100, 1, 1, 1).cuda()
    G_oreder2 = Generator_LSTM_2().cuda()
    out_order2 = G_oreder2(feature1, feature2, z1, z2, grad_1, grad_2, grad_3)
    
    # print(out_order1.shape)
    # print(out_order2[1].shape)
