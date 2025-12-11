from loss import DiceLoss, curvature_loss
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import init_weights, SAM_get_pl_label
from torch.optim import lr_scheduler
from torch.nn import init
import cv2
import os
from skimage.io import imread, imsave

def get_scheduler(optimizer):
    """
    获取学习率调度器，采用StepLR，每4个epoch衰减一次，gamma为0.9。
    optimizer: 优化器对象
    返回: 调度器对象
    """
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重。
    参数：
        net (network)   -- 需要初始化的网络
        init_type (str) -- 初始化方法名称: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 缩放因子，适用于normal、xavier和orthogonal。
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_largest_component(image):
    """
    从二维或三维二值图像中获取最大连通区域。
    参数：
        image: numpy数组，二值图像
    返回：最大连通区域的掩码
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output


class UNetConvBlock(nn.Module):
    """
    UNet编码器/解码器中的基本卷积块，包含两层卷积、BN和LeakyReLU。
    """
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        构造函数
        in_channels: 输入通道数
        out_channels: 输出通道数
        dropout_p: dropout概率
        """
        super(UNetConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        """
        前向传播
        x: 输入特征
        返回: 输出特征
        """
        return self.conv_conv(x)
 
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, dropout_p):
        """
        上采样块，包含上采样和卷积块。
        in_chans: 输入通道数
        out_chans: 输出通道数
        up_mode: 上采样方式 ('upconv' 或 'upsample')
        dropout_p: dropout概率
        """
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTransposed2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode=='upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, dropout_p)

    def centre_crop(self, layer, target_size):
        """
        中心裁剪特征图以匹配目标尺寸。
        layer: 输入特征图
        target_size: 目标尺寸 (高,宽)
        返回: 裁剪后的特征图
        """
        _,_,layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        """
        前向传播，上采样并与跳跃连接特征拼接。
        x: 当前特征
        bridge: 跳跃连接特征
        返回: 输出特征
        """
        up = self.up(x)
        crop1 = self.centre_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out

class Encoder(nn.Module):
    """
    UNet编码器部分，包含多层下采样卷积块。
    """
    def __init__(self,
        in_chns,
        n_classes,
        ft_chns,
        dropout_p
        ):
        """
        构造函数
        in_chns: 输入通道数
        n_classes: 类别数
        ft_chns: 每层特征通道数列表
        dropout_p: dropout概率列表
        """
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.down_path = nn.ModuleList()
        self.down_path.append(UNetConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))

    def forward(self, x):
        """
        前向传播，返回每层特征和最后的潜在特征。
        x: 输入特征
        返回: 跳跃连接特征列表, 最后输出特征
        """
        blocks=[]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x ,2)
        return blocks, x

class aux_dec5oder(nn.Module):
    """
    辅助解码器（疑似拼写错误，建议aux_Decoder），用于UNet解码部分。
    """
    def __init__(self, 
        in_chns,
        n_classes,
        ft_chns,
        dropout_p,
        up_mode):
        """
        构造函数
        in_chns: 输入通道数
        n_classes: 类别数
        ft_chns: 每层特征通道数列表
        dropout_p: dropout概率列表
        up_mode: 上采样方式
        """
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(self.ft_chns[4], self.ft_chns[3], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[3], self.ft_chns[2], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[2], self.ft_chns[1], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[1], self.ft_chns[0], up_mode, self.dropout[0]))
        self.last = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)

    def forward(self, x, blocks):
        """
        前向传播，解码并输出分割结果。
        x: 潜在特征
        blocks: 跳跃连接特征列表
        返回: 分割结果
        """
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i -1])
        return self.last(x)


class aux_Decoder(nn.Module):
    """
    辅助解码器，用于UNet解码部分。
    """
    def __init__(self, 
        in_chns,
        n_classes,
        ft_chns,
        dropout_p,
        up_mode):
        """
        构造函数
        in_chns: 输入通道数
        n_classes: 类别数
        ft_chns: 每层特征通道数列表
        dropout_p: dropout概率列表
        up_mode: 上采样方式
        """
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(self.ft_chns[4], self.ft_chns[3], up_mode, self.dropout[1]))
        self.up_path.append(UNetUpBlock(self.ft_chns[3], self.ft_chns[2], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[2], self.ft_chns[1], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[1], self.ft_chns[0], up_mode, self.dropout[0]))
        self.last = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)

    def forward(self, x, blocks):
        """
        前向传播，解码并输出分割结果。
        x: 潜在特征
        blocks: 跳跃连接特征列表
        返回: 分割结果
        """
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i -1])
        return self.last(x)

class UNet(nn.Module):
    """
    UNet主模型，包含编码器、辅助解码器、优化器、损失函数等。
    """
    def __init__(
        self, params
    ):
        """
        构造函数
        params: 配置参数字典
        """
        super(UNet, self).__init__()
        lr = params['train']['lr']
        dataset = params['train']['dataset']

        if dataset == 'mms':
            in_chns = params['network']['in_chns']
            n_classes = params['network']['n_classes_mms']
            ft_chns = params['network']['ft_chns_mms']
        dropout_p = params['network']['dropout_p']
        up_mode = params['network']['up_mode']
        
        self.enc = Encoder(in_chns,n_classes,ft_chns,dropout_p)
        self.aux_dec1 = aux_Decoder(in_chns,n_classes,ft_chns,dropout_p,up_mode)

        # setting the optimzer
        opt = 'adam'
        if opt == 'adam':
            self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=lr,betas=(0.9,0.999))
            self.aux_dec1_opt = torch.optim.Adam(self.aux_dec1.parameters(),lr=lr,betas=(0.5,0.999))
        elif opt == 'SGD':
            self.enc_opt = torch.optim.SGD(self.enc.parameters(),lr=lr,momentum=0.9)

        # setting the using of loss  
        self.segloss = DiceLoss(n_classes).to(torch.device('cuda'))
        
        self.enc_opt_sch = get_scheduler(self.enc_opt)
        self.dec_1_opt_sch = get_scheduler(self.aux_dec1_opt)


    def initialize(self):
        """
        初始化编码器和解码器的权重。
        """
        init_weights(self.enc)
        init_weights(self.aux_dec1)


    def update_lr(self):
        """
        更新学习率。
        """
        self.enc_opt_sch.step()
        self.dec_1_opt_sch.step()


    def forward(self, x):
        """
        前向传播，支持5维输入（批量、通道、深度、高、宽），输出分割概率。
        x: 输入图像
        返回: 分割概率图
        """
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        A_1 = x
        blocks1, latent_A1 = self.enc(A_1)
        self.aux_seg_1 = self.aux_dec1(latent_A1, blocks1).softmax(1)
        return self.aux_seg_1
        
        
    def train_source(self,imagesa,labelsa):
        """
        源域训练，更新编码器和解码器参数。
        imagesa: 输入图像
        labelsa: 标签
        返回: 分割损失
        """
        self.imgA = imagesa
        self.labA = labelsa
        self.forward(self.imgA)
        # 更新编码器和解码器
        self.enc_opt.zero_grad()
        self.aux_dec1_opt.zero_grad()

        # 原代码没有传ent_dice_weight_map这个参数，导致报错
        ent_dice_weight_map = torch.ones_like(self.labA, dtype=self.aux_seg_1.dtype)

        seg_loss_B = self.segloss(self.aux_seg_1,self.labA, ent_dice_weight_map, one_hot = True)
        seg_loss_B.backward()
        self.loss_seg = seg_loss_B.item()
        self.enc_opt.step()
        self.aux_dec1_opt.step()
        return self.loss_seg


    def test_with_name(self,imagesb):
        """
        测试推理，支持5维输入。
        imagesb: 输入图像
        返回: 分割概率图
        """
        x_shape = list(imagesb.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(imagesb, 1, 2)
            x = torch.reshape(x, new_shape)
            imagesb = x 
        self.forward(imagesb)
        output = self.aux_seg_1
        return output
    def generate_sam_pl(self, images, B_name, predictor, all_result_path, num_classes, sample_times, target):
        """
        生成SAM伪标签及权重图。
        images: 输入图像
        B_name: 图像名称
        predictor: SAM预测器
        all_result_path: 结果保存路径
        num_classes: 类别数
        sample_times: 采样次数
        target: 目标域标识
        """
        self.forward(images)
        pred_aux1 = self.aux_seg_1.cpu().detach().numpy()
        output_dec1 = pred_aux1.copy()
        mask_input = pred_aux1.copy()

        output = np.argmax(output_dec1, axis=1)
        dec_output = np.zeros((output_dec1.shape[0], num_classes, 256, 256), dtype=np.uint8)
        for c in range(num_classes):
            dec_output[:, c, :, :] = (output.copy() == c)
        argmax_output = dec_output.copy()

        save_path_pl = all_result_path + '/pl/' + str(target) + '/'
        save_path_pl_weight = all_result_path + '/pl_weight/' + str(target) + '/'
        if not os.path.exists(save_path_pl):
            os.makedirs(save_path_pl)
        if not os.path.exists(save_path_pl_weight):
            os.makedirs(save_path_pl_weight)

        pseudo_label_sam, all_depth_weight_entropy = SAM_get_pl_label(
            images, argmax_output.copy(), predictor, mask_input, num_classes, sample_times)

        self.dice_weight_map = all_depth_weight_entropy.copy()
        self.sam_pseudo_label = pseudo_label_sam.copy()
        self.pseudo_label_weight = all_depth_weight_entropy.copy()
        SAM_pl_to_save = pseudo_label_sam.copy()

        # np.save(save_path_pl + str(B_name[0][:-7]) + '.npy', SAM_pl_to_save.copy())
        # np.save(save_path_pl_weight + str(B_name[0][:-7]) + '.npy', all_depth_weight_entropy.copy())

    def domain_adaptation(self, B, curve_weight):
        """
        域自适应训练，结合伪标签和曲率损失。
        B: 输入图像
        curve_weight: 曲率损失权重
        返回: dice损失和曲率损失
        """
        self.enc_opt.zero_grad()
        self.aux_dec1_opt.zero_grad()
        device = B.device
        ent_dice_weight_map = torch.from_numpy(self.pseudo_label_weight.copy()).float().to(device)
        pseudo_lab = torch.from_numpy(self.sam_pseudo_label.copy()).float().to(device)
        eara1 = self.aux_seg_1
        diceloss = self.segloss(eara1, pseudo_lab, ent_dice_weight_map, False)
        diceloss = diceloss
        curve_loss_1 = curvature_loss(self.aux_seg_1)
        curve_loss_ = curve_loss_1 * curve_weight
        all_loss = diceloss + curve_loss_
        all_loss.backward()
        self.enc_opt.step()
        self.aux_dec1_opt.step()
        diceloss = diceloss.item()
        curve_loss_ = curve_loss_.item()
        return diceloss, curve_loss_
