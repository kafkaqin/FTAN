import torch.nn as nn
import numpy as np
import math
from timm.models.layers import to_2tuple, to_3tuple
import torch
import collections
import torch.nn.functional as F
from models.ChangeFormer import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn.bricks import DropPath
from models.Decoder import ChangeNeXtDecoder
import torch.nn as nn
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead,BaseDecodeHeadNew
from mmseg.ops import resize
from torch import Tensor
import math
from mmseg.models.utils import SelfAttentionBlock
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from models.ChangeFormer import EncoderTransformer_v3
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2,
                      out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.a = dim
        self.rate = 2
        self.num_scales = 4  # 多尺度
        self.freq_adapt_conv = nn.Conv2d(dim, dim, kernel_size=1)  # 自适应频率响应
        self.freq_conv_weight_real = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.freq_conv_weight_imag = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.freq_conv_bias = None  # 如果需要，可以添加偏置
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # self.conv0_1 = Partial_conv3(dim, 2, 'split_cat')
        # self.conv0_2 = Partial_conv3(dim, 2, 'split_cat')
        # self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv0_3 = Partial_conv3(dim, 4, 'split_cat')
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim,
                                 dim, (1, 21),
                                 padding=(0, 10),
                                 groups=dim)
        self.conv3_2 = nn.Conv2d(dim,
                                 dim, (21, 1),
                                 padding=(10, 0),
                                 groups=dim)

        # self.conv4_1 = nn.Conv2d(dim,
        #                          dim, (1, 31),
        #                          padding=(0, 15),
        #                          groups=dim)
        # self.conv4_2 = nn.Conv2d(dim,
        #                          dim, (31, 1),
        #                          padding=(15, 0),
        #                          groups=dim)
        self.conv4 = nn.Conv2d(dim, dim, 1)

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.rate) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, axis=1). \
            repeat_interleave(scale, axis=2)

    # def frequency_mask(self, x):
    #     # 假设输入是 Batch x Channels x Height x Width
    #     # 执行FFT
    #     fft_x = torch.fft.fft2(x)
    #     fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
    #
    #     # 计算频率掩码（简化示例）
    #     # 这里我们计算一个简单的频率权重，可以根据需要调整
    #     freq_weight = torch.log(torch.abs(fft_shifted) + 1)
    #
    #     # 逆FFT回到时域
    #     ifft_shifted = torch.fft.ifftshift(freq_weight, dim=(-2, -1))
    #     mask = torch.fft.ifft2(ifft_shifted)
    #     mask = torch.abs(mask)
    #
    #     return mask

    def frequency_mask(self, x, scale):


        # 执行FFT
        fft_x = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

        # 根据尺度调整频率掩码
        _, _, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        h_start, h_end = center_h - scale, center_h + scale
        w_start, w_end = center_w - scale, center_w + scale
        mask = torch.zeros_like(fft_shifted)
        mask[:, :, h_start:h_end, w_start:w_end] = 1

        masked_freq = self.dwconv_in_freq(fft_shifted * mask)

        # 应用掩码并逆FFT回到时域
        #masked_freq = fft_shifted * mask
        ifft_shifted = torch.fft.ifftshift(masked_freq, dim=(-2, -1))
        mask = torch.fft.ifft2(ifft_shifted)
        mask = torch.abs(mask)

        return mask

    def dwconv_in_freq(self, x_freq):
        # 设置groups等于输入通道数以独立处理每个通道
        groups = x_freq.shape[1]

        # 扩展权重以匹配输入通道数，实现深度可分离卷积
        #weight = self.freq_conv_weight.repeat(groups, 1, 1, 1)
        weight_real = self.freq_conv_weight_real.repeat(groups, 1, 1, 1)
        weight_imag = self.freq_conv_weight_imag.repeat(groups, 1, 1, 1)

        x_freq_real = F.conv2d(x_freq.real, weight_real, self.freq_conv_bias, groups=groups)
        x_freq_imag = F.conv2d(x_freq.imag, weight_imag, self.freq_conv_bias, groups=groups)

        x_freq_processed = torch.complex(x_freq_real, x_freq_imag)
        return x_freq_processed

    def frequency_mask2(self, x, scale):

        # if scale == 0:
        #     return torch.ones_like(x)  # 返回全为 1 的掩码，即不进行掩码操作

        # 执行FFT
        fft_x = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

        # 根据尺度调整频率掩码
        _, _, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        freq_mask = torch.zeros_like(fft_shifted)
        for i in range(H):
            for j in range(W):
                # 计算当前点到中心的距离
                dist = math.sqrt((i - center_h) ** 2 + (j - center_w) ** 2)
                # 根据距离设置掩码值
                freq_mask[:, :, i, j] = self.gaussian_weight(dist, scale)
        # h_start, h_end = center_h - scale, center_h + scale
        # w_start, w_end = center_w - scale, center_w + scale
        # mask = torch.zeros_like(fft_shifted)
        # mask[:, :, h_start:h_end, w_start:w_end] = 1

        # 应用掩码并逆FFT回到时域
        masked_freq = fft_shifted * freq_mask
        ifft_shifted = torch.fft.ifftshift(masked_freq, dim=(-2, -1))
        mask = torch.fft.ifft2(ifft_shifted)
        mask = torch.abs(mask)

        return mask

    def gaussian_weight(self, dist, scale):
        # 使用高斯权重函数计算掩码值
        #sigma = scale * 0.1
        base_sigma = 0.1
        sigma = base_sigma * (2 ** scale)
        if sigma == 0:
            return 1  # 或者返回一个合适的默认值
        return math.exp(-dist ** 2 / (2 * sigma ** 2))

    # def upsample_mask(self, mask, scale):
    #     # In this context, scale is expected to be a power of 2
    #     # Upsample the mask to match the spatial dimensions of the input
    #     N, C, H, W = mask.shape
    #     mask = nn.functional.interpolate(mask.float(), scale_factor=scale, mode='nearest')
    #     return mask

    def forward(self, x):
        u = x.clone()
        # print(x.shape)
        # print(self.a)
        attn = self.conv0(x)

        #freq_mask = self.frequency_mask(attn)
        # print(attn)
        #attn *= freq_mask

        masks = [self.frequency_mask(x, scale) for scale in range(0, self.num_scales + 1)]
        freq_mask = sum(masks) / self.num_scales

        # 应用自适应频率响应
        freq_mask = self.freq_adapt_conv(freq_mask)
        attn *= freq_mask


        # print("attn", attn.shape)
        # print("mask",freq_mask.shape)

        # mask = self.gen_random_mask(x, 0.6)
        # mask = self.upsample_mask(mask, self.rate)
        #
        # mask = mask.unsqueeze(1).type_as(x)
        #
        # attn *= (1. - mask)

        # attn_0 = self.conv0_1(attn)
        # attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn)
        # print("attn_0",attn_0.shape)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        # print("attn_1",attn_1.shape)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        # print(attn_3.shape)
        # attn_4 = self.conv4_1(attn)
        # attn_4 = self.conv4_2(attn_4)

        attn = attn + attn_0 + attn_1 + attn_2 + attn_3
        # attn = attn + attn_0 + attn_1 + attn_2+attn_3+attn_4
        # attn = attn + attn_1 + attn_2+attn_3
        # attn = attn_1 + attn_2 + attn_3

        attn = self.conv4(attn)

        return attn * u





class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        # (shorcut.device)
        x = self.proj_1(x)
        # print(x.device)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        # print(x.device)
        x = x + shorcut
        # print(x.device)
        return x




class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,

                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print(x.device)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        # @print(x.device)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        # print(x.device)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4):
        super(MSCAN, self).__init__()

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
               ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.ModuleList([
                Block(dim=embed_dims[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=drop_rate,
                      drop_path=dpr[cur + j]) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    nn.init.normal_(m,
                                    mean=0,
                                    std=math.sqrt(2.0 / fan_out),
                                    bias=0)
        elif isinstance(pretrained, str):
            self.load_parameters(torch.load(pretrained))

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            # print(H.device)
            for blk in block:
                x = blk(x, H, W)
            # print(x.device)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = nn.bmm(x.transpose(1, 2), bases)
        coef = nn.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.is_training() else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = nn.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R))

        bases = torch.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = nn.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = nn.bmm(coef, nn.bmm(bases.transpose(1, 2), bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = nn.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = nn.bmm(bases, nn.bmm(coef.transpose(1, 2), coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = nn.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = nn.bmm(coef, nn.bmm(bases.transpose(1, 2), bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__()

        self.ham_in = nn.Sequential(
            collections.OrderedDict([('conv',
                                      nn.Conv2d(ham_channels, ham_channels,
                                                1))]))

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = nn.Sequential(
            collections.OrderedDict([('conv',
                                      nn.Conv2d(ham_channels, ham_channels,
                                                1)),
                                     ('gn', nn.GroupNorm(32, ham_channels))]))

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = nn.relu(enjoy)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = nn.relu(x + enjoy)

        return ham


class LightHamHead(nn.Module):
    def __init__(self, in_channels=[64, 160, 256],
                 in_index=[1, 2, 3],
                 channels=256,
                 dropout_ratio=0.1,
                 num_classes=2,
                 align_corners=False, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead, self).__init__(**kwargs)
        self.ham_channels = ham_channels
        self.in_channels = in_channels
        self.in_channels = in_index
        self.channels = channels
        self.num_classes = num_classes

        self.squeeze = nn.Sequential(
            collections.OrderedDict([('conv',
                                      nn.Conv2d(sum(self.in_channels),
                                                self.ham_channels, 1)),
                                     ('gn', nn.GroupNorm(32, ham_channels)),
                                     ('relu', nn.ReLU())]))
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = nn.Sequential(
            collections.OrderedDict([('conv',
                                      nn.Conv2d(self.ham_channels,
                                                self.channels, 1)),
                                     ('gn', nn.GroupNorm(32, ham_channels)),
                                     ('relu', nn.ReLU())]))

    def forward(self, inputs):
        # inputs = self._transform_inputs(inputs)

        inputs = [
            resize(level,
                   size=inputs[0].shape[2:],
                   mode='bilinear',
                   align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.concat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output





# small embed_dims=[64, 128, 256, 512] ,mlp_ratios=[4, 4, 4, 4] depths=[3, 4, 6, 3],
##base embed_dims=[64, 128, 320, 512] ,mlp_ratios=[8, 8, 4, 4] depths=[3, 3, 12, 3],


# large embed_dims=[64, 128, 320, 512],   mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.3,  depths=[3, 5, 27, 3],

class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(ChannelExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(SpatialExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self,
                 channels,
                 num_paths=2,
                 attn_channels=None,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Aggregation_distribution, self).__init__()
        self.num_paths = num_paths  # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


class ChangeNext_diff(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.1, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNext_diff, self).__init__()
        self.channelexchange = ChannelExchange()
        self.spatialexchange = SpatialExchange()
        self.ad = Aggregation_distribution(128)
        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.num_stages = num_stages
        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=self.embed_dims,
                             mlp_ratios=self.mlp_ratios,
                             drop_rate=self.drop_rate,
                             drop_path_rate=self.drop_path_rate,
                             depths=self.depths,
                             num_stages=self.num_stages)
        self.Denc_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                             output_nc=output_nc,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16])

    # def ChannelExchange(self, x1, x2):
    #     N, c, h, w = x1.shape
    #
    #     exchange_map = torch.arange(c) % self.p == 0
    #     exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
    #
    #     out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
    #     out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
    #     out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
    #     out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
    #     out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
    #
    #     return out_x1, out_x2
    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        x1 = []
        x2 = []
        fx1_1, fx1_2, fx1_3, fx1_4 = fx1
        fx2_1, fx2_2, fx2_3, fx2_4 = fx2
        # a_2,b_2 = self.channelexchange(fx1_2,fx2_2)
        # a_3, b_3 = self.spatialexchange(fx1_3, fx2_3)
        a_2, b_2 = self.ad(fx1_2, fx2_2)
        # a_3, b_3 = self.ad(fx1_3, fx2_3)
        # print(a_2.shape) #128
        # print(a_3.shape) #320
        x1.extend([fx1_1, a_2, fx1_3, fx1_4])
        x2.extend([fx2_1, b_2, fx2_3, fx2_4])
        # print(fx1.shape)
        cp = self.Denc_x2(x1, x2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp


class ChangeNextV1(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512], decoder_softmax=False,
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.2, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNextV1, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims

        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=embed_dims,
                             mlp_ratios=mlp_ratios,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             depths=depths,
                             num_stages=num_stages)
        self.Decode = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                            align_corners=False,
                                            in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                            output_nc=output_nc,
                                            decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # print(fx1.shape)

        cp = self.Decode(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp

class ChangeNextV2(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512], decoder_softmax=False,
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.2, depths=[3, 3, 4, 3], num_stages=4,
                 ):
        super(ChangeNextV2, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims
        self.drop_rate = drop_rate
        self.attn_drop = 0.1
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 4, 8], mlp_ratios=mlp_ratios, qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        self.Decode = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=1,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc, decoder_softmax=False)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # print(fx1.shape)

        cp = self.Decode(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp

class ChangeNextV3(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.,
                 drop_path_rate=0.1, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNextV3, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims

        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=embed_dims,
                             mlp_ratios=mlp_ratios,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             depths=depths,
                             num_stages=num_stages)
        self.Decode = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=64,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc,
                                        decoder_softmax=False)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # for i in range(4):
        #     print(fx1[i].shape)

        cp = self.Decode(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp

class ChangeFormer_decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512], decoder_softmax=False,
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.2, depths=[3, 3, 4, 3], num_stages=4,
                 ):
        super(ChangeFormer_decoder, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims
        self.drop_rate = drop_rate
        self.attn_drop = 0.1
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.Tenc_x2 =  EncoderTransformer_v3(img_size=256, patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])

        self.Decode =  ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=1,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc, decoder_softmax=False)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # print(fx1.shape)

        cp = self.Decode(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp


class ChangeNext_decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.,
                 drop_path_rate=0.1, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNext_decoder, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims

        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=embed_dims,
                             mlp_ratios=mlp_ratios,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             depths=depths,
                             num_stages=num_stages)
        self.Decode = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=1,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc, decoder_softmax=False)

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        # for i in range(4):
        #     print(fx1[i].shape)

        cp = self.Decode(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp


if __name__ == '__main__':
    # x1 = torch.randn(2, 3, 256, 256)
    # # x2 = torch.randn(6, 3, 256, 256).to(device)
    # net = ChangeNextV1()
    # print(net)
    # print(net(x1, x1)[-1].shape)
    # total = sum([param.nelement() for param in net.parameters()])
    #
    # print("Number of parameter: %.2fM" % (total / 1e6))
    #
    # net = ChangeNextV2()
    # print(net)
    # print(net(x1, x1)[-1].shape)
    # total = sum([param.nelement() for param in net.parameters()])
    #
    # print("Number of parameter: %.2fM" % (total / 1e6))

    net = ChangeNextV3().to('cuda')
    # print(net)
    # print(net(x1, x1)[-1].shape)
    # total = sum([param.nelement() for param in net.parameters()])

    #print("Number of parameter: %.2fM" % (total / 1e6))

    from torchsummary import summary


    summary(net, [(3, 256, 256),(3, 256, 256)])
    # input_nc = 3
    # output_nc = 2
    # embed_dim = 256
    # embed_dims = [64, 128, 320, 512],
    # mlp_ratios = [8, 8, 4, 4]
    # drop_rate = 0.,
    # drop_path_rate = 0.1
    # depths = [3, 3, 12, 3]
    # num_stages = 4
    # y = MSCAN(in_chans=input_nc,
    #                          embed_dims=embed_dims,
    #                          mlp_ratios=mlp_ratios,
    #                          drop_rate=drop_rate,
    #                          drop_path_rate=drop_path_rate,
    #                          depths=depths,
    #                          num_stages=num_stages)
    # # fx1=[torch.randn(6, 64, 64, 64),torch.randn(6, 128, 32, 32),torch.randn(6, 320, 16, 16),torch.randn(6, 512, 8, 8)]
    # net = y(x1)
    # for i in range(4):
    #     print(net[i].shape)

    # for i in net(x1,x1):
    #     print(i.shape)
    # print(net(x1,x2).shape)
    # net = EncoderTransformer_v3()
    # for i in net(x1,x2):
    #     print(i.shape)
# print(net(x1,x2).shape)
# print(net(x1)[0].shape)