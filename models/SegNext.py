import torch.nn as nn
import numpy as np
import math
from timm.models.layers import to_2tuple,to_3tuple
import torch
import collections
import torch.nn.functional as F
from models.ChangeFormer import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn.bricks import DropPath
from models.Decoder import ChangeNeXtDecoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)


        self.conv1_1 = nn.Conv2d(dim, dim, (3, 7), padding=(1, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 3), padding=(3, 1), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (3, 11), padding=(1, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 3), padding=(5, 1), groups=dim)

        self.conv3_1 = nn.Conv2d(dim,
                                 dim, (3, 21),
                                 padding=(1, 10),
                                 groups=dim)
        self.conv3_2 = nn.Conv2d(dim,
                                 dim, (21, 3),
                                 padding=(10, 1),
                                 groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        # attn_0 = self.conv0_1(attn)
        # attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        #print(attn_1.shape)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)


        #attn = attn + attn_0 + attn_1 + attn_2+attn_3
        attn = attn + attn_1 + attn_2+attn_3

        attn = self.conv3(attn)

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
        #(shorcut.device)
        x = self.proj_1(x)
        #print(x.device)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        #print(x.device)
        x = x + shorcut
        #print(x.device)
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
        #print(x.device)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        #@print(x.device)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        #print(x.device)
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
            #print(H.device)
            for blk in block:
                x = blk(x, H, W)
            #print(x.device)
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
                     align_corners=False,ham_channels=512, ham_kwargs=dict(), **kwargs):
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
        #inputs = self._transform_inputs(inputs)

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


class EncoderTransformer_v3(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4):
        super().__init__()

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

        # patch embedding definitions


        self.apply(self._init_weights)

    def _init_weights(self, pretrained=None):
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

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

#small embed_dims=[64, 128, 256, 512] ,mlp_ratios=[4, 4, 4, 4] depths=[3, 4, 6, 3],
##base embed_dims=[64, 128, 320, 512] ,mlp_ratios=[8, 8, 4, 4] depths=[3, 3, 12, 3],


#large embed_dims=[64, 128, 320, 512],   mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.3,  depths=[3, 5, 27, 3],
class SegNext_diffV1(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False,embed_dim=256, embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                             drop_path_rate=0.2, depths=[3,3,12,3], num_stages=4):
        super(SegNext_diffV1, self).__init__()
        self.embed_dims=embed_dims
        self.embedding_dim = embed_dim
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
                 depths= self.depths,
                 num_stages=self.num_stages)
        self.Denc_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc,
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16])
    def forward(self, x1, x2):

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        #print(fx1.shape)
        cp = self.Denc_x2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
class ChangeNext_decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False,embed_dim=256,):
        super(ChangeNext_decoder, self).__init__()
        self.embed_dims=[64, 128, 320, 512]
        self.embedding_dim = embed_dim
        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                 embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 depths=[3, 3, 12, 3],
                 num_stages=4)
        self.Denc_x2 = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True, trans_depth=1,
                 att_type="XCA",in_channels=[64, 128, 320, 512],in_index=[0, 1, 2, 3],channels=256,
        dropout_ratio=0.1,num_classes=2,input_transform='multiple_select',align_corners=False,feature_strides=[2, 4, 8, 16],embedding_dim=256,output_nc=2,decoder_softmax=False)
    def forward(self, x1, x2):

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        #print(fx1.shape)
        cp = self.Denc_x2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp
if __name__ == '__main__':
    x1= torch.randn(6,3,256,256).to(device)
    #x2 = torch.randn(6, 3, 256, 256).to(device)
    net = SegNext_diffV1().to(device)
    print(net)
    # print(net(x1,x1)[-1].shape)
    # total = sum([param.nelement() for param in net.parameters()])
    #
    # print("Number of parameter: %.2fM" % (total / 1e6))
    # for i in net(x1,x1):
    #     print(i.shape)
    #print(net(x1,x2).shape)
    #net = EncoderTransformer_v3()
    # for i in net(x1,x2):
    #     print(i.shape)
    # print(net(x1,x2).shape)
    #print(net(x1)[0].shape)