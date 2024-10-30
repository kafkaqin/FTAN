import torch.nn as nn
import torch
from torch import Tensor
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=3, padding=1, stride= 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
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
        #self.a = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # self.conv0_1 = Partial_conv3(dim, 2, 'split_cat')
        # self.conv0_2 = Partial_conv3(dim, 2, 'split_cat')
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

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

    def forward(self, x):
        u = x.clone()
        #print(self.a)
        attn = self.conv0(x)


        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        print("attn_0",attn_0.shape)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        print("attn_1",attn_1.shape)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        #print(attn_3.shape)
        # attn_4 = self.conv4_1(attn)
        # attn_4 = self.conv4_2(attn_4)

        attn = attn + attn_0 + attn_1 + attn_2 + attn_3
        #attn = attn + attn_0 + attn_1 + attn_2+attn_3+attn_4
        #attn = attn + attn_1 + attn_2+attn_3
        #attn = attn_1 + attn_2 + attn_3

        attn = self.conv4(attn)

        return attn * u

if __name__ == '__main__':
    x = torch.randn(2, 64, 64, 64)
    att = AttentionModule(64)
    print(att(x).shape)
    dim = 64
    n_div = 4
    forward = 'split_cat'

    model = Partial_conv3(dim, n_div, forward)

    x = torch.rand(torch.Size([2, 64, 64, 64]))

    y = model(x)

    print(y.shape)