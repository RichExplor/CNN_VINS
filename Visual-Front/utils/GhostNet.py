import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# GhostNet
def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.relu = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        self.relu = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

##################### GhostNet #######################################
class SuperPointNet_GhostNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, width=1.2):
      super(SuperPointNet_GhostNet, self).__init__()

      c4, c5, d1 = 96, 256, 256
      det_h = 65
      self.relu = nn.ReLU(inplace=True)
      print('==> Running SuperPoint_GhostNet.')

      # Network is created here, then will be unpacked into nn.sequential
      self.network_settings = [   # k, t, c, SE, s 
                                  # stage1
                                  [[3,  16,  16, 0, 1]],
                                  # stage2
                                  [[3,  48,  24, 0, 2]],
                                  [[3,  72,  24, 0, 1]],
                                  # stage3
                                  [[5,  72,  40, 0.25, 2]],
                                  [[5, 120,  40, 0.25, 1]],
                                  # stage4
                                  [[3, 240,  80, 0, 1]]
                              ]

      # building first layer
      output_channel = _make_divisible(16 * width, 4)
      input_channel = output_channel

      stages = [
          nn.Sequential(
          nn.Conv2d(1, output_channel, 3, 2, 1, bias=False),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(inplace=True)
          )
      ]

      # building inverted residual blocks
      block = GhostBottleneck
      for cfg in self.network_settings:
          layers = []
          for k, exp_size, c, se_ratio, s in cfg:
              output_channel = _make_divisible(c * width, 4)
              hidden_channel = _make_divisible(exp_size * width, 4)
              layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                            se_ratio=se_ratio))
              input_channel = output_channel
          stages.append(nn.Sequential(*layers))
      
      self.blocks = nn.Sequential(*stages)

      # Detector Head.
      self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
      self.bnPa = torch.nn.BatchNorm2d(c5)
      self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
      self.bnPb = torch.nn.BatchNorm2d(det_h)
      # Descriptor Head.
      self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
      self.bnDa = torch.nn.BatchNorm2d(c5)
      self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
      self.bnDb = torch.nn.BatchNorm2d(d1)
      self.output = None

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x4 = self.blocks(x)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return semi, desc