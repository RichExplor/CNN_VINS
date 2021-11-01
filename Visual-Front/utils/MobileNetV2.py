import torch
import torch.nn as nn
import torch.nn.functional as F


# MobileNet-v2
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(InvertedResidualBlock, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, 1,
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block

def inverted_residual_sequence(in_channels, out_channels, num_units, expansion_factor=6,
                               kernel_size=3,
                               initial_stride=2):
    bottleneck_arr = [
        InvertedResidualBlock(in_channels, out_channels, expansion_factor, kernel_size,
                              initial_stride)]

    for i in range(num_units - 1):
        bottleneck_arr.append(
            InvertedResidualBlock(out_channels, out_channels, expansion_factor, kernel_size, 1))

    return bottleneck_arr


def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        # For efficiency, Dropout is placed before Relu.
        nn.Dropout2d(dropout_prob, inplace=True),
        # Assumption: Relu6 is used everywhere.
        nn.ReLU6(inplace=True)
    )



##################### MobileNet #######################################
class SuperPointNet_MobileNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
      super(SuperPointNet_MobileNet, self).__init__()

      c4, c5, d1 = 96, 256, 256
      det_h = 65
      self.relu = nn.ReLU6(inplace=True)
      print('==> Running SuperPoint_MobileNet.')

      # Network is created here, then will be unpacked into nn.sequential
      self.network_settings = [{'t': -1, 'c': 24, 'n': 1, 's': 2},
                                {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                {'t': 6, 'c': 24, 'n': 1, 's': 2},
                                {'t': 6, 'c': 24, 'n': 1, 's': 1},
                                {'t': 6, 'c': 24, 'n': 1, 's': 2},
                                {'t': 6, 'c': 48, 'n': 1, 's': 1},
                                {'t': 6, 'c': 96, 'n': 1, 's': 1}]

      self.network = [
          conv2d_bn_relu6(
                          in_channels = 1,
                          out_channels = int(self.network_settings[0]['c'] * 1.0),
                          kernel_size = 3,
                          stride = self.network_settings[0]['s'], 
                          dropout_prob = 0.2
                          )
                      ]

      for i in range(1, 7):
          self.network.extend(
              inverted_residual_sequence(
                  in_channels = int(self.network_settings[i - 1]['c'] * 1.0),
                  out_channels = int(self.network_settings[i]['c'] * 1.0),
                  num_units = self.network_settings[i]['n'], 
                  expansion_factor = self.network_settings[i]['t'],
                  kernel_size = 3, 
                  initial_stride = self.network_settings[i]['s'])
                  )
      self.network = nn.Sequential(*self.network)

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
        x4 = self.network(x)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return semi, desc