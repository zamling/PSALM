import torch
import torch.nn as nn
import re
import math
BatchNorm2d=nn.BatchNorm2d
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockLayerNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,norm_shape, stride=1, downsample=None, dcn=None):
        super(BasicBlockLayerNorm, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.LayerNorm(norm_shape)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            raise NotImplementedError
        self.bn2 = nn.LayerNorm(norm_shape)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            raise NotImplementedError
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockWOnorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlockWOnorm, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            raise NotImplementedError
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetWOnorm(nn.Module):
    def __init__(self,
                 dcn=None,
                 out_dim=4096):
        print('using resnet without batchnorm')
        self.dcn = dcn
        self.inplanes = 256
        super(ResNetWOnorm, self).__init__()
        self.layer1 = self._make_layer(
            BasicBlockWOnorm, 1024, 1, stride=2, dcn=dcn)
        self.layer2 = self._make_layer(
            BasicBlockWOnorm, 4096, 1, stride=2, dcn=dcn)
        self.fc = nn.Linear(4096, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.reshape(x.shape[0],x.shape[1],-1)
        x = x.permute(0,2,1)


        x = self.fc(x)

        return x

class ResNetLayerNorm(nn.Module):
    def __init__(self,
                 dcn=None,
                 out_dim=4096):
        print('using resnet with layernorm')
        self.dcn = dcn
        self.inplanes = 256
        h,w = 64,64
        super(ResNetLayerNorm, self).__init__()
        self.layer1 = self._make_layer(
            BasicBlockLayerNorm, 1024, 1,[1024,32,32], stride=2, dcn=dcn)
        self.layer2 = self._make_layer(
            BasicBlockLayerNorm, 4096, 1,[4096,16,16], stride=2, dcn=dcn)
        self.fc = nn.Linear(4096, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks,norm_shape, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.LayerNorm(norm_shape),
            )

        layers = []
        layers.append(block(self.inplanes, planes,norm_shape,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.reshape(x.shape[0],x.shape[1],-1)
        x = x.permute(0,2,1)


        x = self.fc(x)

        return x

class ResNet(nn.Module):
    def __init__(self,
                 dcn=None,
                 out_dim=4096):
        self.dcn = dcn
        self.inplanes = 256
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(
            BasicBlock, 1024, 1, stride=2, dcn=dcn)
        self.layer2 = self._make_layer(
            BasicBlock, 4096, 1, stride=2, dcn=dcn)
        self.fc = nn.Linear(4096, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.reshape(x.shape[0],x.shape[1],-1)
        x = x.permute(0,2,1)


        x = self.fc(x)

        return x

class ResNetSwin(nn.Module):
    def __init__(self,
                 dcn=None,
                 input_dim=1024,
                 out_dim=4096):
        self.dcn = dcn
        self.inplanes = input_dim
        super(ResNetSwin, self).__init__()
        self.layer1 = self._make_layer(
            BasicBlock, 2048, 1, stride=2, dcn=dcn)
        self.fc = nn.Linear(2048, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)

        x = x.reshape(x.shape[0],x.shape[1],-1)
        x = x.permute(0,2,1)


        x = self.fc(x)

        return x

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'conv':
        with_norm = getattr(config, 'with_norm', True)
        with_layernorm = getattr(config, 'with_layernorm', True)
        out_dim = getattr(config,'projector_outdim',4096)
        # print(out_dim)
        if with_layernorm:
            return ResNetLayerNorm(out_dim=out_dim)
        if with_norm:
            return ResNet(out_dim=out_dim)
        else:
            return ResNetWOnorm(out_dim=out_dim)
    if projector_type == 'swin_conv':
        out_dim = getattr(config,'projector_outdim',4096)
        input_dim = getattr(config,'mm_input_embeds',1024)
        return ResNetSwin(input_dim=input_dim,out_dim=out_dim)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.mm_projector_type = 'conv'
            self.with_layernorm = True
            self.with_norm = False
    config = Config()
    net = build_vision_projector(config)
    image = torch.randn((4,256,64,64))
    print(net)
    print(net(image).shape)