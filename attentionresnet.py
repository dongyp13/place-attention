import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    def __init__(self, inplanes, planes):
        super(Attention, self).__init__()
        self.planes = planes
        self.query = nn.Linear(inplanes, planes, bias=False)
        self.key = nn.Linear(inplanes, planes, bias=False)
        self.value = nn.Linear(inplanes, inplanes, bias=False)

        self.fc = nn.Linear(inplanes, inplanes)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        permute_x = x.resize(x.size(0), x.size(1), x.size(2)*x.size(3)).permute(0,2,1)
        q = self.query(permute_x)
        k = self.key(permute_x)
        v = self.value(permute_x)

        attn_weight = nn.functional.softmax(
            torch.matmul(q, k.permute(0,2,1)) / math.sqrt(self.planes),
            dim=2)
        content = permute_x + torch.matmul(attn_weight, v)
        content = self.fc(content)
        content = content.permute(0,2,1).resize(x.size(0), x.size(1), x.size(2), x.size(3))

        out = self.bn(content)
        out = self.relu(out)

        return out


class BilinearAttention(nn.Module):
    def __init__(self, inplanes, planes):
        super(BilinearAttention, self).__init__()
        self.planes = planes
        self.query = nn.Linear(inplanes, planes, bias=False)
        self.key = nn.Linear(inplanes, planes, bias=False)
        self.value = nn.Linear(inplanes, inplanes, bias=False)
        self.tanh = nn.Tanh()
        self.align = nn.Linear(planes, 1, bias=False)
        
        self.fc = nn.Linear(inplanes, inplanes)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        permute_x = x.resize(x.size(0), x.size(1), x.size(2)*x.size(3)).permute(0,2,1)
        q = self.tanh(self.query(permute_x))
        k = self.tanh(self.key(permute_x))
        v = self.value(permute_x)

        q = q.unsqueeze(2).expand(-1,-1,x.size(2)*x.size(3),-1)
        k = k.unsqueeze(1).expand(-1,x.size(2)*x.size(3),-1,-1)

        attn_weight = nn.functional.softmax(
            self.align(q * k).squeeze(),
            dim=2)
        content = permute_x + torch.matmul(attn_weight, v)
        content = self.fc(content)
        content = content.permute(0,2,1).resize(x.size(0), x.size(1), x.size(2), x.size(3))

        out = self.bn(content)
        out = self.relu(out)

        return out

class AttnPool(nn.Module):
    def __init__(self, planes, kernel_size):
        super(AttnPool, self).__init__()
        self.planes = planes
        self.att1 = Attention(planes, planes / 8)
        #self.att1 = BilinearAttention(planes, planes / 8)

        self.conv = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size, stride=1)
        
        self.trans = nn.Linear(planes, 1, bias=False)
        nn.init.constant(self.trans.weight, 0)

    def forward(self, x):
        h = self.att1(x)
        g = self.conv(x)
        g = self.bn(g)
        g = self.relu(g)
        g = self.pool(g)
        g = g.view(g.size(0), -1)

        permute_x = x.resize(x.size(0), x.size(1), x.size(2)*x.size(3)).permute(0,2,1)
        permute_h = h.resize(h.size(0), h.size(1), h.size(2)*h.size(3)).permute(0,2,1)
        attn_weight = nn.functional.softmax(
            self.trans(permute_h + g.unsqueeze(1)).squeeze(),
            dim=1)
        out = torch.matmul(attn_weight.unsqueeze(1), permute_x).squeeze()
        return out
        

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.attention1 = Attention(256 * block.expansion, 256 * block.expansion / 8)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.attention2 = AttnPool(512 * block.expansion, 7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x_att1 = self.attention1(x)
        x = self.layer4(x)
        #x_att2 = self.attention2(x)
        x = self.attention2(x)

        #x_att1 = self.pool1(x_att1)
        #x_att1 = x_att1.view(x_att1.size(0), -1)
        #x_att2 = self.pool2(x_att2)
        #x_att2 = x_att2.view(x_att2.size(0), -1)
        #x = torch.cat((x_att1, x_att2), 1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
