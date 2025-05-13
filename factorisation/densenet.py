import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_groups(in_channels, out_channels):
    max_possible_groups = math.gcd(in_channels, out_channels)
    if max_possible_groups < 4:
        return 1
    return max_possible_groups


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_planes: int,
            growth_rate: int,
            groups1: int = 1,
            groups2: int = 1,
        ) -> None:
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            4*growth_rate,
            kernel_size=1,
            bias=False,
            groups=groups1,
        )
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(
            4*growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=groups2,
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class GroupedCroissantBottleneck(Bottleneck):
    def __init__(self, in_planes, growth_rate):
        super(GroupedCroissantBottleneck, self).__init__(
            in_planes,
            growth_rate,
            get_groups(in_planes, 4*growth_rate),
            get_groups(4*growth_rate, growth_rate),
        )


class Transition(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, groups: int = 1) -> None:
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            bias=False,
            groups=groups,
        )

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class GroupedTransition(Transition):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super(GroupedTransition, self).__init__(
            in_planes,
            out_planes,
            get_groups(in_planes, out_planes),
        )


class DenseNet(nn.Module):
    def __init__(
            self,
            nblocks: list[int],
            block: Type[Bottleneck],
            transition: Type[Transition],
            growth_rate: int = 12,
            reduction: float = 0.5,
            num_classes: int= 10,
        ) -> None:
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_standard_densnet121():
    return DenseNet([6,12,24,16], Bottleneck, Transition, growth_rate=32)

def get_increasing_grouped_densenet121():
    return DenseNet([6,12,24,16], GroupedCroissantBottleneck, Transition, growth_rate=32)

def get_transition_grouped_densenet121():
    return DenseNet([6,12,24,16], Bottleneck, GroupedTransition, growth_rate=32)

def get_increasing_transition_grouped_densenet121():
    return DenseNet([6,12,24,16], GroupedCroissantBottleneck, GroupedTransition, growth_rate=32)
