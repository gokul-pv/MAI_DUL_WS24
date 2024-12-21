import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depth_to_space = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_to_space(x)
        x = self.conv(x)
        return x

class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.space_to_depth = SpaceToDepth(2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = self.space_to_depth(x)
        chunks = torch.chunk(x, 4, dim=1)
        x = sum(chunks) / 4.0
        x = self.conv(x)
        return x

class ResBlockUp(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.upsample1 = Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        self.upsample2 = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
        
    def forward(self, x):
        residual = self.bn1(x)
        residual = F.relu(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = F.relu(residual)
        residual = self.upsample1(residual)
        shortcut = self.upsample2(x)
        return residual + shortcut

class ResBlockDown(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.downsample1 = Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        self.downsample2 = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
        
    def forward(self, x):
        residual = F.relu(x)
        residual = self.conv1(residual)
        residual = F.relu(residual)
        residual = self.downsample1(residual)
        shortcut = self.downsample2(x)
        return residual + shortcut

class ResBlock(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=1)
        
    def forward(self, x):
        residual = F.relu(x)
        residual = self.conv1(residual)
        residual = F.relu(residual)
        residual = self.conv2(residual)
        return x + residual

class Generator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.linear = nn.Linear(128, 4*4*256)
        self.resblock1 = ResBlockUp(256, n_filters=n_filters)
        self.resblock2 = ResBlockUp(n_filters, n_filters=n_filters)
        self.resblock3 = ResBlockUp(n_filters, n_filters=n_filters)
        self.bn = nn.BatchNorm2d(n_filters)
        self.conv = nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1)
        
    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 256, 4, 4)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.resblock1 = ResBlockDown(3, n_filters=n_filters)
        self.resblock2 = ResBlockDown(n_filters, n_filters=n_filters)
        self.resblock3 = ResBlock(n_filters, n_filters=n_filters)
        self.resblock4 = ResBlock(n_filters, n_filters=n_filters)
        self.linear = nn.Linear(n_filters, 1)
        
    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = F.relu(x)
        x = torch.sum(x, dim=(2, 3))  # global sum pooling
        x = self.linear(x)
        return x
