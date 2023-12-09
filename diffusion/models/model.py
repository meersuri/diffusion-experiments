import torch
import numpy as np

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        if stride >= 1:
            ConvClass = torch.nn.Conv2d
            kwargs = dict(stride=stride, padding=kernel_size - 2)
        else:
            ConvClass = torch.nn.ConvTranspose2d
            kwargs= dict(stride=int(1/stride), padding=kernel_size - 2, output_padding=kernel_size - 2)

        self.conv = ConvClass(in_channels,
                              out_channels,
                              kernel_size,
                              **kwargs)
        self.norm = torch.nn.InstanceNorm2d(out_channels)
        self.act = torch.nn.GELU()
        self.out_channels = out_channels

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, factor=8):
        super().__init__()

        self.filter_sizes = [int(in_channels*2**(i)) for i in range(int(np.log2(factor)) + 1)]
        self.convs = torch.nn.ModuleList([
            ConvBlock(self.filter_sizes[i],
                self.filter_sizes[i + 1],
                stride=2)
            for i in range(len(self.filter_sizes) - 1)
            ])

    def forward(self, x):
        fmaps = []
        out = x
        for conv in self.convs:
            out = conv(out)
            fmaps.append(out)
        return fmaps

class Decoder(torch.nn.Module):
    def __init__(self, filter_sizes):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            ConvBlock(filter_sizes[i], filter_sizes[i + 1], stride=0.5)
            for i in range(len(filter_sizes) - 1)
            ])

    def forward(self, encoder_outputs):
        assert len(encoder_outputs) == len(self.convs)
        encoder_outputs.reverse()
        out = encoder_outputs[0]
        for i in range(len(encoder_outputs)):
            out = self.convs[i](out)
            if i + 1 < len(encoder_outputs):
                out += encoder_outputs[i + 1]
            print(out.shape)
        return out

class UNet(torch.nn.Module):
    def __init__(self, start_channels=8, factor=8):
        super().__init__()
        self.in_conv = torch.nn.Conv2d(3, start_channels, 3, padding='same')
        self.enc = Encoder(start_channels, factor=factor)
        self.dec = Decoder(self.enc.filter_sizes[::-1])
        self.out_conv = torch.nn.Conv2d(start_channels, 3, 3, padding='same')

    def forward(self, x):
        out = self.out_conv(self.dec(self.enc(self.in_conv(x))))
        return out

if __name__ == '__main__':
    model = UNet(start_channels=8, factor=16)
    x = torch.randn(1,3,64,64)
    xh = model(x)
    torch.onnx.export(model, x, 'model.onnx', opset_version=17)

