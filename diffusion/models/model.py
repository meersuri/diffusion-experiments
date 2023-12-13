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

class Block(torch.nn.Module):
    def __init__(self, channels, time_dim=32):
        super().__init__()
        self.in_norm = torch.nn.InstanceNorm2d(channels)
        self.in_conv = torch.nn.Conv2d(channels, channels, 3, padding='same')
        self.in_act = torch.nn.GELU()
        self.time_proj = torch.nn.Linear(time_dim, channels, bias=False)
        self.out_norm = torch.nn.InstanceNorm2d(2*channels)
        self.out_conv = torch.nn.Conv2d(2*channels, channels, 3, padding='same')
        self.out_act = torch.nn.GELU()

    def forward(self, x, t):
        time = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        out = self.in_act(self.in_conv(self.in_norm(x)))
        time = torch.broadcast_to(time, out.shape)
        out = torch.cat([out, time], axis=1)
        return x + self.out_act(self.out_conv(self.out_norm(out)))

class Layer(torch.nn.Module):
    def __init__(self, conv_block, block):
        super().__init__()
        self.conv = conv_block
        self.block = block

    def forward(self, x, t):
        return self.block(self.conv(x), t)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, factor=8, time_dim=32):
        super().__init__()

        self.filter_sizes = [int(in_channels*2**(i)) for i in range(int(np.log2(factor)) + 1)]
        self.layers = torch.nn.ModuleList([
            Layer(
                ConvBlock(self.filter_sizes[i],
                    self.filter_sizes[i + 1],
                    stride=2),
                Block(self.filter_sizes[i + 1], time_dim=time_dim)
                )
            for i in range(len(self.filter_sizes) - 1)
            ])

    def forward(self, x, t):
        fmaps = []
        out = x
        for layers in self.layers:
            out = layers(out, t)
            fmaps.append(out)
        return fmaps

class Decoder(torch.nn.Module):
    def __init__(self, filter_sizes, time_dim=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Layer(
                ConvBlock(filter_sizes[i], filter_sizes[i + 1], stride=0.5),
                Block(filter_sizes[i + 1], time_dim=time_dim),
                )
                for i in range(len(filter_sizes) - 1)
            ])

    def forward(self, encoder_outputs, t):
        assert len(encoder_outputs) == len(self.layers)
        encoder_outputs.reverse()
        out = encoder_outputs[0]
        for i in range(len(encoder_outputs)):
            out = self.layers[i](out, t)
            if i + 1 < len(encoder_outputs):
                out += encoder_outputs[i + 1]
        return out

class TimeEncoding(torch.nn.Module):
    def __init__(self, max_time, dim=32):
        super().__init__()
        self.dim = dim
        self.max_time = max_time
        pos = torch.arange(dim).unsqueeze(1)
        self.register_buffer('pos', pos)

    def forward(self, t):
        return torch.sin(self.pos*1000**(t/self.max_time)).reshape(1, self.dim)

class UNet(torch.nn.Module):
    def __init__(self, start_channels=8, factor=8, time_dim=32):
        super().__init__()
        self.in_conv = torch.nn.Conv2d(3, start_channels, 3, padding='same')
        self.time_embed = TimeEncoding(100, time_dim)
        self.enc = Encoder(start_channels, factor=factor)
        self.dec = Decoder(self.enc.filter_sizes[::-1])
        self.out_conv = torch.nn.Conv2d(start_channels, 3, 3, padding='same')

    def forward(self, x, t):
        time = self.time_embed(t)
        out = self.out_conv(self.dec(self.enc(self.in_conv(x), time), time))
        return out

if __name__ == '__main__':
    model = UNet(start_channels=8, factor=8)
    x = torch.randn(1,3,64,64)
    t = 0
    xh = model(x, t)
    torch.onnx.export(model, (x, t), 'model.onnx', opset_version=17)

