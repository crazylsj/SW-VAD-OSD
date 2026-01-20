import torch
from torch import nn
from asteroid.masknn import norms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, norm_type="bN", delta=False):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        self.delta = delta
        if delta:
            self.linear = nn.Linear(in_chan, in_chan)
            self.linear_norm = norms.get(norm_type)(in_chan*2)

        in_bottle = in_chan if not delta else in_chan*2
        in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        if self.delta:
            delta = self.linear(x.transpose(1, -1)).transpose(1, -1)
            x = torch.cat((x, delta), 1)
            x = self.linear_norm(x)

        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out

class TCN(nn.Module):
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512,  kernel_size=3,
                 norm_type="gLN"):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv).to(device)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type).to(device))
        out_conv = nn.Conv1d(bn_chan, n_src*out_chan, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)

    def forward(self, mixture_w):
 
        output = self.bottleneck(mixture_w)
     
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        logits = self.out(output)
 

        return logits.squeeze(1)

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

