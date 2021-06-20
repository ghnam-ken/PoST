import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

from utils.proc import transform, cnt2mask, cnt2poly, \
                        generate_pos_emb, get_adj_ind, \
                                        rel2abs, abs2rel


def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class PointSetTracker(nn.Module):
    def __init__(self, num_iter=5):
        super(PointSetTracker, self).__init__()
        self.num_iter = num_iter
        self.global_alignment = GlobalAlignment()
        self.local_alignment = LocalAlignment(num_iter=num_iter)
    
    def initialize(self, cnt):
        B, N = cnt.shape[:2]

        rnn_fs = [ None for _ in range(self.num_iter) ]

        pos_emb = generate_pos_emb(N)
        pos_emb = pos_emb.unsqueeze(0).expand(B, 2, N)
        pos_emb = pos_emb.cuda() if torch.cuda.is_available() else pos_emb

        return rnn_fs, pos_emb

    def propagate(self, x0, x1, cnt0, pos_emb, prev_rnn_fs):
        size = x0.shape[:-3:-1]
        cnt0 = abs2rel(cnt0, size)

        theta = self.global_alignment(x0, x1, cnt0)
        x1_h, cnt1_h = transform(x0, cnt0, theta)

        cnt1, rnn_fs = self.local_alignment(x1_h, x1, cnt1_h, 
                                            pos_emb, prev_rnn_fs)
        
        cnt1 = rel2abs(cnt1, size)
        return cnt1, rnn_fs

    def forward(self, *args, **kwargs):
        in_len = len(args) + len(kwargs)
        if in_len == 1:
            return self.initialize(*args, **kwargs)
        elif in_len == 5:
            return self.propagate(*args, **kwargs)
        else:
            raise ValueError(f'input length of {in_len} is not supported')


class GlobalAlignment(nn.Module):
    def __init__(self):
        super(GlobalAlignment, self).__init__()
        self.encoder = GlobalEncoder(out_dim=256)
        self.masking = Masking(dim=256)
        self.affine = Affine(dim=512)

        init_He(self)

    def forward(self, x0, x1, cnt0):
        f0 = self.encoder(x0)
        f1 = self.encoder(x1)

        m0 = cnt2mask(cnt0, f0.shape[:-3:-1])
        f0_m = self.masking(f0, m0)

        theta = self.affine(f1, f0_m)
        return theta


class LocalAlignment(nn.Module):
    def __init__(self, num_iter=5, adj_num=4):
        super(LocalAlignment, self).__init__()
        self.num_iter = num_iter
        self.adj_num = adj_num

        self.encoder = LocalEncoder(out_dim=128)

        align_modules = []
        for i in range(num_iter):
            module = LAM(feature_dim=128+4, state_dim=128)
            align_modules.append(module)
        self.align_modules = nn.Sequential(*align_modules)

        init_He(self)

    def forward(self, x1_h, x1, cnt1_h, pos_emb, prev_rnn_fs):
        size = x1_h.shape[:-3:-1]
        N = cnt1_h.shape[1]

        m1_h = cnt2mask(cnt1_h, size)
        fs = self.encoder(x1, x1_h, m1_h)

        rnn_fs = []
        cnt1 = cnt1_h
        for f, align_module, prev_rnn_f in zip(fs, self.align_modules, prev_rnn_fs):
            B, C, H, W = f.shape 

            s = F.grid_sample(f, cnt1)
            s = s.view(B, C, N)

            poly = cnt2poly(cnt1) # B, num_cp, 1, 2
            poly = poly.transpose(1, 3).view(B, 2, N).contiguous()
            
            m_in = torch.cat((s, pos_emb, poly), dim=1)
            adj = get_adj_ind(self.adj_num, N)

            offset, rnn_f = align_module(m_in, prev_rnn_f, adj)
            rnn_fs.append(rnn_f)

            offset = offset.transpose(1, 2).view(B, N, 1, 2).contiguous()
            cnt1 = cnt1 + offset
           
        cnt1 = cnt1.clamp(-1, 1)
        return cnt1, rnn_fs
        

class GlobalEncoder(nn.Module):
    def __init__(self, out_dim):
        super(GlobalEncoder, self).__init__()
        self.conv12 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2) # 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # 2
        self.conv23 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 4
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 4
        self.conv34 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 8
        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 8
        self.conv4b = nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1) # 8

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, in_f):
        f = F.interpolate(in_f, size=(224, 224), mode='bilinear', align_corners=False)
        f = (f - self.mean) / self.std
        x = self.conv12(f)
        x = self.conv2(F.relu(x))
        x = self.conv23(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv34(F.relu(x))
        x = self.conv4a(F.relu(x))
        x = self.conv4b(F.relu(x))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.conv45 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) # 16
        self.conv5a = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) # 16
        self.conv5b = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) # 16
        self.conv56 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) # 32
        self.conv6a = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) # 32
        self.conv6b = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) # 32

        self.fc = nn.Linear(dim, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(F.relu(x))
        x = self.conv5b(F.relu(x))
        x = self.conv56(F.relu(x))
        x = self.conv5a(F.relu(x))
        x = self.conv5b(F.relu(x))

        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])

        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)
        return theta


class Masking(nn.Module):
    def __init__(self, dim):
        super(Masking, self).__init__()
        self.conv1 = nn.Conv2d(dim+1, dim, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, feat, mask):
        x = torch.cat((feat, mask), dim=1)
        x = self.conv1(x)
        x = self.conv2(F.relu(x))

        return x + feat


class MHA(nn.Module):
    def __init__(self, dim, n_heads):
        super(MHA, self).__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.conv_q = nn.Linear(dim, dim//4 * n_heads, bias=False)
        self.conv_k = nn.Linear(dim, dim//4 * n_heads, bias=False)
        self.conv_v = nn.Linear(dim, dim * n_heads, bias=False)
        self.conv_w = nn.Linear(dim * n_heads, dim, bias=False)
        self.conv_a = nn.Linear(dim*2, dim, bias=False)

    def forward(self, x):
        B, C, K = x.shape
        H = self.n_heads

        x = x.transpose(1, 2).contiguous() # B, K, C
        residual = x
        
        x = self.layer_norm(x)

        q = self.conv_q(x) # B, K, H*C//4
        k = self.conv_k(x) # B, K, H*C//4
        v = self.conv_v(x) # B, K, H*C

        q = q.view(B, K, H, C//4) # B, K, H, C//4
        k = k.view(B, K, H, C//4) # B, K, H, C//4
        v = v.view(B, K, H, C) # B, K, H, C

        q = q.transpose(1,2).contiguous() # B, H, K, C//4
        k = k.transpose(1,2).contiguous() # B, H, K, C//4
        v = v.transpose(1,2).contiguous() # B, H, K, C

        attn = torch.matmul(q / (C//4)**0.5, k.transpose(-2, -1)) # B, H, K, K
        attn = F.softmax(attn, dim=-1) # B, H, K, K
        r = torch.matmul(attn, v) # B, H, K, C

        r = r.transpose(1, 2).contiguous() # B, K, H, C
        r = r.view(B, K, H*C)
        r = self.conv_w(r) # B, K, C
        r = torch.cat([r, residual], dim=2)
        r = self.conv_a(r)
        r = r.transpose(1, 2).contiguous() # B, C, K
        return r

class LAM(nn.Module):
    def __init__(self, feature_dim, state_dim):
        super(LAM, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        
        rnn_state_dim = 256            
        self.rnn_state = nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, rnn_state_dim, 1)
        self.rnn = nn.LSTM(rnn_state_dim, rnn_state_dim, 1)
        self.rnn.flatten_parameters()

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim + rnn_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, pf, adj):
        states = []

        x = self.head(x, adj) # B, C, N

        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, adj) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        rnn_state = self.rnn_state(state)
        rB, rC, rN = rnn_state.shape
        i = rnn_state.transpose(1, 2).contiguous() # B, N, C
        i = i.view(1, rB*rN, rC)
        o, npf = self.rnn(i, pf)
        o = o.view(rB, rN, rC)
        o = o.transpose(1, 2).contiguous() # B, C, N
        state = torch.cat([o, state], dim=1)

        x = self.prediction(state) # B, 2, N

        return x, npf


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.mha = MHA(state_dim, n_heads=4)
        self.conv = CircConv(state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj=None):
        x = self.mha(x)
        x = self.conv(x, adj)
        x = self.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        resnet = models.resnet50(pretrained=True).eval()
        self.conv1 = resnet.conv1
        self.conv1_p = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.conv1_p.weight.data = self.conv1.weight.data.clone()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # 1/4, 64

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024
        self.res5 = resnet.layer4 # 1/32, 2048

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, p, m):
        x = (x - Variable(self.mean)) / Variable(self.std)
        p = (p - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(x)
        p = self.conv1_p(p)
        m = self.conv1_m(m)

        x = (x + p + m) / 3

        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        r5 = self.res5(r4) # 1/32, 2048

        return r5, r4, r3, r2, c1


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=(1,1), stride=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1), stride=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1):
            laterals[i + 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        return outs


class LocalEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super(LocalEncoder, self).__init__()
        self.res50 = ResNet50()
        self.fpn = FPN([2048, 1024, 512, 256, 64], out_dim)

    def forward(self, x, p, m):
        rs = self.res50(x, p, m)
        outs = self.fpn(rs)

        return outs
