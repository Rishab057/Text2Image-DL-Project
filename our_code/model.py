import torch
import torch.nn as nn
import torch.nn.parallel

# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.GELU())
    return block

def DownSampling(in_plane, n_channel, ds_num, gen_flag):
    layers = []
    if gen_flag == True:
        layers.append(ConvBlock(in_plane, n_channel))
    else:
        layers.append(downBlock(in_plane, n_channel))
    for i in range(ds_num):
        layers.append(downBlock(n_channel, 2*n_channel))
        n_channel = 2*n_channel
    return nn.Sequential(*layers)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.GELU())
    return block

def UpSampling(ngf):
    block = nn.Sequential(
        upBlock(ngf, ngf // 2),
        upBlock(ngf // 2, ngf // 4),
        upBlock(ngf // 4, ngf // 8),
        upBlock(ngf // 8, ngf // 16))
    return block

def ConvBlock(in_planes, out_planes):
    block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.GELU())
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_num),
            nn.GELU(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    """Conditional Augmentation."""
    def __init__(self, text_dim, condition_dim, cuda):
        super(CA_NET, self).__init__()
        self.t_dim = text_dim
        self.c_dim = condition_dim
        self.cuda = cuda
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.activation = nn.GELU()

    def forward(self, text_embedding):
        # Generating mu, var for Gaussian distribution
        x = self.activation(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        # Sampling latent variable from Gaussian distribution
        std = torch.exp(0.5*logvar)
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        c_code = mu + eps*std
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8 + nef, ndf * 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self, text_dim, gf_dim, condition_dim, z_dim, cuda):
        super(STAGE1_G, self).__init__()
        self.gf_dim = gf_dim * 8
        self.ef_dim = condition_dim
        self.z_dim = z_dim
        
        self.ninput = self.z_dim + self.ef_dim
        self.ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET(text_dim, condition_dim, cuda)

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(self.ninput, self.ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(self.ngf * 4 * 4),
            nn.GELU())

        # ngf x 4 x 4 -> ngf/16 x 64 x 64
        self.upsampling_block = UpSampling(self.ngf)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            nn.Conv2d(self.ngf // 16, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsampling_block(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar



class STAGE1_D(nn.Module):
    def __init__(self, df_dim, condition_dim):
        super(STAGE1_D, self).__init__()
        self.ndf = df_dim
        self.nef = condition_dim
        
        # 3 x 32 x 32 --> 8ndf x 4 x 4
        self.downsampling_block = DownSampling(3, self.ndf, ds_num = 3, gen_flag = False)
        self.get_cond_logits = D_GET_LOGITS(self.ndf, self.nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.downsampling_block(image)
        return img_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, text_dim, gf_dim, condition_dim, z_dim, res_num, cuda):
        super(STAGE2_G, self).__init__()
        self.gf_dim = gf_dim
        self.ef_dim = condition_dim
        self.z_dim = z_dim
        self.STAGE1_G = STAGE1_G
        self.res_num = res_num
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False

        self.ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET(text_dim, condition_dim, cuda)
        # 3 x 64 x64 --> 4ngf x 16 x 16
        self.downsampling_block = DownSampling(3, self.ngf, ds_num = 2, gen_flag = True)
        # (4ngf + ef_dim) x 16 x 16 --> 4ngf x 16 x16
        self.hr_joint = ConvBlock(self.ef_dim + self.ngf * 4, self.ngf * 4)      
        self.residual = self._make_layer(ResBlock, self.ngf * 4)
        # 4ngf x 16 x 16 --> ngf // 4 x 256 x 256
        self.upsampling_block = UpSampling(self.ngf * 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            nn.Conv2d(self.ngf //4, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh())

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.res_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    
    def forward(self, text_embedding, noise):
        _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.downsampling_block(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        
        h_code = self.residual(h_code)
        h_code = self.upsampling_block(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self, df_dim, condition_dim):
        super(STAGE2_D, self).__init__()
        self.ndf = df_dim
        self.nef = condition_dim
        self.downsampling_block = nn.Sequential(
                # ndf x 128 x 128 --> 32ndf x4 x 4 --> 8ndf x 4 x 4
                DownSampling(3, self.ndf, ds_num = 5, gen_flag = False),
                ConvBlock(self.ndf * 32, self.ndf * 16),
                ConvBlock(self.ndf * 16, self.ndf * 8))

        self.get_cond_logits = D_GET_LOGITS(self.ndf, self.nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(self.ndf, self.nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.downsampling_block(image)
        return img_embedding
