"""Implementation of StackGAN Model."""
import torch
import torch.nn as nn
import torch.nn.parallel

def downBlock(in_planes, out_planes):
    """
    Downscale the spatial size by a factor of 2

    Parameters
    ----------
    in_planes : int
        Number of Input channels
    out_planes : int
        Number of Output channels

    Returns
    -------
    block : object
        Downsampling block
    """
    block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True))
    return block

def DownSampling(in_plane, n_channel, ds_num, gen_flag):
    """
    Downsampling the spatial size N times by a factor of 2

    Parameters
    ----------
    in_planes : int
        Number of Input channels
    out_planes : int
        Number of Output channels
    ds_num : int
        Number of times to use downsampling block
    gen_flag : bool
        Flag whether it is generator or not

    Returns
    -------
    out : object
        Downsampling block
    """
    layers = []
    if gen_flag == True:
        layers.append(ConvBlock(in_plane, n_channel))
    else:
        layers.append(downBlock(in_plane, n_channel))
    for i in range(ds_num):
        layers.append(downBlock(n_channel, 2*n_channel))
        n_channel = 2*n_channel
    return nn.Sequential(*layers)

def upBlock(in_planes, out_planes):
    """
    Upscale the spatial size by a factor of 2

    Parameters
    ----------
    in_planes : int
        Number of Input channels
    out_planes : int
        Number of Output channels

    Returns
    -------
    block : object
        Downsampling block
    """
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

def UpSampling(in_planes):
    """
    Upscale the spatial size by a factor of 16

    Parameters
    ----------
    in_planes : int
        Number of Input channels

    Returns
    -------
    block : object
        Downsampling block
    """
    block = nn.Sequential(
        upBlock(in_planes, in_planes // 2),
        upBlock(in_planes // 2, in_planes // 4),
        upBlock(in_planes // 4, in_planes // 8),
        upBlock(in_planes // 8, in_planes // 16))
    return block

def ConvBlock(in_planes, out_planes):
    """
    Convolution block with conv3x3, batchnorm and relu

    Parameters
    ----------
    in_planes : int
        Number of Input channels
    out_planes : int
        Number of Output channels

    Returns
    -------
    block : object
        Downsampling block
    """
    block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True))
    return block

class ResBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, channel_num):
        """ 
        Initialise Residual block.

        Parameters
        ----------
        channel_num : int
            Number of Input channels
        """
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Apply Residual block.
        
        Parameters
        ----------
        x : torch.tensor
            Input tensor of size batch_size x h x w x d 
            
        Returns
        -------
        out : torch.tensor
            Output tensor of size batch_size x h x w x d 
        """
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    """Conditional Augmentation."""
    def __init__(self, text_dim, condition_dim, cuda):
        """
        Initialise Conditional Augmentation.
        
        Parameters
        ----------
        text_dim : int
            Dimension of text
        condition_dim : int
            Dimension of conditional variable c
        cuda: bool
            Flag whether to use cuda
        """
        super(CA_NET, self).__init__()
        self.t_dim = text_dim
        self.c_dim = condition_dim
        self.cuda = cuda
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.activation = nn.ReLU()

    def forward(self, text_embedding):
        """
        Apply Conditional Augmentation.
        
        Parameters
        ----------
        text_embedding : torch.tensor
            Input tensor containing text embedding, batchsize x text_dim
            
        Returns
        -------
        c_code : torch.tensor
            conditional variable c with size batch_size x condition_dim
        mu : torch.tensor
            mean of gaussian distribution
        logvar : torch.tensor
            Log of variance of gaussian distribution
        """
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
    """Decision Logits class."""
    def __init__(self, ndf, nef, bcondition=True):
        """
        Initialise Decision Logits class.
        
        Parameters
        ----------
        ndf : int
            channel dimension of downsampled image
        nef : int
            channel dimension of conditional variable c
        bcondition: bool
            boolean flag
        """
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
        """
        Apply Decision Logits.
        
        Parameters
        ----------
        h_code : torch.tensor
            input tensor from downsampled image with size batch_size x h x w x d 
        c_code : torch.tensor
            conditional variable c with size batch_size x condition_dim

        Returns
        -------
        c_code : tensor
            Output tensor with size batch_size x 1
        """
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
    """Stage 1 Generator."""
    def __init__(self, text_dim, gf_dim, condition_dim, z_dim, cuda):
        """
        Initialise Stage 1 generator.
        
        Parameters
        ----------
        text_dim : int
            dimension of text embedding
        gf_dim : int
            channel dimension of image before upsampling
        condition_dim: int
            dimension of conditional variable c
        z_dim : int
            dimension of noise vector
        cuda: bool
            boolean flag for cuda
        """
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
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/16 x 64 x 64
        self.upsampling_block = UpSampling(self.ngf)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            nn.Conv2d(self.ngf // 16, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        """
        Apply Stage 1 Generator.
        
        Parameters
        ----------
        text_embedding : torch.tensor
            Input tensor containing text embedding, batchsize x text_dim
        noise : torch.tensor
            Input tensor containing noise vector, batchsize x z_dim

        Returns
        -------
        fake_img : torch.tensor
            output tensor of image with size batch_size x h x w x d
        mu : torch.tensor
            mean of gaussian distribution
        logvar : torch.tensor
            Log of variance of gaussian distribution
        """
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsampling_block(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar



class STAGE1_D(nn.Module):
    """Initialise Stage 1 Discriminator."""
    def __init__(self, df_dim, condition_dim):
        """
        Initialise Stage 1 Discriminator.
        
        Parameters
        ----------
        df_dim : int
            channel dimension of image 
        condition_dim: int
            dimension of conditional variable c
        """
        super(STAGE1_D, self).__init__()
        self.ndf = df_dim
        self.nef = condition_dim
        
        # 3 x 32 x 32 --> 8ndf x 4 x 4
        self.downsampling_block = DownSampling(3, self.ndf, ds_num = 3, gen_flag = False)
        self.get_cond_logits = D_GET_LOGITS(self.ndf, self.nef)
        self.get_uncond_logits = None

    def forward(self, image):
        """
        Apply Stage 1 Discriminator.
        
        Parameters
        ----------
        image : torch.tensor
            Input tensor of image, batchsize x h x w x d

        Returns
        -------
        img_embedding : torch.tensor
            output tensor of image after downsampling, batch_size x h x w x d
        """
        img_embedding = self.downsampling_block(image)
        return img_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    """Stage 2 Generator."""
    def __init__(self, STAGE1_G, text_dim, gf_dim, condition_dim, z_dim, res_num, cuda):
        """
        Initialise Stage 2 generator.
        
        Parameters
        ----------
        STAGE1_G: object
            Object of StageI Generator
        text_dim : int
            dimension of text embedding
        gf_dim : int
            channel dimension of image before upsampling
        condition_dim: int
            dimension of conditional variable c
        z_dim : int
            dimension of noise vector
        res_num : int
            Number of times to use residual block 
        cuda: bool
            boolean flag for cuda
        """
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
        """
        Apply Stage 2 Generator.
        
        Parameters
        ----------
        text_embedding : torch.tensor
            Input tensor containing text embedding, batchsize x text_dim
        noise : torch.tensor
            Input tensor containing noise vector, batchsize x z_dim

        Returns
        -------
        stage1_img : torch.tensor
            output tensor of StageI generator image with size batch_size x h x w x d
        fake_img : torch.tensor
            output tensor of image with size batch_size x h x w x d
        mu : torch.tensor
            mean of gaussian distribution
        logvar : torch.tensor
            Log of variance of gaussian distribution
        """
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
    """Stage 2 Discriminator."""
    def __init__(self, df_dim, condition_dim):
        """
        Initialise Stage 2 Discriminator.
        
        Parameters
        ----------
        df_dim : int
            channel dimension of image 
        condition_dim: int
            dimension of conditional variable c
        """
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
        """
        Apply Stage 2 Discriminator.
        
        Parameters
        ----------
        image : torch.tensor
            Input tensor of image, batchsize x h x w x d

        Returns
        -------
        img_embedding : torch.tensor
            output tensor of image after downsampling, batch_size x h x w x d
        """
        img_embedding = self.downsampling_block(image)
        return img_embedding