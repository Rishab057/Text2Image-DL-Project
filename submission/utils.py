import os
import errno
import numpy as np

from copy import deepcopy

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
import math


#############################
def KL_loss(mu, logvar):
    """
    KL Loss between a Gaussian distribution (mu and logvar), and Gaussian distribution (0, 1)

    Parameters
    ----------
    mu : tensor
        mean with shape, (Batch_size x dim)
    logvar : tensor 
        log of variance with shape, (Batch_size x dim)

    Returns
    -------
    KLD : float
        KL Loss
    """
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def KL_loss2(mu1, mu2, logvar1, logvar2):
    """
    KL Loss between a Gaussian distribution (mu1 and logvar1), and Gaussian distribution (mu2, logvar2)

    Parameters
    ----------
    mu1 : tensor
        mean with shape, (Batch_size x dim)
    mu2 : tensor
        mean with shape, (Batch_size x dim)
    logvar1 : tensor 
        log of variance with shape, (Batch_size x dim)
    logvar2 : tensor 
        log of variance with shape, (Batch_size x dim)

    Returns
    -------
    KLD : float
        KL Loss
    """
    KLD = 0.5*torch.mean((logvar2 - logvar1  - 1 + ((logvar1.exp())+ (mu1-mu2).pow(2))/(logvar2.exp())))
    return KLD
    
def JSD_loss(mu, logvar):
    """
    JSD Loss between a Gaussian distribution (mu and logvar), and Gaussian distribution (0, 1)

    Parameters
    ----------
    mu : tensor
        mean with shape, (Batch_size x dim)
    logvar : tensor 
        log of variance with shape, (Batch_size x dim)

    Returns
    -------
    JSD : float
        JSD Loss
    """
    M = mu/2
    var = ((logvar.exp())+1)/4
    jsd = KL_loss(M, var)/2 + KL_loss2(M, mu, torch.log(var), logvar)/2
    return jsd
   
    
    
def comp_err(criterion, features, cond, module, labels, gpus):
    
    if cond == None:
        inputs = (features)
    else:
        inputs = (features, cond)
        
    logits = nn.parallel.data_parallel(module, inputs, gpus)
    error = criterion(logits, labels)
    return error

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    
    errD_real = comp_err(criterion, real_features, cond, netD.get_cond_logits, real_labels, gpus)
    errD_wrong = comp_err(criterion, real_features[:(batch_size-1)], cond[1:], netD.get_cond_logits, fake_labels[1:], gpus )
    errD_fake = comp_err(criterion, fake_features, cond, netD.get_cond_logits, fake_labels, gpus)

    if netD.get_uncond_logits is not None:
                
        uncond_errD_real = comp_err(criterion, real_features, None, netD.get_uncond_logits, real_labels, gpus)
        uncond_errD_fake = comp_err(criterion, fake_features, None, netD.get_uncond_logits, fake_labels, gpus)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    
    errD_fake = comp_err(criterion, fake_features, cond, netD.get_cond_logits, real_labels, gpus)
    
    if netD.get_uncond_logits is not None:        
        uncond_errD_fake = comp_err(criterion, fake_features, None, netD.get_uncond_logits, real_labels, gpus)
        errD_fake += uncond_errD_fake
        
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = 64
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
