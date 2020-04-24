from __future__ import print_function
import torch
import torchvision.transforms as transforms

import click
import os
import random
import sys
import datetime
import dateutil
import dateutil.tz


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from trainer import GANTrainer

@click.command()
@click.option("-cn", "--config_name", default = 'stageI')
@click.option("-dn", "--dataset_name", default = 'birds')
@click.option("-et", "--embedding_type", default = 'cnn-rnn')
@click.option("-gpu", "--gpu_id", default='0')
@click.option("-zd", "--z_dim", default = 4)
@click.option("-dir", "--data_dir", default='../data/birds')
@click.option("-is", "--image_size", default = 64)
@click.option("-wo", "--workers", default = 4)
@click.option("-st", "--stage", default = 1)
@click.option("-cu", "--cuda", default = True)

@click.option("-tf", "--train_flag", default = True)
@click.option("-bs", "--batch_size", default = 128)
@click.option("-me", "--max_epoch", default = 120)
@click.option("-si", "--snapshot_interval", default = 10)
@click.option("-lrd", "--lr_decay_epoch", default = 20)
@click.option("-dlr", "--discriminator_lr", default = 0.0002)
@click.option("-glr", "--generator_lr", default = 0.0002)
@click.option("-ckl", "--coef_kl", default = 2.0)

@click.option("-cd", "--condition_dim", default = 128)
@click.option("-dfd", "--df_dim", default = 96)
@click.option("-gfd", "--gf_dim", default = 192)
@click.option("-rn", "--res_num", default = 2)

@click.option("-td", "--text_dim", default = 1024)
@click.option("-ng", "--net_g", default = '')
@click.option("-nd", "--net_d", default = '')
@click.option("-stg", "--stage1_g", default = '')
@click.option("-ms", "--manual-seed", help='manual seed')


def main(gpu_id, data_dir, manual_seed, cuda, train_flag, image_size,
         batch_size, workers, stage, dataset_name, config_name, max_epoch, snapshot_interval,
         net_g, net_d, z_dim, generator_lr, discriminator_lr, lr_decay_epoch, coef_kl,
         stage1_g, embedding_type, condition_dim, df_dim, gf_dim, res_num, text_dim):

    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if cuda:
        torch.cuda.manual_seed_all(manual_seed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % (dataset_name, config_name, timestamp)

    num_gpu = len(gpu_id.split(','))
    if train_flag:
        image_transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = TextDataset(data_dir, 'train',
                              imsize=image_size,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size= batch_size * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(workers))

        algo = GANTrainer(output_dir, max_epoch, snapshot_interval,
                 gpu_id, batch_size, train_flag, net_g, net_d, cuda, 
                 stage1_g, z_dim, generator_lr, discriminator_lr, 
                 lr_decay_epoch, coef_kl)
        algo.train(dataloader, stage, text_dim, gf_dim, condition_dim, z_dim, df_dim, res_num)
    else:
        datapath= '%s/test/val_captions.t7' % (data_dir)
        algo = GANTrainer(output_dir, max_epoch, snapshot_interval,
                 gpu_id, batch_size, train_flag, net_g, net_d, cuda, 
                 stage1_g, z_dim, generator_lr, discriminator_lr, 
                 lr_decay_epoch, coef_kl)
        algo.sample(datapath, stage)

if __name__ == "__main__":
    main()