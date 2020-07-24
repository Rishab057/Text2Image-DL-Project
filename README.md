# Text2Image using multi-stage GAN models

- The steps to train a our version of StackGAN model on the Birds dataset using the preprocessed embeddings.

  - Step 1: train Stage-I GAN  
  `python main.py --config_name 'stageI' --dataset_name 'birds' --embedding_type 'cnn-rnn' --gpu_id '2,3' --z_dim 100 --data_dir '../data/birds' --image_size 64 --workers 4 --stage 1 --cuda True --train_flag True --batch_size 128 --max_epoch 120 --snapshot_interval 10 --lr_decay_epoch 20 --discriminator_lr 0.0002 --generator_lr 0.0002 --coef_kl 2.0 --condition_dim 128 --df_dim 96 --gf_dim 192 --text_dim 1024 --regularizer 'JSD'`

  - Step 2: train Stage-II GAN 
  `python main.py --config_name 'stageII' --dataset_name 'birds' --embedding_type 'cnn-rnn' --gpu_id '2' --z_dim 100 --data_dir '../data/birds' --image_size 256 --workers 4 --stage 2 --cuda True --train_flag True --batch_size 16 --max_epoch 100 --snapshot_interval 5 --lr_decay_epoch 20 --discriminator_lr 0.0002 --generator_lr 0.0002 --coef_kl 2.0 --condition_dim 128 --df_dim 96 --gf_dim 192 --res_num 2 --text_dim 1024 --stage1_g '../output/Birds_Stage_I_JSD/Model/netG_epoch_100.pth'  --regularizer 'JSD'`