"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
import copy
import torch

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions

from util.util import mkdirs
import wandb

from util.validation import get_fids

if __name__ == '__main__':
    print('=== CycleGANAS: Train CycleGAN with searched architecture. ===')
    
    opt = TrainOptions().parse()   # get training options
    opt.name += '_train_{}ch'.format(opt.netG_A_ngf)
    
    c = 0
    while os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
        opt.name = opt.name[:-2] + '_{}'.format(c)
        c += 1
    
    train_dataset, test_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The number of training images = %d' % len(test_dataset))
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    
    # directory for fid computation
    fid_images_dir = os.path.join(opt.checkpoints_dir, opt.name, 'fid_images')
    # directory for model.pth
    model_pth_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.mode)
    # directory for tensorboard logging
    model_runs_dir = os.path.join(opt.checkpoints_dir, opt.name, 'runs')
    
    mkdirs([fid_images_dir, model_pth_dir, model_runs_dir])
    
    wandb.init(
        project='CycleGANAS',
        entity='antaegun20',
        name=opt.name,
        
        dir=model_runs_dir,
        
        # configs
        save_code=True,
        config=vars(opt),
    )
    
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    lowest_FIDs_AB, lowest_FIDs_ABA, lowest_FIDs_BA, lowest_FIDs_BAB = 9999, 9999, 9999, 9999
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(zip(train_dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            iter_data_time = time.time()
            
        # changed the order
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            # validate
            fids = get_fids(opt, fid_images_dir, test_dataset, model)
            
            # visualize
            #grid_image = get_grid_image(fid_images_dir)
            #writer.add_image(f"Translation Results", grid_image, epoch)
            wandb.log({
                        "Clean FIDs/A->B": fids[0],
                        "Clean FIDs/A->B->A": fids[1],
                        "Clean FIDs/B->A": fids[2],
                        'Clean FIDs/B->A->B': fids[3],
                    },
                    step=epoch
                )
            
            if fids[0] < lowest_FIDs_AB:
                # save main net
                G_A_sd = copy.deepcopy(model.netG_A.state_dict())
                G_A_sd = {k: v.cpu() for k, v in G_A_sd.items()}
                torch.save(G_A_sd, os.path.join(model_pth_dir, 'best_G_A.pth'))

                # save paired net
                G_B_sd = copy.deepcopy(model.netG_B.state_dict())
                G_B_sd = {k: v.cpu() for k, v in G_B_sd.items()}
                torch.save(G_B_sd, os.path.join(model_pth_dir, 'paired_G_B.pth'))

                lowest_FIDs_AB = fids[0]

            if fids[2] < lowest_FIDs_BA:
                # save main net
                G_B_sd = copy.deepcopy(model.netG_B.state_dict())
                G_B_sd = {k: v.cpu() for k, v in G_B_sd.items()}
                torch.save(G_B_sd, os.path.join(model_pth_dir, 'best_G_B.pth'))

                # save paired net
                G_A_sd = copy.deepcopy(model.netG_A.state_dict())
                G_A_sd = {k: v.cpu() for k, v in G_A_sd.items()}
                torch.save(G_A_sd, os.path.join(model_pth_dir, 'paired_G_A.pth'))

                lowest_FIDs_BA = fids[2]