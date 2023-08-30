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
import numpy as np
from PIL import Image

from data import create_dataset
from models import create_model
from options.search_options import SearchOptions

from util.util import mkdirs
import wandb
from util.validation import get_fids, get_arch_diffs, save_arch

if __name__ == '__main__':
    print('=== CycleGANAS: NAS for CycleGAN starts. ===')
    opt = SearchOptions().parse()   # get training options
    train_dataset, test_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    if isinstance(train_dataset, tuple):
        train_dataset_1, train_dataset_2 = train_dataset[0], train_dataset[1]
        print('The number of training images = {} and {}'.format(len(train_dataset_1), len(train_dataset_2)))
        dataset_size = len(train_dataset_1) + len(train_dataset_2)    # get the number of images in the dataset.
    else:
        print('The number of training images = {}'.format(len(train_dataset)))
        train_dataset_1, train_dataset_2 = train_dataset, range(len(train_dataset)) # meaningless value
        
    print('The number of test images = {}'.format(len(test_dataset)))
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    
    # directory for saving architectures
    archs_dir = os.path.join(opt.checkpoints_dir, opt.name, 'archs')
    # directory for fid computation
    fid_images_dir = os.path.join(opt.checkpoints_dir, opt.name, 'fid_images')
    # directory for model.pth
    model_pth_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.mode)
    # directory for tensorboard logging
    model_runs_dir = os.path.join(opt.checkpoints_dir, opt.name, 'runs')
    
    mkdirs([archs_dir, fid_images_dir, model_pth_dir, model_runs_dir])
    
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
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        if opt.gumbel_softmax:        # when using gumbel softmax
            model.set_tau(epoch)
                
        for i, data in enumerate(zip(train_dataset_1, train_dataset_2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
            iter_data_time = time.time()
        
        # save architecture
        save_arch(model, archs_dir, epoch)
        # plot architecture
        arch_diffs = get_arch_diffs(model)
        wandb.log({
                    "Architecture/G_A": arch_diffs[0],
                    "Architecture/G_B": arch_diffs[1],
                    "Architecture/D_A": arch_diffs[2],
                    "Architecture/D_B": arch_diffs[3],
                },
                step=epoch
            )
        
        if epoch % opt.save_epoch_freq == 0:
            # validate
            fids = get_fids(opt, fid_images_dir, test_dataset, model)
            
            # visualize
            wandb.log({
                        "Clean FIDs/A->B": fids[0],
                        "Clean FIDs/A->B->A": fids[1],
                        "Clean FIDs/B->A": fids[2],
                        'Clean FIDs/B->A->B': fids[3],
                    },
                    step=epoch
                )
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        