"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_images_dataset
from util import html
import torch

# FID Score calculator
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.stats import entropy
import numpy as np
from torch.distributions import MultivariateNormal
#import seaborn as sns # This is for visualization
import scipy
#import pandas as pd

# FUNCTION TO CALCULATE FID SCORE
def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    '''
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    #### START CODE HERE ####
    return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2*torch.trace(matrix_sqrt(sigma_x @ sigma_y)) 
    #### END CODE HERE ####

# Preprocess image function to fit the inception model
def preprocess(img):
    print("Preprocessing image")
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img

def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))





if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    if opt.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        opt.world_size = torch.distributed.get_world_size()

    print(opt.gpu)

    # # INCEPTION V3 MODEL for FID
    # inception_model = inception_v3(pretrained=False)
    # inception_model.load_state_dict(torch.load("inception_v3_google-1a9a5a14.pth"))
    # # inception_model.to(device)
    # inception_model = inception_model.eval()

    # inception_model.fc = torch.nn.Identity()

    # # UNIT TEST FOR INCEPTION MODEL
    # test_identity_noise = torch.randn(100, 100)
    # assert torch.equal(test_identity_noise, inception_model.fc(test_identity_noise))
    # print("Unit test for Inception model Success!")

    # # UNIT TEST FOR FID CALCULATION FUNCTIONS

    # # UNIT TEST

    # mean1 = torch.Tensor([0, 0]) # Center the mean at the origin
    # covariance1 = torch.Tensor( # This matrix shows independence - there are only non-zero values on the diagonal
    #     [[1, 0],
    #     [0, 1]]
    # )
    # dist1 = MultivariateNormal(mean1, covariance1)

    # mean2 = torch.Tensor([0, 0]) # Center the mean at the origin
    # covariance2 = torch.Tensor( # This matrix shows dependence 
    #     [[2, -1],
    #     [-1, 2]]
    # )
    # dist2 = MultivariateNormal(mean2, covariance2)

    # assert torch.isclose(
    #     frechet_distance(
    #         dist1.mean, dist2.mean,
    #         dist1.covariance_matrix, dist2.covariance_matrix
    #     ),
    #     4 - 2 * torch.sqrt(torch.tensor(3.))
    # )

    # assert (frechet_distance(
    #         dist1.mean, dist1.mean,
    #         dist1.covariance_matrix, dist1.covariance_matrix
    #     ).item() == 0)

    # print("Unit test for FID calculation Success!")

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    length_ds = len(dataset)
    print("len dataset: ",length_ds)

    if opt.dataset_mode == "unaligned_coco":
        pass
    else:
        # FOR FID SCORE CALCULATION
        fake_features_list = []
        real_features_list = []

        # create a website
        web_dir = os.path.join(
            opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
        )  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
        print("creating web directory", web_dir)
        webpage = html.HTML(
            web_dir,
            "Experiment = %s, Phase = %s, Epoch = %s"
            % (opt.name, opt.phase, opt.epoch),
        )
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()

        start_time = time.time()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            for key in list(visuals.keys()):
                if key != 'real_A' and key != 'fake_B' and key != 'real_B' and key != 'fake_A':
                    del visuals[key]
            img_path = model.get_image_paths()  # get image paths
            if i % 5 == 0:  # save images to an HTML file
                current_time = time.time()
                time_per_image = (current_time - start_time)/(i+1)
                remaining_time = (time_per_image * (length_ds - i))/60
                print("processing (%04d/%04d)-th image... %s - Remaining time: %01d" % (i,length_ds, img_path, remaining_time))
            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize,
            )

        webpage.save()  # save the HTML