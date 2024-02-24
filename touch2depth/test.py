"""
This script will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It then runs inference and saves results to an HTML file.

Instructions:
    - Evaluating checkpoints for pix2pix trained for Closure via Patching Policy:
    
    python3 test.py --dataroot /local/crv/ege/datasets/closure/closure_patching/test --model pix2pix --name closure_patching --results_dir ./test_results/closure_patching

    python3 test.py --dataroot /local/crv/ege/datasets/closure/test_ood --model pix2pix --name closure_boundary_policy --results_dir ./test_results/test_boundary_ood


    - Evaluating checkpoints for pix2pix trained for Closure via Boundary Policy:

    python3 test.py --dataroot /local/crv/ege/datasets/closure/closure_boundary/test --model pix2pix --name closure_boundary_policy --results_dir ./test_results/closure_boundary

    - Evaluating checkpoints for pix2pix trained for Closure via Occlusion Policy:

    python3 test.py --dataroot /proj/vondrick4/ege/datasets/Closure-and-Whole/ClosureViaBoundary/test --model pix2pix --name closure_via_occlusion --results_dir ./test_results/closure_occlusion



Note: Change --eval True adds batchnorm and dropout to evaluation, as in the original pix2pix.
    
See options/base_options.py and options/test_options.py for more test options.

Adapted from pix2pix official PyTorch implementation:
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
Image-to-Image Translation with Conditional Adversarial Networks
"""
import os
from options.test_options import TestOptions
from dataset import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test

    opt.eval = True #  adds batchnorm and dropout, as in the original pix2pix

    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.num_test = 1000
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = 'closure'
    dataset = create_dataset(opt)  # create ClosureDataset 
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='ClosureGAN')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode: This only affects layers like batchnorm and dropout.
    # For [pix2pix]:  use batchnorm and dropout in the original pix2pix. We experimented with and without eval() mode.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
