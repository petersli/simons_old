from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import math
# our data loader
import MultipieLoader
import gc
import numpy as np

# my functions
#import zx

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPath', default='', help="path to model (to continue training)")
parser.add_argument('--dirCheckpoints', default='.', help='folder to model checkpoints')
parser.add_argument('--dirImageoutput', default='.', help='folder to output images')
parser.add_argument('--dirTestingoutput', default='.', help='folder to testing results/images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=200, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
opt = parser.parse_args()
print(opt)


## do not change the data directory
opt.data_dir_prefix = '/nfs/bigdisk/zhshu/data/fare/'

## change the output directory to your own
opt.output_dir_prefix = '/Users/Peter/Desktop/VAEtriplet'
opt.dirCheckpoints    = opt.output_dir_prefix + '/checkpoints/ExpressionTripletTest'
opt.dirImageoutput    = opt.output_dir_prefix + '/images/ExpressionTripletTest'
opt.dirTestingoutput  = opt.output_dir_prefix + '/testing/ExpressionTripletTest'

opt.imgSize = 64



try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass


# sample iamges
def visualizeAsImages(img_list, output_dir, 
                      n_sample=4, id_sample=None, dim=-1, 
                      filename='myimage', nrow=2, 
                      normalize=False):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    vutils.save_image(images, 
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, normalize = normalize, padding=2)


def parseSampledDataTripletMultipie(dp0_img,  dp9_img, dp1_img):
    ###
    dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
    dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    ###
    dp9_img  = dp9_img.float()/255 # convert to float and rerange to [0,1]
    dp9_img  = dp9_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    ###
    dp1_img  = dp1_img.float()/255 # convert to float and rerange to [0,1]
    dp1_img  = dp1_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img, dp9_img, dp1_img


def setFloat(*args):
    barg = []
    for arg in args: 
        barg.append(arg.float())
    return barg

def setCuda(*args):
    barg = []
    for arg in args: 
        barg.append(arg.cuda())
    return barg

def setAsVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg))
    return barg    

def setAsDumbVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg,requires_grad=False))
    return barg   



# Training data folder list
TrainingData = []
#session 01

TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_07_select/')

#session 02
'''
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_07_select/')

#session 03
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_05_select/')

#session 04
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_07_select/')

'''
# Testing
TestingData = []
TestingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_select_test/')


# ------------ training ------------ #
doTraining = True
doTesting = False
iter_mark=0
for epoch in range(opt.epoch_iter):
    if doTraining:
        train_loss = 0
        train_amount = 0+1e-6
        gc.collect() # collect garbage
        for subprocid in range(10):
            # random sample a dataroot
            dataroot = random.sample(TrainingData,1)[0]
            aaaa=0
            dataset = MultipieLoader.FareMultipieExpressionTripletsFrontal(opt, root=dataroot, resize=64)
            print('# size of the current (sub)dataset is %d' %len(dataset))
            train_amount = train_amount + len(dataset)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
            for batch_idx, data_point in enumerate(dataloader, 0):

                aaaa +=1
                if aaaa>20:
                    break

                gc.collect() # collect garbage
                # sample the data points: 
                # dp0_img: image of data point 0
                # dp9_img: image of data point 9, which is different in ``expression'' compare to dp0
                # dp1_img: image of data point 1, which is different in ``person'' compare to dp0
                dp0_img, dp9_img, dp1_img = data_point
                dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
                if opt.cuda:
                    dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
                dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )

                #############################
                ## put training code here ###
                #############################

                print(dp0_img.size())
                print(dp9_img.size())
                print(dp1_img.size())
                visualizeAsImages(dp0_img.data.clone(), 
                    opt.dirImageoutput, 
                    filename='iter_'+str(iter_mark)+'_img0', n_sample = 25, nrow=5, normalize=False)
                visualizeAsImages(dp9_img.data.clone(), 
                    opt.dirImageoutput, 
                    filename='iter_'+str(iter_mark)+'_img9', n_sample = 25, nrow=5, normalize=False)
                visualizeAsImages(dp1_img.data.clone(), 
                    opt.dirImageoutput, 
                    filename='iter_'+str(iter_mark)+'_img1', n_sample = 25, nrow=5, normalize=False)

                print('Test image saved, kill the process by Ctrl + C')
    gc.collect() # collect garbage




# ------------ testing ------------ #
























    ##
