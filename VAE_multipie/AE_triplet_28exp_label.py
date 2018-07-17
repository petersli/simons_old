from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
import math
#import matplotlib.pyplot as plt
import time
from glob import glob
#from util import *
import numpy as np
from PIL import Image
import os
import random

# our data loader
import MultipieLoader
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
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
parser.add_argument('--epoch_iter', type=int,default=500, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
opt = parser.parse_args()
print(opt)

## do not change the data directory
opt.data_dir_prefix = '/nfs/bigdisk/zhshu/data/fare/'

## change the output directory to your own
opt.output_dir_prefix = '/home/peterli/simons/VAE_multipie/AE'
opt.dirCheckpoints	= opt.output_dir_prefix + '/checkpoints'
opt.dirImageoutput	= opt.output_dir_prefix + '/images'
opt.dirTestingoutput  = opt.output_dir_prefix + '/testing'

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

class AE(nn.Module):
	def __init__(self, latent_variable_size):
		super(AE, self).__init__()
		self.latent_variable_size = latent_variable_size

		# ENCODER

		# img: 64 x 64 

		self.e1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(8)

		# 32 x 32 

		self.e2 = nn.Conv2d(8, 16, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(16)

		# 16 x 16

		self.e3 = nn.Conv2d(16, 32, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(32)

		# 8 x 8

		self.e4 = nn.Conv2d(32, 64, 4, 2, 1)
		self.bn4 = nn.BatchNorm2d(64)

		# 4 x 4

		self.e5 = nn.Conv2d(64, 64, 4, 2, 1)
		self.bn5 = nn.BatchNorm2d(64)

		# 2 x 2

		self.fc1 = nn.Linear(64*2*2, latent_variable_size)

		# batch_size x latent_variable_size (100 x 128)

		# DISENTANGLING

		self.disentangle1 = nn.Linear(latent_variable_size, latent_variable_size / 2)
		self.disentangle2 = nn.Linear(latent_variable_size, latent_variable_size / 2)
		self.disentangle3 = nn.Linear(latent_variable_size, latent_variable_size)

		# DECODER

		self.d1 = nn.Linear(latent_variable_size, 64*2*2*2)

		# 2 x 2

		self.up1 = nn.Upsample(scale_factor=2) # removes the *2*2 from output of d1 b/c scale_factor scales both H and W
		self.pd1 = nn.ReplicationPad2d(1) # +2 to height/width
		self.d2 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1)  # -2 to height/width
		self.bn6 = nn.BatchNorm2d(64, eps=1.e-3) 
		# eps is added to denominator for numerical stability

		# 4 x 4

		self.up2 = nn.Upsample(scale_factor=2)
		self.pd2 = nn.ReplicationPad2d(1)
		self.d3 = nn.Conv2d(64, 32, 3, 1)
		self.bn7 = nn.BatchNorm2d(32, 1.e-3)

	 	# 8 x 8

		self.up3 = nn.Upsample(scale_factor=2)
		self.pd3 = nn.ReplicationPad2d(1)
		self.d4 = nn.Conv2d(32, 16, 3, 1)
		self.bn8 = nn.BatchNorm2d(16, 1.e-3)

		# 16 x 16

		self.up4 = nn.Upsample(scale_factor=2)
		self.pd4 = nn.ReplicationPad2d(1)
		self.d5 = nn.Conv2d(16, 8, 3, 1)
		self.bn9 = nn.BatchNorm2d(8, 1.e-3)

		# 32 x 32

		self.up5 = nn.Upsample(scale_factor=2)
		self.pd5 = nn.ReplicationPad2d(1)
		self.d6 = nn.Conv2d(8, 3, 3, 1)

		# 64 x 64

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU()
		self.hardtanh = nn.Hardtanh()
		self.sigmoid = nn.Sigmoid()

 	def encode(self, x):
		#print("encode")
		h1 = self.leakyrelu(self.bn1(self.e1(x)))
		h2 = self.leakyrelu(self.bn2(self.e2(h1)))
		h3 = self.leakyrelu(self.bn3(self.e3(h2)))
		h4 = self.leakyrelu(self.bn4(self.e4(h3)))
		h5 = self.leakyrelu(self.bn5(self.e5(h4)))
		h5 = h5.view(-1, 64*2*2)

		return self.sigmoid(self.fc1(h5))

	def decode(self, z):
		#print("decode")
		h1 = self.relu(self.d1(z))
		h1 = h1.view(-1, 64*2, 2, 2)
		h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
		h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
		h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
		h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

		return self.hardtanh(self.d6(self.pd5(self.up5(h5))))

	def get_latent_vectors(self, x):
		z = self.encode(x) # whole latent vector
		z_per = z[:,0:100].contiguous() # part of z repesenenting identity of the person
		z_exp = z[:,100:128].contiguous()  # part of z representing the expression
		return z, z_per, z_exp

		# z_enc = self.encode(x)
		# z_per = self.sigmoid(self.disentangle1(z_enc))
		# z_exp = self.sigmoid(self.disentangle2(z_enc))	
		# z_dec = self.sigmoid(self.disentangle3(torch.cat((z_per, z_exp), dim=1)))

		return z_dec, z_per, z_exp


	def forward(self, x):
		z, z_per, z_exp = self.get_latent_vectors(x)
		recon_x = self.decode(z)
		return recon_x, z, z_per, z_exp

model=AE(latent_variable_size=128)


if opt.cuda:
	 model.cuda()

def recon_loss_func(recon_x, x):
	recon_func = nn.MSELoss()
	recon_func.size_average = False
	return recon_func(recon_x, x)

def siamese_loss_func(z1, z2, label):
	siamese_func = nn.CosineEmbeddingLoss()
	siamese_func.size_average = False
	siamese_func.margin = 0.5
	#y = torch.ones_like(z2)
	y = torch.ones(z1.size()[0]).cuda()

	#size of target has to match size of inputs
	y.requires_grad_(False)
	if label == 1: # measure similarity
		return siamese_func(z1, z2, target=y)
	elif label == -1: # measure dissimilarity
		y = y * -1
		return siamese_func(z1, z2, target=y)

def BCE(x, target):
	BCE_func = nn.BCEWithLogitsLoss() # combined sigmoid and BCE into one layer
	return BCE_func(x, target)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

lossfile = open(opt.output_dir_prefix + "losses.txt", "w")
sim_loss = 0
dis_loss = 0

def train(epoch):
	print("train")
	model.train()
	recon_train_loss = 0
	siamese_train_loss = 0
	expression_train_loss = 0
	dataroot = random.sample(TrainingData,1)[0]

	dataset = MultipieLoader.FareMultipieExpressionTripletsFrontal(opt, root=dataroot, resize=64)
	print('# size of the current (sub)dataset is %d' %len(dataset))
 #   train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
	
	for batch_idx, data_point in enumerate(dataloader, 0):

		gc.collect() # collect garbage
		# sample the data points: 
		# dp0_img: image of data point 0
		# dp9_img: image of data point 9, which is different in ``expression'' compare to dp0
		# dp1_img: image of data point 1, which is different in ``person'' compare to dp0
		dp0_img, dp9_img, dp1_img, dp0_ide, dp9_ide, dp1_ide = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )


		z_dp9, z_per_dp9, z_exp_dp9 = model.get_latent_vectors(dp9_img)
		z_dp1, z_per_dp1, z_exp_dp1 = model.get_latent_vectors(dp1_img)

		optimizer.zero_grad()
		model.zero_grad()

		recon_batch_dp0, z_dp0, z_per_dp0, z_exp_dp0 = model(dp0_img)

		# calc reconstruction loss (dp0 only)

		recon_loss = recon_loss_func(recon_batch_dp0, dp0_img)
		optimizer.zero_grad()
		recon_loss.backward(retain_graph=True)
		recon_train_loss += recon_loss.data[0].item()

		# calc siamese loss

		sim_loss = siamese_loss_func(z_per_dp0, z_per_dp9, 1) + siamese_loss_func(z_exp_dp0, z_exp_dp1, 1) # similarity
		dis_loss = siamese_loss_func(z_exp_dp0, z_exp_dp9, -1) + siamese_loss_func(z_per_dp0, z_per_dp1, -1) # dissimilarity
		siamese_loss = sim_loss + dis_loss

		siamese_loss.backward(retain_graph=True)
		siamese_train_loss += siamese_loss.data[0].item()

		# BCE expression loss
		
		smile_target = torch.ones(z_exp_dp0.size()).cuda()
		neutral_target = torch.zeros(z_exp_dp0.size()).cuda()

		if dp0_ide == '01': #neutral
			expression_loss = BCE(z_exp_dp0, neutral_target)
		else: #smile
			expression_loss = BCE(z_exp_dp0, smile_target)

		if dp9_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp9, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp9, smile_target)

		if dp1_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp1, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp1, smile_target)

		expression_loss.backward()
		expression_train_loss += expression_loss[0].item()



		optimizer.step()
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReconLoss: {:.6f}\tSimLoss: {:.6f}\tDisLoss: {:.6f}\tExpLoss: {:.6f}'.format(
			epoch, batch_idx * opt.batchSize, (len(dataloader) * opt.batchSize),
			 100. * batch_idx / len(dataloader),
			  recon_loss.data[0].item() / opt.batchSize, sim_loss.data[0].item() / opt.batchSize,
			   dis_loss.data[0].item() / opt.batchSize, expression_loss[0].item() / opt.batchSize))
			#loss is calculated for each img, so divide by batch size to get average loss for the batch

	lossfile.write('Epoch: {} Recon: {:.4f}\n'.format(epoch, recon_train_loss / (len(dataloader) * opt.batchSize)))
	lossfile.write('Epoch: {} SiameseSim: {:.4f} SiameseDis: {:.4f}\n'.format(epoch, sim_loss.data[0].item() / opt.batchSize, 
		dis_loss.data[0].item() / opt.batchSize))
	lossfile.write('Epoch: {} Expression: {:.4f}\n'.format(epoch, expression_train_loss / (len(dataloader) * opt.batchSize)))


	print('====> Epoch: {} Avg recon loss: {:.4f} Avg siamese loss: {:.4f} Avg exp loss: {:.4f}'.format(
		  epoch, recon_train_loss / (len(dataloader) * opt.batchSize), siamese_train_loss / (len(dataloader) * opt.batchSize),
		   expression_train_loss / (len(dataloader) * opt.batchSize)))
			#divide by (batch_size * num_batches) to get loss for the epoch


	#data
	visualizeAsImages(dp0_img.data.clone(), 
		opt.dirImageoutput, 
		filename='epoch_'+str(epoch)+'_img0', n_sample = 18, nrow=5, normalize=False)
	visualizeAsImages(dp9_img.data.clone(), 
		opt.dirImageoutput, 
		filename='epoch_'+str(epoch)+'_img9', n_sample = 18, nrow=5, normalize=False)
	visualizeAsImages(dp1_img.data.clone(), 
		opt.dirImageoutput, 
		filename='epoch_'+str(epoch)+'_img1', n_sample = 18, nrow=5, normalize=False)

	#reconstruction (dp0 only)
	visualizeAsImages(recon_batch_dp0.data.clone(), 
		opt.dirImageoutput,
		filename='epoch_'+str(epoch)+'_recon0', n_sample = 18, nrow=5, normalize=False)

	print('Data and reconstructions saved.')


	return recon_train_loss / (len(dataloader) * opt.batchSize), siamese_train_loss / (len(dataloader) * opt.batchSize)


def test(epoch):
	print("test")
	model.eval()
	recon_test_loss = 0
	siamese_test_loss = 0
	expression_test_loss = 0
	dataroot = random.sample(TestingData,1)[0]

	dataset = MultipieLoader.FareMultipieExpressionTripletsFrontal(opt, root=dataroot, resize=64)
	print('# size of the current (sub)dataset is %d' %len(dataset))
   # train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
	for batch_idx, data_point in enumerate(dataloader, 0):
		gc.collect() # collect garbage
		# sample the data points: 
		dp0_img, dp9_img, dp1_img, dp0_ide, dp9_ide, dp1_ide = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )


		z_dp9, z_per_dp9, z_exp_dp9 = model.get_latent_vectors(dp9_img)
		z_dp1, z_per_dp1, z_exp_dp1 = model.get_latent_vectors(dp1_img)

		recon_batch_dp0, z_dp0, z_per_dp0, z_exp_dp0 = model(dp0_img)

		# test disentangling

		z_per0_exp9 = torch.cat((z_per_dp0, z_exp_dp9), dim=1) # should be person 0 with expression 9
		recon_per0_exp9 = model.decode(z_per0_exp9)

		visualizeAsImages(recon_per0_exp9.data.clone(), 
		opt.dirImageoutput, 
		filename='epoch_'+str(epoch)+'_per0_exp9', n_sample = 18, nrow=5, normalize=False)

		z_per0_exp1 = torch.cat((z_per_dp0, z_exp_dp1), dim=1) # should look the same as dp0_img (exp1 and exp0 are the same)
		recon_per0_exp1 = model.decode(z_per0_exp1)

		visualizeAsImages(recon_per0_exp1.data.clone(), 
		opt.dirImageoutput, 
		filename='epoch_'+str(epoch)+'_per0_exp1', n_sample = 18, nrow=5, normalize=False)


		# calc reconstruction loss (dp0 only)

		recon_loss = recon_loss_func(recon_batch_dp0, dp0_img)
		recon_test_loss += recon_loss.data[0].item()

		# calc siamese loss

		sim_loss = siamese_loss_func(z_per_dp0, z_per_dp9, 1) + siamese_loss_func(z_exp_dp0, z_exp_dp1, 1) # similarity
		dis_loss = siamese_loss_func(z_exp_dp0, z_exp_dp9, -1) + siamese_loss_func(z_per_dp0, z_per_dp1, -1) # dissimilarity
		siamese_loss = sim_loss + dis_loss

		siamese_test_loss = siamese_loss.data[0].item()


		# BCE expression loss

		smile_target = torch.ones(z_exp_dp0.size()).cuda()
		neutral_target = torch.zeros(z_exp_dp0.size()).cuda()

		if dp0_ide == '01': #neutral
			expression_loss = BCE(z_exp_dp0, neutral_target)
		else: #smile
			expression_loss = BCE(z_exp_dp0, smile_target)

		if dp9_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp9, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp9, smile_target)

		if dp1_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp1, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp1, smile_target)

		expression_test_loss += expression_loss[0].item()


	print('====> Test set recon loss: {:.4f}\tSiamese loss:  {:.4f}\t Exp loss:'.format(recon_test_loss / (opt.batchSize * len(dataloader)), 
		siamese_test_loss / (opt.batchSize * len(dataloader)), expression_test_loss / (opt.batchSize * len(dataloader))))


def load_last_model():
	 models = glob(opt.dirCheckpoints + '/*.pth')
	 model_ids = [(int(f.split('_')[1]), f) for f in models]
	 start_epoch, last_cp = max(model_ids, key=lambda item:item[0])  # max returns the model_id with the largest proxy value (item)
	 model.load_state_dict(torch.load(last_cp))
	 return start_epoch, last_cp

def start_training():
	# start_epoch, _ = load_last_model()
	start_epoch = 0


	for epoch in range(start_epoch + 1, start_epoch + opt.epoch_iter + 1):
		recon_loss, siamese_loss = train(epoch)
		torch.save(model.state_dict(),
		 opt.dirCheckpoints + '/Epoch_{}_Recon_{:.4f}_Siamese_{:.4f}.pth'.format(epoch, recon_loss, siamese_loss))
		if epoch % 10 == 0 or epoch == 1:
			test(epoch)

	lossfile.close()

def last_model_to_cpu():
	_, last_cp = load_last_model()
	model.cpu()
	torch.save(model.state_dict(), opt.dirCheckpoints + '/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
	start_training()
	# last_model_to_cpu()
	# load_last_model()
	# rand_faces(10)
	# da = load_pickle(test_loader[0])
	# da = da[:120]
	# it = iter(da)
	# l = zip(it, it, it)
	# # latent_space_transition(l)
	# perform_latent_space_arithmatics(l)

