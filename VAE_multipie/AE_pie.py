from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import time
from glob import glob
#from util import *
import numpy as np
from PIL import Image

import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import gradcheck
from torch.autograd import Function
import math
# our data loader
import MultipieLoader
import gc

# parser = argparse.ArgumentParser(description='PyTorch VAE')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#					 help='input batch size for training (default: 64)')
# parser.add_argument('--epochs', type=int, default=20, metavar='N',
#					 help='number of epochs to train (default: 20)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#					 help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#					 help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=1, metavar='N',
#					 help='how many batches to wait before logging training status')
# args = parser.parse_args()
# # args = parser.parse_args(args=[])  --for ipynb
# args.cuda = not args.no_cuda and torch.cuda.is_available()



# torch.manual_seed(args.seed)
# if args.cuda:
#	 torch.cuda.manual_seed(args.seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


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
parser.add_argument('--epoch_iter', type=int,default=200, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
opt = parser.parse_args()
print(opt)

## do not change the data directory
opt.data_dir_prefix = '/nfs/bigdisk/zhshu/data/fare/'

## change the output directory to your own
opt.output_dir_prefix = '/Users/Peter/Desktop/VAEtriplet'
opt.dirCheckpoints	= opt.output_dir_prefix + '/checkpoints/ExpressionTripletTest'
opt.dirImageoutput	= opt.output_dir_prefix + '/images/ExpressionTripletTest'
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



class AE(nn.Module):
	def __init__(self, nc, ngf, ndf, latent_variable_size):
		super(AE, self).__init__()

		self.nc = nc # num channels
		self.ngf = ngf # num generator filters
		self.ndf = ndf # num discriminator filters
		self.latent_variable_size = latent_variable_size

		# encoder
		self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
		self.bn1 = nn.BatchNorm2d(ndf)

		self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(ndf*2)

		self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(ndf*4)

		self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
		self.bn4 = nn.BatchNorm2d(ndf*8)

		self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
		self.bn5 = nn.BatchNorm2d(ndf*8)

		self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size) #if ndf=64, args are (8192, 128)

		# decoder
		self.d1 = nn.Linear(latent_variable_size, ngf*8*4*4*2)

		self.up1 = nn.Upsample(scale_factor=2)
		self.pd1 = nn.ReplicationPad2d(1)
		self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
		self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

		self.up2 = nn.Upsample(scale_factor=2)
		self.pd2 = nn.ReplicationPad2d(1)
		self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
		self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

		self.up3 = nn.Upsample(scale_factor=2)
		self.pd3 = nn.ReplicationPad2d(1)
		self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
		self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

		self.up4 = nn.Upsample(scale_factor=2)
		self.pd4 = nn.ReplicationPad2d(1)
		self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
		self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

		self.up5 = nn.Upsample(scale_factor=2)
		self.pd5 = nn.ReplicationPad2d(1)
		self.d6 = nn.Conv2d(ngf, nc, 3, 1)

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		#print("encode")
		h1 = self.leakyrelu(self.bn1(self.e1(x)))
		h2 = self.leakyrelu(self.bn2(self.e2(h1)))
		h3 = self.leakyrelu(self.bn3(self.e3(h2)))
		h4 = self.leakyrelu(self.bn4(self.e4(h3)))
		h5 = self.leakyrelu(self.bn5(self.e5(h4)))
		h5 = h5.view(-1, self.ndf*8*4*4)

		return self.fc1(h5)
		#return self.bn5(self.e5(h4))

	def decode(self, z):
		#print("decode")
		h1 = self.relu(self.d1(z))
		h1 = h1.view(-1, self.ngf*8*2, 2, 2)
		h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
		h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
		h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
		h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

		return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

	def get_latent_vectors(self, x):
		z = self.encode(x.view(-1, self.nc, self.ndf, self.ngf)) # whole latent vector
		#print("pre slice {}".format(z.size()))
		# z_per = z[0:64] # part of z repesenenting identity of the person
		# z_exp = z[64:]  # part of z representing the expression
		#print("post slice {}".format(z.size()))
		return z

	def forward(self, x):
		z = self.encode(x)
		recon_x = self.decode(z)
		return recon_x, z


model = AE(nc=3, ngf=64, ndf=64, latent_variable_size=128)

if opt.cuda:
	 model.cuda()

def recon_loss_func(recon_x, x):
	recon_func = nn.MSELoss()
	recon_func.size_average = False
	return recon_func(recon_x, x)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):
	print("train")
	model.train()
	recon_train_loss = 0
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
		dp0_img, dp9_img, dp1_img = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )

		# optimizer.zero_grad()
		# z_dp9, z_per_dp9, z_exp_dp9 = model.get_latent_vectors(dp9_img)
		# z_dp1, z_per_dp1, z_exp_dp1 = model.get_latent_vectors(dp1_img)

		recon_batch_dp0, z_dp0 = model(dp0_img)
		recon_loss = recon_loss_func(recon_batch_dp0, dp0_img)

		optimizer.zero_grad()
		recon_loss.backward()
		recon_train_loss += recon_loss.data[0]

		#calc siamese loss

		# siamese_loss = siamese_loss_func(z_per_dp0, z_per_dp9, 1) + siamese_loss_func(z_exp_dp0, z_exp_dp1, 1)
		# siamese_loss += siamese_loss_func(z_exp_dp0, z_exp_dp9, -1)
		# siamese_loss += siamese_loss_func(z_per_dp0, z_per_dp1, -1)

		# optimizer.zero_grad()
		# siamese_loss.backward()
		# siamese_train_loss = siamese_loss.data[0]

	   


		optimizer.step()
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRecon Loss: {:.6f}'.format(
			epoch, batch_idx * batch_size, (len(dataloader)*batch_size),
			100. * batch_idx / len(dataloader),
			recon_loss.data[0] / len(dataloader)))

	print('====> Epoch: {} Average recon loss: {:.4f}'.format(
		  epoch, recon_train_loss / (len(dataloader)*64)))

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

	return recon_train_loss / (len(dataloader)*64)

def test(epoch):
	print("test")
	model.eval()
	recon_test_loss = 0
	siamese_test_loss = 0
	dataroot = random.sample(TestingData,1)[0]

	dataset = MultipieLoader.FareMultipieExpressionTripletsFrontal(opt, root=dataroot, resize=64)
	print('# size of the current (sub)dataset is %d' %len(dataset))
   # train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
	for batch_idx, data_point in enumerate(dataloader, 0):
		gc.collect() # collect garbage
		dp0_img, dp9_img, dp1_img = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )

		# z_dp9, z_per_dp9, z_exp_dp9 = model.get_latent_vectors(dp9_img)
		# z_dp1, z_per_dp1, z_exp_dp1 = model.get_latent_vectors(dp1_img)

		recon_batch_dp0, z_dp0 = model(dp0_img)
		recon_test_loss += recon_loss_func(recon_batch_dp0, dp0_img).data[0]

		#calc siamese loss

		# siamese_loss += siamese_loss_func(z_per_dp0, z_per_dp9, 1)
		# siamese_loss += siamese_loss_func(z_exp_dp0, z_exp_dp1, 1)
		# siamese_loss += siamese_loss_func(z_exp_dp0, z_exp_dp9, -1)
		# siamese_loss += siamese_loss_func(z_per_dp0, z_per_dp1, -1)

		# siamese_test_loss = siamese_loss.data[0]



	# for batch_idx in test_loader:
	#	 data = load_batch(batch_idx, False)

	#	 if args.cuda:
	#		 data = data.cuda()
	#	 recon_batch, mu, logvar = model(data)
	#	 test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

	#	 torchvision.utils.save_image(data.data, '../imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
	#	 torchvision.utils.save_image(recon_batch.data, '../imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

	recon_test_loss /= (len(dataloader)*64)
	siamese_test_loss /= (len(dataloader)*64)
	print('====> Test set recon loss: {:.4f}'.format(recon_test_loss))
	return recon_test_loss


def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
	#load_last_model()
	model.eval()
	data = [im for item in items for im in item]
	data = [totensor(i) for i in data]
	data = torch.stack(data, dim=0)
	data = Variable(data, volatile=True)
	if args.cuda:
		data = data.cuda()
	z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
	it = iter(z.split(1))
	z = zip(it, it, it)
	zs = []
	numsample = 11
	for i,j,k in z:
		for factor in np.linspace(0,1,numsample):
			zs.append((i-j)*factor+k)
	z = torch.cat(zs, 0)
	recon = model.decode(z)

	it1 = iter(data.split(1))
	it2 = [iter(recon.split(1))]*numsample
	result = zip(it1, it1, it1, *it2)
	result = [im for item in result for im in item]

	result = torch.cat(result, 0)
	torchvision.utils.save_image(result.data, '../imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): # input is list of tuples of  (a,b)
	#load_last_model()
	model.eval()
	data = [im for item in items for im in item[:-1]]
	data = [totensor(i) for i in data]
	data = torch.stack(data, dim=0)
	data = Variable(data, volatile=True)
	if args.cuda:
		data = data.cuda()
	z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
	it = iter(z.split(1))
	z = zip(it, it)
	zs = []
	numsample = 11
	for i,j in z:
		for factor in np.linspace(0,1,numsample):
			zs.append(i+(j-i)*factor)
	z = torch.cat(zs, 0)
	recon = model.decode(z)

	it1 = iter(data.split(1))
	it2 = [iter(recon.split(1))]*numsample
	result = zip(it1, it1, *it2)
	result = [im for item in result for im in item]

	result = torch.cat(result, 0)
	torchvision.utils.save_image(result.data, '../imgs/trans.jpg', nrow=2+numsample, padding=2)


def rand_faces(num=5):
	#load_last_model()
	model.eval()
	z = torch.randn(num*num, model.latent_variable_size)
	#z = Variable(z, volatile=True)
	if opt.cuda:
		z = z.cuda()
	recon = model.decode(z)
	torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)

# def load_last_model():
#	 models = glob('../models/*.pth')
#	 model_ids = [(int(f.split('_')[1]), f) for f in models]
#	 start_epoch, last_cp = max(model_ids, key=lambda item:item[0])  # max returns the model_id with the largest proxy value (item)
#	 model.load_state_dict(torch.load(last_cp))
#	 return start_epoch, last_cp

def start_training():
	# start_epoch, _ = load_last_model()
	start_epoch = 0

	for epoch in range(start_epoch + 1, start_epoch + opt.epoch_iter + 1):
		recon_train_loss = train(epoch)
		test_loss = test(epoch)
		# torch.save(model.state_dict(), '../models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))

def last_model_to_cpu():
	_, last_cp = load_last_model()
	model.cpu()
	# torch.save(model.state_dict(), '../models/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
	start_training()
	# last_model_to_cpu()
	# load_last_model()
	rand_faces(10)
	# da = load_pickle(test_loader[0])
	# da = da[:120]
	# it = iter(da)
	# l = zip(it, it, it)
	# # latent_space_transition(l)
	# perform_latent_space_arithmatics(l)

class waspSlicer(nn.Module):
   def __init__(self, opt, ngpu=1, pstart = 0, pend=1):
       super(waspSlicer, self).__init__()
       self.ngpu = ngpu
       self.pstart = pstart
       self.pend = pend
   def forward(self, input):
       output = input[:,self.pstart:self.pend].contiguous()
       return output