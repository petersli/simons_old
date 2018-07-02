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

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
# args = parser.parse_args(args=[])  --for ipynb
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#the ranges are the number of batches
train_loader = range(153)  
test_loader = range(2)  # 3 batches is 192 imgs, test folder only has 189

totensor = transforms.ToTensor()
def load_batch(batch_idx, istrain):
    if istrain:
        template = '/home/peterli/simons/VAE_celeba/cropped/train/%s.jpg' 
        l = [str(batch_idx*64 + i + 10190).zfill(6) for i in range(64)]  #first 190 img indices are test, the rest are train
    else:
        template = '/home/peterli/simons/VAE_celeba/cropped/test/%s.jpg' 
        l = [str(batch_idx*64 + i + 10000).zfill(6) for i in range(64)]  
    data = []
    for idx in l:
        img = Image.open(template%idx)
        data.append(np.array(img))
    data = [totensor(i) for i in data]
    return torch.stack(data, dim=0)


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



class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

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

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

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

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #print("decode")
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 2, 2)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        print("get latent var")
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        #print("FORWARD")
        mu_per, logvar_per = self.encode(x.view(-1, self.nc, self.ndf, self.ngf)) # "person" distribution 
        mu_ex, logvar_ex = self.encode(x.view(-1, self.nc, self.ndf, self.ngf)) # "expression" distribution
        z_per = self.reparametrize(mu_per, logvar_per)
        z_ex = self.reparametrize(mu_ex, logvar_ex)
        recon_per = self.decode(z_per)
        recon_ex = self.decode(z_ex)
        return recon_per, mu_per, logvar_per, recon_ex, mu_ex, logvar_ex 


model = VAE(nc=3, ngf=64, ndf=64, latent_variable_size=500)

if args.cuda:
    model.cuda()

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):
    print("train")
    model.train()
    train_loss = 0
    dataroot = random.sample(TrainingData,1)[0]

    dataset = MultipieLoader.FareMultipieExpressionTripletsFrontal(opt, root=dataroot, resize=64)
    print('# size of the current (sub)dataset is %d' %len(dataset))
    train_amount = train_amount + len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    for batch_idx, data_point in enumerate(dataloader, 0):

        gc.collect() # collect garbage
        # sample the data points: 
        # dp0_img: image of data point 0
        # dp9_img: image of data point 9, which is different in ``expression'' compare to dp0
        # dp1_img: image of data point 1, which is different in ``person'' compare to dp0
        dp0_img, dp9_img, dp1_img = data_point
        dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
        if args.cuda:
            dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
        dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )

        optimizer.zero_grad()
        recon_batch_per0, mu_per0, logvar_per0, recon_batch_ex0, mu_ex0, logvar_ex0 = model(dp0_img)
        recon_batch_per9, mu_per9, logvar_per9, recon_batch_ex9, mu_ex9, logvar_ex9 = model(dp9_img)
        recon_batch_per1, mu_per1, logvar_per1, recon_batch_ex1, mu_ex1, logvar_ex1 = model(dp0_img)

        

        loss = loss_function(recon_batch, data_point, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*64),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*64)))
    return train_loss / (len(train_loader)*64)

def test(epoch):
    print("test")
    model.eval()
    test_loss = 0
    for batch_idx in test_loader:
        data = load_batch(batch_idx, False)
     #   data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

        torchvision.utils.save_image(data.data, '../imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch.data, '../imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

    test_loss /= (len(test_loader)*64)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


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
    if args.cuda:
        z = z.cuda()
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)

# def load_last_model():
#     models = glob('../models/*.pth')
#     model_ids = [(int(f.split('_')[1]), f) for f in models]
#     start_epoch, last_cp = max(model_ids, key=lambda item:item[0])  # max returns the model_id with the largest proxy value (item)
#     model.load_state_dict(torch.load(last_cp))
#     return start_epoch, last_cp

def start_training():
    # start_epoch, _ = load_last_model()
    start_epoch = 0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(epoch)
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