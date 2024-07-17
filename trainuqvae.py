# Importing the necessary libraries:

import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#import os

from utils.SLcheckpoint import load_ckp, save_ckp

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.nn import functional as F

from uqvae import NNmodel
from utils.genmatrices import genpriordist, genA, gendiagLe

###############################################################################
class ClearCache:
    # Clearing GPU Memory
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

###############################################################################
def gettraindata(cache_dir,datadate):
    
    print('Obtaining data for training...')
    
    #X = np.load(cache_dir + 'Xdas' + datadate + '.npy')
    #Y = np.load(cache_dir + 'Y' + datadate + '.npy') 
    #Z = np.load(cache_dir + 'ES' + datadate + '.npy') 
    
    X = np.load(cache_dir + 'X0_128'  + '.npy')
    Y = np.load(cache_dir + 'Y0_128' + '.npy') 
    Z = np.load(cache_dir + 'Z0_128' + '.npy') 
    
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    Z = Z.astype(np.float32)

    print('done')
    
    return X,Y,Z

###############################################################################
class OAImageDataset(Dataset):
    def __init__(self, X, Y, Z):
        super(OAImageDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.Z = Z

    def __getitem__(self, item):
        return self.X[item, :, :], self.Y[item, :, :], self.Z[item, :, :]

    def __len__(self):
        return self.X.shape[0]

###############################################################################
def get_trainloader(X, Y, Z, val_percent, batch_size): 
        
    dataset_train = OAImageDataset(X, Y, Z)
    
    # Split into train / validation partitions
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True) # for local uncomment this
    #loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True) # for google_colab uncomment this
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args) #drop_last=True, drop the last batch if the dataset size is not divisible by the batch size.
    
    return train_loader, val_loader, n_train, n_val 

###############################################################################
def train_uqvae(config):
    
    # Use the context manager
    with ClearCache():
        # Train parameters    
        batch_size = config.train_batch_size
        epochs = config.num_epochs
        lr = config.learning_rate
        ckp_last = config.ckp_last
        ckp_best = config.ckp_best
        cache_dir = config.cache_dir
        datadate = config.datadate
        logfilename = config.logfilename
        continuetrain = config.continuetrain
        plotresults = config.plotresults
        pretrain = config.pretrain
   
        # Set device
        device = "[set_device]"
    
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device to be used: {device}")
    
        # Create the network    
        net = NNmodel().to(device=device)
        #print(net)
    
        # Number of net parameters
        NoP = sum(p.numel() for p in net.parameters())
        print(f"The number of parameters of the network to be trained is: {NoP}")    
    
        # Define loss function and optimizer and the the learning rate scheduler
        optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2,threshold=0.005,eps=1e-6,verbose=True)
       
        # Get data
        print('Loading data...')
        X,Y,Z = gettraindata(cache_dir,datadate)
        
        # Create data loader
        val_percent = config.val_percent
        X = torch.as_tensor(X).type(torch.float32) 
        Y = torch.as_tensor(Y).type(torch.float32) 
        Z = torch.as_tensor(Z).type(torch.float32) 
        
        #X=X[0:100,:,:]; Y=Y[0:100,:,:]; Z=Z[0:100,:,:]; # ---------------------OJO!!!!!!!!!!!!
        
        train_loader, val_loader, n_train, n_val = get_trainloader(X, Y, Z, val_percent, batch_size)
        
        # Parameters and matrices of the UQVAE loss model
        lamba = config.lamba  # UQ-VAE scaling parameter
        etae = config.etae # noise mean value
        eta0 = config.eta0 # prior mean value
        
        print('Creating Prior Distribution Matrix...')
        Gp0 = genpriordist(config) # Covariance of the Gaussian Ornstein-Uhlenbeck distribution used as prior.
        Gp0 = torch.as_tensor(Gp0).type(torch.float32).to(device=device) # [nx**2,nx**2]
        print( "Gp0: ",Gp0.shape )
        iGp0 = torch.linalg.inv(Gp0) # [nx**2,nx**2]
        logdetGp0 = torch.log(torch.linalg.det(Gp0) + 2e-45) #  ----> el determinante da muy bajo o cero en float32!!!!!!
        print('done')
        
        print('Creating Le...')
        diagLe = gendiagLe(config.Ns*config.Nt,config.semin,config.semax) # [Ns*Nt,]
        diagLe = diagLe.to(device=device) # [Ns*Nt,]
        print('done')
        
        if config.usesavedAj:
            print('Loading Forward Model Matrix created with j-Wave...')
            import scipy as sp
            Ao = sp.sparse.load_npz(cache_dir + 'Aj' + config.traindate + '.npz') # Forward model matrix
        else:
            print('Creating Forward Model Matrix...')
            #Ao =  np.ones((config.Nt*config.Ns,config.nx**2),dtype=np.float32)
            Ao = genA(config) # Forward model matrix
        Ao = Ao.todense()
        Ao = torch.as_tensor(Ao).type(torch.float32).to(device=device) # [Ns*Nt,nx**2]
        print('done')
    
        # Initialize logging and initialize weights or continue a previous training 
    
        if continuetrain:
            net, optimizer, last_epoch, valid_loss_min = load_ckp(ckp_last, net, optimizer)
            print('Values loaded:')
            #print("model = ", net)
            print("optimizer = ", optimizer)
            print("last_epoch = ", last_epoch)
            print("valid_loss_min = ", valid_loss_min)
            print("valid_loss_min = {:.6f}".format(valid_loss_min))
            start_epoch = last_epoch + 1
            lr = optimizer.param_groups[0]['lr']
            logging.basicConfig(filename=logfilename,format='%(asctime)s - %(message)s', level=logging.INFO)
            logging.info(f'''Continuing training:
                Epochs:                {epochs}
                Batch size:            {batch_size}
                Initial learning rate: {lr}
                Training size:         {n_train}
                Validation size:       {n_val}
                Device:                {device.type}
                ''')
        else:
            # Apply the weights_init function to randomly initialize all weights
            net.apply(initialize_weights)
            start_epoch = 1
            valid_loss_min = config.valid_loss_min
            logging.basicConfig(filename=logfilename, filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
            logging.info(f'''Starting training:
                Epochs:                {epochs}
                Batch size:            {batch_size}
                Initial learning rate: {lr}
                Training size:         {n_train}
                Validation size:       {n_val}
                Device:                {device.type}
                ''')
    
    
                # Print model
                # print(net)

        # Begin training
        TLV=np.zeros((epochs,)) #vector to record the train loss per epoch 
        VLV=np.zeros((epochs,)) #vector to record the validation loss per epoch
        EV=np.zeros((epochs,)) # epoch vector to plot later
        global_step = 0
    
        #for epoch in range(epochs):
        for epoch in range(start_epoch, start_epoch+epochs):
            net.train() # Let pytorch know that we are in train-mode
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            sigma_mean = 0.0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs+start_epoch-1}', unit='sino') as pbar:
                for x,y,z in train_loader:
                    # clear the gradients
                    optimizer.zero_grad(set_to_none=True)
                    # input, truth image and measured sinogram to device
                    x = x.to(device=device)
                    x = x.type(torch.float32)
                    y = y.to(device=device)
                    y = y.type(torch.float32)
                    z = z.to(device=device)
                    z = z.type(torch.float32)
                    # net prediction
                    m, logs2 = net.forward(x)
                    # calculate loss
                    if (pretrain)&(epoch<config.ptepochs):
                        logy2 = torch.log(torch.square(torch.rand(batch_size,config.nx,config.nx).uniform_(config.semin,config.semax)))
                        logy2 = logy2.to(device=device)
                        train_loss = net.loss_functionPT(y,logy2,m,logs2)
                    else:
                        train_loss = net.loss_function(y, m, logs2, lamba, Ao, 
                                                       etae, z, diagLe, iGp0, eta0, logdetGp0, device)
                    # credit assignment
                    train_loss.backward()
                    # update model weights
                    optimizer.step()
                
                    pbar.update(x.shape[0])
                    global_step += 1
                    
                    epoch_loss += train_loss.item()
                
                    pbar.set_postfix(**{'loss (batch)': train_loss.item()})
            
                epoch_train_loss = epoch_loss / len(train_loader)

            # Evaluation round
            with torch.no_grad():
                for xv,yv,zv in tqdm(val_loader, total=len(val_loader), desc='Validation round', position=0, leave=True):
                    # input and truth to device
                    xv = xv.to(device=device)
                    xv = xv.type(torch.float32)
                    yv = yv.to(device=device)
                    yv = yv.type(torch.float32)
                    zv = zv.to(device=device)
                    zv = zv.type(torch.float32)
                    # net prediction
                    mv, logs2v = net.forward(xv)
                    # calculate loss
                    val_loss = F.mse_loss(yv, mv)
                    epoch_val_loss += val_loss.item()
                    sigma_mean += torch.mean(torch.sqrt(torch.exp(logs2v)))
                    
            epoch_val_loss = epoch_val_loss / len(val_loader)
            sigma_mean = sigma_mean / len(val_loader)
            
            # Scheduler ReduceLROnPlateau
            scheduler.step(epoch_val_loss)
        
            # logging validation score per epoch
            #logging.info(f'''Epoch: {epoch} - Validation score: {np.round(epoch_val_loss,5)}''')
            logging.info(f'''Epoch: {epoch} - Validation score: {np.round(epoch_val_loss,5)} - Sigma mean: {np.round(sigma_mean.detach().to("cpu").numpy(),5)}''')
        
            # print training/validation statistics 
            #print('\n Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            #    epoch,
            print('\n Training Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
                    epoch_train_loss,
                    epoch_val_loss
                    ))
        
            # Loss vectors for plotting results
            TLV[epoch-start_epoch]=epoch_train_loss
            VLV[epoch-start_epoch]=epoch_val_loss
            EV[epoch-start_epoch]=epoch
        
            # create checkpoint variable and add important data
            checkpoint = {
                    'epoch': epoch,
                    'valid_loss_min': epoch_val_loss,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
        
            # save checkpoint
            save_ckp(checkpoint, False, ckp_last, ckp_best)
        
        
            # save the model if validation loss has decreased
            if epoch_val_loss <= valid_loss_min:
                print('\n Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss),'\n')
                # save checkpoint as best model
                save_ckp(checkpoint, True, ckp_last, ckp_best)
                valid_loss_min = epoch_val_loss
                logging.info(f'Val loss deccreased on epoch {epoch}!')
    
        if plotresults:
            plt.figure();
            plt.grid(True,linestyle='--')
            plt.xlabel('epoch'); plt.ylabel('Loss')
            plt.plot(EV,TLV,'--',label='Train Loss')
            plt.plot(EV,VLV,'-',label='Val Loss')
            plt.legend(loc='best',shadow=True, fontsize='x-large')
    
        return EV,TLV,VLV
               
###############################################################################
def initialize_weights(m):
    if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data,0.0,0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)):
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

###############################################################################
if __name__=='__main__':
    
    img_size = 128  # the image size and shape
    batch = 20
    X = torch.rand((batch,img_size,img_size)).uniform_(0,1)
    print(X.shape)
    
    net = NNmodel()
    
    NoP = sum(p.numel() for p in net.parameters())
    print(f"The number of parameters of the network to be trained is: {NoP}")  
    
    m, s = net(X)
    
    print(m.shape)
    print(s.shape)
