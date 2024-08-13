### v0.1

from dataclasses import dataclass
import numpy as np
#import scipy as sp
from trainuqvae import train_uqvae
from uqvae import NNmodel
import torch
from tqdm import tqdm
from utils.quality import FoM
from utils.genmatrices import genAj

#########################
@dataclass
class TrainingConfig:
    
    train = True
    pretrain = False
    ptepochs = 5
    continuetrain = False
    
    precalcA = False
    usesavedAj = True
    
    predict = False
    numtest = 500
    
    # UQ-VAE parameters
    image_size = 128  # the image size and shape
    lamba = 0.5 # sacling parameters
    l = 0.5e-3 # [m] characteristic length
    s0 = 0.25 # prior standard deviation
    eta0 = 0.5 # prior expected value
    etae = 0 # noise mean value
    semin = 1e-3 # min value of the noise standard deviation
    semax = 5e-3 # max value of the noise standard deviation
    
    # Training parameters    
    train_batch_size = 20
    num_epochs = 200
    learning_rate = 1e-6
    val_percent = 0.2
    valid_loss_min = 100
    
    # OAT setup parameters
    Ns = 36         # number of detectors
    Nt = 1024       # number of time samples
    dx = 115e-6     # pixel size  in the x direction [m] 
    nx = 128        # number of pixels in the x direction for a 2-D image region
    Rs = 44e-3     # radius of the circunference where the detectors are placed [m]
    arco = 360      # arc of the circunferencewhere the detectors are placed
    vs = 1490       # speed of sound [m/s]
    to = 21.5e-6    # initial time [s].
    T = 25e-6       # durantion of the time window [s].
    tf = to + T     # final time [s] 
    LBW = True      # Apply detector impulse reponse?
    
    # 
    cache_dir = './data/' 
    traindate = '17jul24'
    
    datadate = '24mar24F'
    
    plotresults = False 
    
    ckp_last='uqvae' + traindate + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='uqvae_best' + traindate + '.pth'
    
    logfilename = 'TrainingLog_UQVAE' + traindate + '.log' 

###############################################################################
def gettestndata(cache_dir,datadate):
    
    print('Obtaining data for testing...')
    
    Xt = np.load(cache_dir + 'Xdast' + datadate + '.npy')
    Yt = np.load(cache_dir + 'Yt' + datadate + '.npy') 
    #Zt = np.load(cache_dir + 'ESt' + datadate + '.npy') 
    
    Xt = Xt.astype(np.float32)
    Yt = Yt.astype(np.float32)
    #Zt = Zt.astype(np.float32)

    print('done')
    
    return Xt,Yt#,Zt
  
##############################################################################
def predict(config):
    
    numtest = config.numtest
    device = 'cpu'
    net = NNmodel().to(device=device)
    Xt,Yt = gettestndata(config.cache_dir,config.datadate)
    x = Xt[0:numtest,:,:]
    y = Yt[0:numtest,:,:]
    #z = Zt[0:numtest,:,:]
    #
    inp = torch.as_tensor(x)
    inp = inp.to(device=device)
    inp = inp.type(torch.float32)
    checkpoint = torch.load(config.ckp_best,map_location=torch.device(device))
    net.load_state_dict(checkpoint['state_dict'])
    m, logs2 = net.predict(inp)
    
    return x,y,m,logs2

##############################################################################
def quality(test,recons):
      
    numtest,H,W = test.shape
    SSIM = np.zeros((numtest,))
    PC = np.zeros((numtest,))
    RMSE = np.zeros((numtest,))
    PSNR = np.zeros((numtest,))
    
    for i in tqdm(range(numtest)):
        SSIM[i],PC[i],RMSE[i],PSNR[i] = FoM(test[i,:,:],recons[i,:,:])
        if PC[i] == 'nan':
            PC[i] == 0
    
    print('Results:')    
    print('SSIM: ',np.mean(SSIM),np.std(SSIM))
    print('PC: ', np.mean([~np.isnan(PC)]),np.std([~np.isnan(PC)]))
    print('RMSE: ', np.mean(RMSE),np.std(RMSE))
    print('PSNR: ', np.mean(PSNR),np.std(PSNR))
        
    return SSIM, PC, RMSE, PSNR

##############################################################################
if __name__ == '__main__':
    
    # Training hyperparameters
    config = TrainingConfig()
    
    #if config.precalcA:
    #    Aj = genAj(config)
    #    Aj = sp.sparse.csc_matrix(Aj,dtype='float32')
    #    sp.sparse.save_npz(config.cache_dir+'Aj'+config.traindate+'.npz',Aj)
        
    if config.train:
        EV,TLV,VLV = train_uqvae(config)
        np.savez('lossUQVAE'+config.traindate+'.npz',E=EV,T=TLV,V=VLV)
    
    if config.predict:
        x,y,pm,plogs2 = predict(config)
        SSIM, PC, RMSE, PSNR = quality(y,pm)
        np.savez('qualityUQVAE'+config.traindate+'.npz',SSIM=SSIM,PC=PC,RMSE=RMSE,PSNR=PSNR)
