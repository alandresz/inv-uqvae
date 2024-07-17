import numpy as np
import numpy.matlib
import torch
#from utils.OAT import createForwMatdotdetMIR
#from utils.jwavematrix import createJForwMatdotdet
from utils.OAT import createForwMatdotdetMIR
from utils.jwavematrix import createJForwMatdotdet



def asd():
    print('asdasda')

###############################################################################
def createrectgrid2D(config):
    nx = config.nx
    dx = config.dx
    N = int(nx*nx)
    x = np.linspace(-nx*dx/2,nx*dx/2,nx)
    y = np.linspace(-nx*dx/2,nx*dx/2,nx)
    xv, yv = np.meshgrid(x,y,indexing='xy')
    rj = np.zeros((2,N)) # pixel position [rj]=(2,N))
    rj[0,:] = xv.ravel()
    rj[1,:] = yv.ravel()
    rj = rj.astype(np.float32)
    return rj

###############################################################################
def genpriordist(config):

    N = config.nx**2
    rij = createrectgrid2D(config) # [2,nx**2]
    Xi = np.matlib.repmat(np.reshape(rij[0,0:],(1,len(rij[0,0:]))),N,1);
    X = np.matlib.repmat(np.reshape(rij[0,0:],(len(rij[0,0:]),1)),1,N);
    Yi = np.matlib.repmat(np.reshape(rij[1,0:],(1,len(rij[1,0:]))),N,1);
    Y = np.matlib.repmat(np.reshape(rij[1,0:],(len(rij[1,0:]),1)),1,N);
    DR = np.sqrt((X-Xi)**2+(Y-Yi)**2);
    Gpo = config.s0**2 * np.exp(-DR/config.l);
    #Gpo = torch.as_tensor(Gpo).type(torch.float32)
    
    return Gpo 

###############################################################################
def gendiagLe(NsNt,smin,smax):
    se2 = torch.square(torch.rand(NsNt).uniform_(smin,smax))
    #iGe = torch.diag(1/se2)
    #Le = torch.linalg.cholesky(iGe)
    #diagLe = torch.sqrt(iGe) 
    diagLe = torch.sqrt(1/se2) # es lo mismo que hacer Cholesky de una matriz diagonal!
    
    return diagLe.type(torch.float32) # [B,Ns*Nt]

###############################################################################
def genA(config):
    
    A = createForwMatdotdetMIR(config.Ns,config.Nt,config.dx,config.nx,config.Rs,config.arco,config.vs,config.to,config.tf)
    A = A.astype(np.float32)
    
    return A

###############################################################################
def getdiagGs(logs2,device):
    s2 = torch.exp(logs2)
    diagGs = torch.flatten(s2,start_dim=1) # (-1,nx**2)
        
    return diagGs

###############################################################################
def genAj(config):
    Aj = createJForwMatdotdet(config.Ns,config.Nt,config.dx,config.nx,config.Rs,config.vs,config.to,config.tf,config.LBW)
    Aj = Aj.astype(np.float32) 
    
    return Aj


    
