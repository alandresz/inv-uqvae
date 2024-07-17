# Importing the necessary libraries:
import torch
from torch import nn
from torch.nn import functional as F
from utils.genmatrices import getdiagGs

# ---------------------------------------------------------------------------
class ConvBN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.lrelu(x)
        
        return x

# ---------------------------------------------------------------------------
class ConvBND(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding='same')
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.lrelu(x)
        
        return x

# ---------------------------------------------------------------------------
class ConvBNDlast(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding='same')
    
    def forward(self, x):
        x = self.conv(x)
        
        return x

# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding='same')
        self.bnorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding='same')
        self.bnorm2 = nn.BatchNorm2d(in_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bnorm1(y)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.bnorm2(y)
        y = self.lrelu(y)
        
        return x + y    

# ---------------------------------------------------------------------------
class ResidualBlock1(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),padding='same')
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding='same')
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bnorm1(y)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.bnorm2(y)
        y = self.lrelu(y)
        
        return self.conv0(x) + y 

# ---------------------------------------------------------------------------
class ConvTRBNCat(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels*2,out_channels,kernel_size=(3,3),padding='same')
    
    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x1 = self.bnorm(x1)
        x1 = self.lrelu(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return x

# ---------------------------------------------------------------------------
class NNmodel(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.unet1 = ResidualBlock1(1,128,(3,3))
        self.unet2 = ConvBN(128,256,(3,3),(2,2),(1,1))
        self.unet3 = ResidualBlock(256,(3,3))
        self.unet4 = ConvBN(256,512,(3,3),(2,2),(1,1))
        self.unet5 = ResidualBlock(512,(3,3))
        self.unet6 = ConvBN(512,1024,(3,3),(2,2),(1,1))
        self.unet7 = ResidualBlock(1024,(3,3))
        self.unet8 = ConvBN(1024,1024,(3,3),(2,2),(1,1))
        self.unet9 = ResidualBlock(1024,(3,3))
        self.unet10 = ConvBN(1024,1024,(3,3),(2,2),(1,1))
        self.unet11 = ResidualBlock(1024,(3,3))
        self.unet12 = ConvBND(1024,1024,(3,3))
        self.unet13 = ConvTRBNCat(1024,1024,(2,2),(2,2))
        self.unet14 = ResidualBlock(1024,(3,3))
        self.unet15 = ConvTRBNCat(1024,1024,(2,2),(2,2))
        self.unet16 = ResidualBlock(1024,(3,3))
        self.unet17 = ConvTRBNCat(1024,512,(2,2),(2,2))
        self.unet18 = ResidualBlock(512,(3,3))
        self.unet19 = ConvTRBNCat(512,256,(2,2),(2,2))
        self.unet20 = ResidualBlock(256,(3,3))
        
        #self.unet23 = ConvBNDlast(256,2,(3,3))  # MODIFICACION
        
        self.unet21 = ConvTRBNCat(256,128,(2,2),(2,2))
        self.unet22 = ResidualBlock(128,(3,3))
        self.unet23 = ConvBNDlast(128,2,(3,3))    

    def forward(self, x):
        # input: (B,128,128)
        
        x = torch.unsqueeze(x,1)  # (B,1,128,128)
        x1 = self.unet1(x)        # (B,128,128,128)
        x2 = self.unet2(x1)       # (B,256,64,64)
        x2 = self.unet3(x2)       # (B,256,64,64)
        x3 = self.unet4(x2)       # (B,512,32,32)
        x3 = self.unet5(x3)       # (B,512,32,32)
        x4 = self.unet6(x3)       # (B,1024,16,16)
        x4 = self.unet7(x4)       # (B,1024,16,16)
        x5 = self.unet8(x4)       # (B,1024,8,8)
        x5 = self.unet9(x5)       # (B,1024,8,8)
        x6 = self.unet10(x5)      # (B,1024,4,4)
        x6 = self.unet11(x6)      # (B,1024,4,4)
        x6 = self.unet12(x6)      # (B,1024,4,4)
        x7 = self.unet13(x6,x5)   # (B,1024,8,8)
        x7 = self.unet14(x7)      # (B,1024,8,8)
        x8 = self.unet15(x7,x4)   # (B,1024,16,16)
        x8 = self.unet16(x8)      # (B,1024,16,16)
        x9 = self.unet17(x8,x3)   # (B,512,32,32)
        x9 = self.unet18(x9)      # (B,512,32,32)
        x10 = self.unet19(x9,x2)  # (B,256,64,64)
        x10 = self.unet20(x10)    # (B,256,64,64)
        
        #x11 = self.unet23(x10)  # MODIFICACION
        
        x11 = self.unet21(x10,x1) # (B,128,128,128)
        x11 = self.unet22(x11)    # (B,128,128,128)
        x11 = self.unet23(x11)    # (B,2,128,128)
        
        
        m = x11[:,0,:,:]          # (B,128,128)
        s = x11[:,1,:,:]          # (B,128,128)
        
        return m,s
    
    def loss_function(self, y, m, logs2, lamba, A, etae, z, diagLe, iGp0, eta0, logdetGp0, device):
        
        B = y.shape[0]
        
        # Calculating term 1
        diagGs = getdiagGs(logs2,device) # [B,nx**2]
        
        term11 = torch.sum(torch.log(diagGs + 2e-45),1) # [B,]
        
        diagLs2 = torch.sqrt(diagGs) # [B,nx**2]
        
        #term12 = torch.sum(torch.square(torch.mul(diagLs2,torch.flatten(m-y,start_dim=1))),1) # [B,]
        print("m:", m.shape)
        print("y: ", y.shape)
        term12 = torch.sum(torch.square(torch.einsum('bi,bi->bi',diagLs2,torch.flatten(m-y,start_dim=1))),1) # [B,]
        
        term1 = ((1-lamba)/lamba) * (term11 + term12) # [B,]
        
        # Calculating term 2
        eps = torch.normal(0,torch.eye(y.shape[1])) # [nx,nx]
        eps = eps.to(device=device)
        
        p0m = torch.flatten(m,start_dim=1) + torch.einsum('bi,i->bi', diagLs2,torch.flatten(eps)) # [B,nx**2]
        
        #Sm = torch.matmul(A,p0m.T).T # ((Ns*Nt,nx*nx) @ (nx*nx,B)).T = (B,Ns*Nt)
        Sm = torch.einsum('ij,bj ->bi',A,p0m) # (B,Ns*Nt)
        maxSm = Sm.max(dim=-1,keepdim=True)[0]
        Sm = Sm/maxSm # Normalization
        
        aux = torch.flatten(z,start_dim=1)
        
        
        SS = torch.flatten(z,start_dim=1)-Sm-etae # (B,Ns*Nt)
        term2 = torch.sum(torch.square(torch.einsum('i,bi->bi', diagLe, SS)),1) # [B,]      
        
        # Calculating term 3
        term31 = torch.sum(torch.einsum('ij,bj->bi', iGp0,diagGs),1) # [B,]
        
        #term32 = torch.sum(torch.square(torch.mul(diagLs2,torch.flatten(m,start_dim=1)-eta0)),1) # [B,]
        term32 = torch.sum(torch.square(torch.einsum('ij,bj->bi',torch.linalg.cholesky(iGp0),torch.flatten(m,start_dim=1)-eta0)),1) # [B,]
                
        term3 = term31 + term32 + logdetGp0 - term11

        # Determination of the final loss
        
        loss = (1/B) * torch.sum(term1 + term2 + term3) # [1]
        
        return loss
    
    def loss_functionPT(self, y, logy2, m, logs2):
        
        y = torch.unsqueeze(y,1)
        logy2 = torch.unsqueeze(logy2,1)
        T = torch.cat([y, logy2], dim=1)
        
        m = torch.unsqueeze(m,1)
        logs2 = torch.unsqueeze(logs2,1)
        P = torch.cat([m, logs2], dim=1)
        
        lossPT =F.mse_loss(P, T)
        
        return lossPT
        
    def predict(self, x):
        
        with torch.no_grad():
            m, logs2 = self.forward(x)
               
        return m.detach().to("cpu").numpy(), logs2.detach().to("cpu").numpy()

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    
    B = 10 # batch size
    nx = 128 # number of pixels in the x direction for a 2-D image region

    x = torch.randn(B,nx,nx)

    model = NNmodel()
    # print(model)
    # Number of net parameters
    NoP = sum(p.numel() for p in model.parameters())
    print(f"The number of parameters of the network to be trained is: {NoP}")  

    s,m = model(x)
    
    print(x.shape)
    print(s.shape)
    print(m.shape)
