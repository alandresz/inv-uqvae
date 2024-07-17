# Pyhton modules:
import math
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

###############################################################################
def FoM(Po,Pp):
    
    SSIM=structural_similarity(Po,Pp) 
    PC=stats.pearsonr(Po.ravel(),Pp.ravel())[0]  
    RMSE=math.sqrt(mean_squared_error(Po,Pp))
    PSNR=peak_signal_noise_ratio(Po,Pp)
    
    return SSIM,PC,RMSE,PSNR
