# UQVAE
Reacreation of the work [Sahlstroem2023] in Pytorch.

Using the following training parameters,
```
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
learning_rate = 1e-5 # (initial)
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
```

we obtained the results presented below:

![plot](./results24mar24/loss24mar24.png)

![plot](./results24mar24/results24mar24_1.png)

![plot](./results24mar24/results24mar24_2.png)

Figure of merit: we got the following values over 500 testing images:
```
SSIM:  0.8187 +/- 0.0652
PC:  0.943 +/- 0.0238 
RMSE:  0.0839 +/- 0.0236
PSNR:  21.8642 +/- 2.4802
```


References:
```
@inproceedings{Sahlstroem2023,
author = {Teemu Sahlstr{\"o}m and Tanja Tarvainen},
title = {{Utilizing Variational Autoencoders in Photoacoustic Tomography}},
volume = {12379},
booktitle = {Photons Plus Ultrasound: Imaging and Sensing 2023},
editor = {Alexander A. Oraevsky and Lihong V. Wang},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {1237914},
year = {2023},
doi = {10.1117/12.2644801},
URL = {https://doi.org/10.1117/12.2644801}
}
```
