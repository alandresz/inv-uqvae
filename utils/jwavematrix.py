# Python Modules:
import numpy as np 
from scipy.interpolate import interp1d
from tqdm import tqdm # progress bar for loops

# JAX modules
import jax 
from jax import jit
from jax import numpy as jnp

# j-wave modules
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.geometry import Sensors
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation


###############################################################################
def createJForwMatdotdet(Ns,Nt,dx,nx,dsa,vs,to,tf,LBW): # 
    """Creating Forward Model-based Matrix for point sensors
    """ 
    N = nx*8
    thresh = 6 
    Aj = build_sym_matrix(N,dx,vs,Ns,dsa,nx,Nt,to,tf,thresh)
    if LBW:
        Aj = applyDIR2JWM(to,tf,Nt,Ns,Aj)
    
    return Aj

###############################################################################
def SensorMaskCartCircleArcJ(circle_radius, circle_arc, num_sensor_points):
    """
    Matrix with the Ns locations (num_sensor_points) of the sensors arranged 
    on a circunference arc (circle_arc) with a radius circle_radius
    """
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:(len(th) - 1)]  # Angles
    # position of the center of the sensors:
    posSens = np.array([np.cos(th), np.sin(th)])*circle_radius  
    posSens = posSens.astype(np.float32)
    return posSens # (2,Ns)

###############################################################################
# Resampling time
def resamptime(Time,Sensordata,t):
    jS = np.asarray(Sensordata.T)
    tj = np.asarray(Time)
    Ns = jS.shape[0]
    Nt = t.shape[0]
    #pd = jnp.zeros((Ns,Nt)) # [Ns,Nt]
    pd = np.zeros((Ns,Nt)).astype(np.float32) # [Ns,Nt]
    for i1 in range(0,Ns):
        fp=interp1d(tj[:],jS[i1,:])
        aux=fp(t)
        #pd = pd.at[i1,:].set(aux)
        pd[i1,:]=aux
    return pd

##############################################################################
def build_sym_matrix(N,dx,vs,Ns,dsa,nx,Nt,to,tf,thresh):
    """
    Model-based Matrix using j-wave -> A: (Ns*Nt,N) assuming a OAT system with
    a symmetrical detection subsystem.
    P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    N: number of pixels in the j-wave 2-D Domain
    dx: pixel size  in the x direction [m]
    vs: speed of sound (homogeneous medium with density = 1000 kg/m3) [m/s]
    Ns: number of detectors
    dsa: radius of the circunference where the detectors are placed [m]
    nx: number of pixels in the x direction for a 2-D image region
    Nt: number of time samples
    to: initial time [s]
    tf: final time [s] 
    thresh: threshold the matrix to remove small entries and make it more 
    sparse 10**(-thresh)
    
    References:
        [1] A. Hauptmann, et al., "Deep learning in PAT: current approaches 
        and future directions", J. Bio. Opt. 25, p. 112903 (2020).
        [2] A. Stanziola, et al. "j-Wave: An open-source differentiable wave 
        simulator", arXiv (2022).
        [3] N. Awasthi, et al., "Deep Neural Network Based Sinogram 
        Super-resolution and Bandwidth Enhancement for Limited-data PAT",
        IEEE TUFFC. 67, pp. 2660-2673 (2020).
        [4] B. Treeby, et al. "k-Wave: MATLAB toolbox for the simulation and 
        reconstruction of PA  wave-fields", J. Biomed. Opt. 15, 021314 (2010).
    """    
    
    # Domain
    jN, jdx = (N, N), (dx, dx)
    domain = Domain(jN, jdx)
    xg = np.arange(0,N)
    yg = np.arange(0,N)
    Xgrid, Ygrid = np.meshgrid(xg,yg,indexing='ij') # normalized grid position
    
    # Acoustic medium
    medium = Medium(domain=domain, sound_speed=vs)
    
    # Time
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    Time = time_axis.to_array()
    jdt = time_axis.dt
    
    # Sensors
    #num_sensors = Ns
    posSens = SensorMaskCartCircleArcJ(dsa, 360, Ns)
    y = posSens[0,:]/dx + N/2
    x = posSens[1,:]/dx + N/2
    x = x.astype(int)
    y = y.astype(int)
    #x, y = points_on_circle(num_sensors, int(np.round(dsa/dx)), (N//2, N//2))
    sensors_positions = (x, y)
    sensors = Sensors(positions=sensors_positions)
    
    # Compile simulation
    @jit
    def compiled_simulator(medium, p0):
        a = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
        return a
    
    # Initialise system matrix
    A = np.zeros((Ns*Nt,nx**2)).astype(np.float32)
    A = A.T
    
    # Ouptut axis time
    t = np.linspace(to, tf, Nt).astype(np.float32)
    val = 1.0
    
    # Create initial pressure distribution
    p0 = jnp.zeros((N,N)).astype(jnp.float32);
    corx=N//2-nx//2;
    cory=N//2-nx//2;
    p0 = p0.at[corx,cory].set(val)
        
    # Set the initial condition (PA source term)
    p0 = 1.0 * jnp.expand_dims(p0, -1)
    p0 = FourierSeries(p0, domain)
        
    # Run the jwave simulation
    sensors_data = compiled_simulator(medium, p0)[..., 0] # [jNt,Ns]
    sd1 = np.asarray(sensors_data)
    #sd1 = sd1.T  # [Ns,jNt]   
    
    # Taking advantage of the symmetry of the OAT system
    sd2 = np.zeros(sd1.shape); # [jNt,Ns] 
    sensor_distance0 = np.sqrt((Xgrid[corx,cory]-x)**2 + (Ygrid[corx,cory]-y)**2) # [Ns,] 
    
    cont = -1; # MM row number
    with tqdm(total=nx**2, leave=True) as pbar:
        for kkx in range(nx):
            for kky in range(nx):
                cont = cont + 1;
            
                corx=N//2-nx//2+kkx;
                cory=N//2-nx//2+kky;
                
                r = np.sqrt((Xgrid[corx,cory]-x)**2 + (Ygrid[corx,cory]-y)**2) # [Ns,] 
                r1 = np.abs(r - sensor_distance0) # [Ns,] 
                
                # Threshold the matrix to remove small entries and make it more sparse
                if thresh>0: 
                    r1[abs(r1)<10**(-thresh)] = 0
                
                ind = np.ceil(r1*dx/vs/jdt); # [Ns,] 
                
                for ks in range(0,Ns):
                    sdks = sd1[:,ks]*np.sqrt(sensor_distance0[ks]/r[ks]) # [jNt,Ns]
                    
                    if r[ks] >= sensor_distance0[ks]:
                        sd2[:,ks] = np.roll(sdks,int(ind[ks]),axis=0)
                    else:
                        sd2[:,ks] = np.roll(sdks,int(-ind[ks]),axis=0)
                    
                # Record row cont
                pd = resamptime(Time,sd2,t) # [Ns,Nt]
                pd = pd.ravel() # [Ns*Nt,1]
                A[cont,:] = pd
                
                pbar.update()
                
    return np.asarray(A.T)

##############################################################################
def applyDIR2JWM(to,tf,Nt,Ns,A):
    
    print('Applying detector impulse response to j-wave matrix...');
    t = np.linspace(to, tf, Nt) # time grid
    ti = t-to
    # Detector impulse response (limited bandwidth)
    from utils.transducers import getECO_IR
    from scipy.linalg import convolution_matrix
    #from scipy import sparse
    #from scipy.sparse import csc_matrix # Column sparse
    source_freq = 2.25e6
    mu = ti[Nt//2]#2.04e-6
    sigma = 0.24e-6  
    impP = getECO_IR(ti,source_freq,mu,sigma)
    #impP = impP*-1
    MDIR = convolution_matrix(impP,Nt,'same')
    MDIR = MDIR.astype(np.float32)
    #An = sparse.kron(np.eye(Ns,dtype='float32'),MDIR)@csc_matrix(A,dtype='float32')
    An = np.kron(np.eye(Ns,dtype='float32'),MDIR)@A
    
    return An

##############################################################################