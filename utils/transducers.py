import numpy as np

##############################################################################
def apply_ramp(signal,
               dt: float,
               center_freq: float,
               warmup_cycles: float = 3):
    """Processes the signal $s(t)$ as

    $$
    s(t)\cdot \text{min}(1, f_0t/\sigma)
    $$

    Args:
        signal (np.ndarray): [description]
        dt (float): [description]
        center_freq (float): $f_0$
        warmup_cycles (float, optional): $\sigma$. Defaults to 3.

    Returns:
        np.ndarray: [description]
    """
    # Raise ValueError if the center frequency is negative
    if center_freq <= 0:
        raise ValueError(
            f"Center frequency must be positive, got {center_freq}")

    # Raise an error if the signal is not 1D
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {signal.ndim}D")

    t = np.arange(signal.shape[0]) * dt
    period = 1 / center_freq
    ramp_length = warmup_cycles * period
    return signal * np.where(t < ramp_length, (t / ramp_length), 1.0)

##############################################################################
def gaussian_window(signal, time, mu, sigma):
    """Returns the gaussian window

    $$
    s(t)\cdot \exp \left( - \frac{(t-\mu)^2}{\sigma^2} \right)
    $$

    Args:
        signal (np.ndarray): $s(t)$
        time (np.ndarray): $t$
        mu (float): $\mu$
        sigma (float): $\sigma$

    Returns:
        np.ndarray: [description]
    """
    return signal * np.exp(-((time - mu)**2) / sigma**2)

##############################################################################
def getECO_IR(time,source_freq,mu,sigma):
    """Get ECO impulse response of a commercial detector
    
    Args:
        time (np.ndarray): $t$
        source_freq (float): $fs$ # 2.25e6
        mu (float): $\mu$ # 2.04e-6
        sigma (float): $\sigma$ # 0.3e-6  (ECO case)  ## 0.24e-6 --> PA case 
    
    Returns:
        np.ndarray: [description]
    """
    dt = time[1]-time[0]
    s1 = 1 * np.sin(2 * np.pi * source_freq * time)
    signal = gaussian_window(apply_ramp(s1, dt, source_freq), time, mu, sigma)
    
    return signal

