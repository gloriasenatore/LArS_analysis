## Functions to define models for histograms fitting

import numpy as np

def gaus(x,a,x0,sigma):
    '''
        gaus model with free amplitude, scale and width
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaus_fixed(x,a,mu,N, sigma):
    '''
        gaus model with free amplitude, and fixed scale (N*mu) and width (sqrt(N)*mu)
        This is useful to find the multiPE peaks in a spectrum (the N peak is N*SPE peak)
    '''
    return a*np.exp(-(x-(N*mu))**2/(2*(np.sqrt(N)*sigma)**2))

def expp(x,a,tau):
    return a*np.exp(-x/tau)

def calib_fit_model(xvalues, a0,x00,sigma0, a1,x01,sigma1, a2,x02,sigma2, a3, a4, a5, ae,tau):
    '''
        The model used for calibration with small pulses is the combination of three gaussians (for pedestal, SPE and DPE) with all params free,
        other three gaussians (for multiPE peaks till n. 5) and one exponential
    '''
    return [gaus(xx, a0, x00, sigma0)+gaus(xx, a1, x01, sigma1) + gaus(xx, a2,x02,sigma2) + gaus_fixed(xx, a3, x01, 3, sigma1) +
            gaus_fixed(xx, a4, x01, 4, sigma1) + gaus_fixed(xx, a5, x01, 5, sigma1) + expp(xx, ae, tau) 
            for xx in xvalues]

def gaus_list(xvalues, a, x0, sigma):
    return [gaus(xx, a, x0, sigma) for xx in xvalues]

def exp_list(xvalues, a, tau):
    return [expp(xx, a, tau) for xx in xvalues]

def combined_gaus_with_exp(x, a0,x00,sigma0, ae,tau):
    gn = gaus(x, a0, x00, sigma0)
    ee = expp(x, ae, tau)
    
    f = gn+ee
    
    return f

def comb_list_gaus_with_exp(xvalues, a0,x00,sigma0, ae,tau):
    return [gaus(xx, a0, x00, sigma0) +
            expp(xx, ae, tau)
            for xx in xvalues]


def combined_gaus_LED_calib_delta(xvalues, a0,x00,sigma0, a1,delta,sigma1, a2):
    '''
        This model used for calibration with LED is the combination of three gaussians (for pedestal, SPE and DPE), in which delta (distance
        between mean value of pedestal and mean value of SPE) is fixed: muSPE = muNoise + delta, muDPE = muNoise + 2delta, 
        sigmaDPE = sqrt(2)*sigmaSPE
    '''
    return (
    gaus(xvalues, a0, x00, sigma0)
    + gaus(xvalues, a1, x00 + delta, sigma1)
    + gaus(xvalues, a2, x00 + 2*delta, np.sqrt(2.)*sigma1)
    )

def combined_gaus_LED_calib_free(xvalues, a0,x00,sigma0, a1,x01,sigma1, a2,x02,sigma2):
    '''
        This model is just for check if we obtain results compatible to the previous model. In this model, all parameters are free
    '''
    return [gaus(xx, a0, x00, sigma0) + gaus(xx, a1, x01, sigma1) + gaus(xx, a2, x02, sigma2)
            for xx in xvalues]