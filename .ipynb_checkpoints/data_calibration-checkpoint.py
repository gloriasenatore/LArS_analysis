## Functions for spectrum calibration

from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit
from fit_models import calib_fit_model, combined_gaus_with_exp, gaus, expp, combined_gaus_LED_calib_delta, combined_gaus_LED_calib_free

def small_pulses(df):
    areas = []
    for i in df.index.to_list():
        for peak in df["Peaks_area"][i]:
            if(peak < 600): areas.append(peak)
    
    return areas
    
            
def peak_position(counts,height=0,prominence=20):
    peaks, props = find_peaks(
        counts,
        prominence=prominence,
        distance=15,
        width=4,
        height=height
    )
    
    idx = np.argsort(props["prominences"])[::-1]
    peaks = peaks[idx]
    widths = props["widths"][idx]
    heights = props["peak_heights"][idx]
    
    return peaks, widths, heights


def fit_hist(xdata, ydata, peaks_guess, heights_guess, cfg):
    '''
        Fit performed with scipy curve_fit using the least squares method
    '''
    noise_peak_guess = peaks_guess[0]
    SPE_peak_guess   = peaks_guess[1]

    noise_width_guess = cfg["noise"]["width"]
    SPE_width_guess   = cfg["spe"]["width"]

    noise_height_guess = heights_guess[0]
    SPE_height_guess   = heights_guess[1]

    bounds_low = [
        noise_height_guess * (1 - cfg["noise"]["bound_frac_height"]),
        noise_peak_guess   * (1 - cfg["noise"]["bound_frac_height"]),
        noise_width_guess  * (1 - cfg["noise"]["width_frac"]),

        SPE_height_guess * (1 - cfg["spe"]["bound_frac_height"]),
        SPE_peak_guess   * (1 - cfg["spe"]["bound_frac_height"]),
        SPE_width_guess  * (1 - cfg["spe"]["width_frac"]),

        cfg["dpe"]["height_bounds_low"],
        2 * SPE_peak_guess * 0.9,
        np.sqrt(2) * SPE_width_guess * 0.9,

        0,
        0,
        0,

        0,
        cfg["exp"]["tau_low"]
    ]

    bounds_high = [
        noise_height_guess * (1 + cfg["noise"]["bound_frac_height"]),
        noise_peak_guess   * (1 + cfg["noise"]["bound_frac_height"]),
        noise_width_guess  * (1 + cfg["noise"]["width_frac"]),

        SPE_height_guess * (1 + cfg["spe"]["bound_frac_height"]),
        SPE_peak_guess   * (1 + cfg["spe"]["bound_frac_height"]),
        SPE_width_guess  * (1 + cfg["spe"]["width_frac"]),

        cfg["dpe"]["height_bounds_high"],
        2 * SPE_peak_guess * 1.1,
        np.sqrt(2) * SPE_width_guess * 1.1,

        1e4,
        1e4,
        1e4,

        5e4, #1.2e4
        cfg["exp"]["tau_high"]
    ]
    
    popt_tot, pcov = curve_fit(calib_fit_model, xdata, ydata, bounds=(bounds_low, bounds_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))

    return popt_tot, perr_tot
    

def fit_hist_alpha_bump(xdata, ydata, peak_guess, height_guess, cfg, model=combined_gaus_with_exp):
    
    bound_low = [
        height_guess * (1 - cfg["alpha"]["bound_frac_height"]),
        peak_guess * (1 - cfg["alpha"]["bound_frac_peak"]),
        cfg["alpha"]["width_low"],
        cfg["alpha"]["exp_a_low"],
        cfg["alpha"]["exp_tau_low"],
    ]
    
    bound_high = [
        height_guess * (1 + cfg["alpha"]["bound_frac_height"]),
        peak_guess * (1 + cfg["alpha"]["bound_frac_peak"]),
        cfg["alpha"]["width_high"],
        cfg["alpha"]["exp_a_high"],
        cfg["alpha"]["exp_tau_high"]
    ]
    
    if model==gaus:
        bound_low = [
            height_guess * (1 - cfg["alpha"]["bound_frac_height"]),
            peak_guess * (1 - cfg["alpha"]["bound_frac_peak"]),
            cfg["alpha"]["width_low"]
        ]

        bound_high = [
            height_guess * (1 + cfg["alpha"]["bound_frac_height"]),
            peak_guess * (1 + cfg["alpha"]["bound_frac_peak"]),
            cfg["alpha"]["width_high"]
        ]
    
    popt_tot, pcov = curve_fit(model, xdata, ydata, bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))
    
    return popt_tot, perr_tot


def fit_fprompt(xdata, ydata, peaks_guess, heights_guess, cfg, interval = 0.005):
    
    peak_ER = peaks_guess[0]
    peak_alpha = peaks_guess[1]
    
    height_ER = heights_guess[0]
    height_alpha = heights_guess[1]
    
    ## ER and alpha Fprompt are fitted separately, ER first and alpha later:
    
    fit_interval_ER_low = np.maximum(0, int(peak_ER * (1 - cfg["PID_analysis"]["fit_interval_frac"])/interval))
    fit_interval_ER_high = int(peak_ER * (1 + cfg["PID_analysis"]["fit_interval_frac"])/interval)
    
    bound_low = [
        height_ER * (1 - cfg["PID_analysis"]["bound_frac_height"]),
        peak_ER * (1 - cfg["PID_analysis"]["bound_frac_peak"]),
        cfg["PID_analysis"]["width_low"]
    ]
    
    bound_high = [
        height_ER * (1 + cfg["PID_analysis"]["bound_frac_height"]),
        peak_ER * (1 + cfg["PID_analysis"]["bound_frac_peak"]),
        cfg["PID_analysis"]["width_high"]
    ]
    
    popt_ER, pcov = curve_fit(gaus, xdata[fit_interval_ER_low:fit_interval_ER_high], ydata[fit_interval_ER_low:fit_interval_ER_high], bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata[fit_interval_ER_low:fit_interval_ER_high],1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_ER = np.sqrt(np.diag(pcov))
    
    fit_interval_alpha_low = int(peak_alpha * (1 - cfg["PID_analysis"]["fit_interval_frac"])/interval)
    fit_interval_alpha_high = np.minimum(1, int(peak_alpha * (1 + cfg["PID_analysis"]["fit_interval_frac"])))
    if fit_interval_alpha_high == 1: fit_interval_alpha_high = -1
    else: fit_interval_alpha_high = fit_interval_alpha_high/interval
    
    bound_low = [
        height_alpha * (1 - cfg["PID_analysis"]["bound_frac_height"]),
        peak_alpha * (1 - cfg["PID_analysis"]["bound_frac_peak"]),
        cfg["PID_analysis"]["width_low"]
    ]
    
    bound_high = [
        height_alpha * (1 + cfg["PID_analysis"]["bound_frac_height"]),
        peak_alpha * (1 + cfg["PID_analysis"]["bound_frac_peak"]),
        cfg["PID_analysis"]["width_high"]
    ]
    
    popt_alpha, pcov = curve_fit(gaus, xdata[fit_interval_alpha_low:fit_interval_alpha_high], ydata[fit_interval_alpha_low:fit_interval_alpha_high], bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata[fit_interval_alpha_low:fit_interval_alpha_high],1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_alpha = np.sqrt(np.diag(pcov))
    
    return popt_ER, perr_ER, popt_alpha, perr_alpha



def fit_stacked_waveforms(ydata, cfg, sigma, model=expp, n_samples=800):
    
    samples = np.linspace(0, int(n_samples*10), int(n_samples), endpoint=False)
    
    lower_boundary = cfg["triplet_lifetime"]["lower_boundary_fit"]
    upper_boundary = cfg["triplet_lifetime"]["upper_boundary_fit"]
    
    popt, pcov = curve_fit(expp, samples[lower_boundary:upper_boundary], ydata[lower_boundary:upper_boundary], bounds=[[0.1,500],[1000, 4000]], absolute_sigma=True, maxfev=cfg["fit"]["maxfev"]) #sigma= sigma[lower_boundary:upper_boundary]
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr



def fit_LED_calibration(xdata, ydata, cfg, bins, _range, model=combined_gaus_LED_calib_delta):
    
    bound_low = [
        1e3, #a0
        -5., #x00
        1.,  #sigma0
        1e1, #a1
        20., #delta
        10., #sigma1
        0.   #a2
    ]
    
    bound_high = [
        1e5,
        5.,
        10.,
        1e4,
        80.,
        60.,
        3e2
    ]
    
    if model == combined_gaus_LED_calib_free:
        bound_low = [
        1e3, #a0
        -5., #x00
        1.,  #sigma0
        1e2, #a1
        20., #x01
        10., #sigma1
        1.,   #a2
        60., #x02
        15.  #sigma2
        ]

        bound_high = [
            1e5,
            5.,
            50.,
            1e4,
            80.,
            60.,
            1e2,
            100.,
            20.
        ]
        
    
    popt_tot, pcov = curve_fit(model, xdata, ydata, bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))
    
    interval = (_range[1]-_range[0]) / bins
    
    #peak_valley_ratio = np.popt_tot[3]/np.min(model(xdata,*popt_tot)[(int(popt_tot[1]/interval-_range[0])):int((popt_tot[1]+popt_tot[4])/interval-_range[0])]) #this is evaluated on the model
    peak_valley_ratio = np.max(ydata[int((popt_tot[1]+popt_tot[4])/interval-_range[0]-5):int((popt_tot[1]+popt_tot[4])/interval-_range[0]+5)])/np.min(ydata[(int(popt_tot[1]/interval-_range[0])):int((popt_tot[1]+popt_tot[4])/interval-_range[0])]) #this is evaluated on the data histogram
    
    return popt_tot, perr_tot, peak_valley_ratio



def fit_hist_visible_light_bump(xdata, ydata, peak_guess, height_guess, cfg, model=gaus):
    
    bound_low = [
        height_guess * (1 - cfg["LED_high"]["bound_frac_height"]),
        peak_guess * (1 - cfg["LED_high"]["bound_frac_peak"]),
        cfg["LED_high"]["width_low"]
    ]
    
    bound_high = [
        height_guess * (1 + cfg["LED_high"]["bound_frac_height"]),
        peak_guess * (1 + cfg["LED_high"]["bound_frac_peak"]),
        cfg["LED_high"]["width_high"]
    ]
    
    popt_tot, pcov = curve_fit(model, xdata, ydata, bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))
    
    return popt_tot, perr_tot