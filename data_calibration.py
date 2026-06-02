## Functions for spectrum calibration

from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit
from fit_models import calib_fit_model, combined_gaus_with_exp

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

        1.2e4,
        cfg["exp"]["tau_high"]
    ]
    
    popt_tot, pcov = curve_fit(calib_fit_model, xdata, ydata, bounds=(bounds_low, bounds_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))

    return popt_tot, perr_tot
    

def fit_hist_alpha_bump(xdata, ydata, peak_guess, height_guess, cfg):
    
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
    
    popt_tot, pcov = curve_fit(combined_gaus_with_exp, xdata, ydata, bounds=(bound_low, bound_high), sigma=np.sqrt(np.maximum(ydata,1)), absolute_sigma=True, maxfev=cfg["fit"]["maxfev"])
    perr_tot = np.sqrt(np.diag(pcov))
    
    return popt_tot, perr_tot
    