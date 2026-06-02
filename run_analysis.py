## Program for LArS PMT post-processed waveforms calibration and light-yield estimation.
## Gloria Senatore (University of Zurich)

import matplotlib.pyplot as plt
import uproot
import json
import legendstyles
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare, goodness_of_fit
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.ndimage import gaussian_filter1d
import data_io
import pandas as pd
import data_calibration
import functions
import data_plot
plt.style.use(legendstyles.LEGEND)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

import argparse

def main():

    parser = argparse.ArgumentParser(
        description="ROOT files analysis"
    )
    
    parser.add_argument(
        "filenames",
        nargs="+",
        help="ROOT files list"
    )
    args = parser.parse_args()
    
    for filename in args.filenames:
        print("\n ################################ \n Analyzing file " + str(filename))
        df = data_io.import_tree(filename)
        
        areas = data_calibration.small_pulses(df)
        histo_areas = functions.make_histo(areas, bins=200, _range=(0,200))

        ## Obtain a rough estimation of the peak positions (pedestal and SPE):
        print("Initial estimation of peak position")
        histo_areas_smooth = gaussian_filter1d(histo_areas[0], sigma=2) # Smooth the spectrum and ease the peak finding
        peaks = data_calibration.peak_position(histo_areas_smooth, prominence=np.max(histo_areas_smooth)*0.005)
        print("First peaks at " + str(peaks[0]) + " ADC*ns, with heights " +str(peaks[2]) +" counts")
        
        ## Fitting the histogram in order to obtain SPE peak and calibrate the spectrum
        print("Low histogram fitting")
        centers = (histo_areas[1][:-1] + histo_areas[1][1:]) / 2

        with open("config.json", "r") as f:
            cfg = json.load(f)

        calib_popt_tot, calib_perr_tot = data_calibration.fit_hist(centers, histo_areas[0], peaks[0], peaks[2], cfg)
        print("Best fit params: " + str(calib_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(calib_perr_tot, calib_popt_tot)]))

        data_plot.plot_calib_spectrum(centers, histo_areas[0], calib_popt_tot, calib_perr_tot, functions.make_output_name(filename), ylim=(1e1, 2e6))
        
        ## Calibrate the spectrum and fit the alpha bump to obtain light-yield
        bins= cfg["alpha"]["bins"]
        _range=(cfg["alpha"]["range_low"], cfg["alpha"]["range_high"])
        histo_alpha = functions.make_histo(df["Integral"]/calib_popt_tot[4], bins=bins, _range=_range)
        centers = (histo_alpha[1][:-1] + histo_alpha[1][1:]) / 2
        histo_alpha_smooth = gaussian_filter1d(histo_alpha[0], sigma=2) # Smooth the spectrum and ease the peak finding
        alpha_peak = data_calibration.peak_position(histo_alpha[0], height=(0,1e3), prominence=20)
        alpha_peak_PE = alpha_peak[0][0]*(_range[1]-_range[0])/bins+_range[0]
        print("\n ################################ \n Fitting histogram to find light-yield")
        print("Guess peak at " + str(alpha_peak_PE) + " PE, with heights " +str(alpha_peak[2][0]) +" counts")
        alpha_popt_tot, alpha_perr_tot = data_calibration.fit_hist_alpha_bump(centers[alpha_peak[0][0]-50:], histo_alpha[0][alpha_peak[0][0]-50:], alpha_peak_PE, alpha_peak[2][0], cfg)
        print("Best fit params: " + str(alpha_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(alpha_perr_tot, alpha_popt_tot)]))
        
        data_plot.plot_alpha_spectrum(centers, histo_alpha[0], alpha_popt_tot, alpha_perr_tot, functions.make_output_name(filename, prefix="light_yield_pre_PID"), ylim=(1, 1e4), interval=(_range[1]-_range[0])/bins)
        
        ##TODO: write to txt file the important parameters (SPE, light-yield, triplet lifetime, ...)

if __name__ == "__main__":
    main()