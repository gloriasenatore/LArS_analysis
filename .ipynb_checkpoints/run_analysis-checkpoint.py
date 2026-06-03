## Program for LArS PMT post-processed waveforms calibration and light-yield estimation.
## Gloria Senatore (University of Zurich)

import matplotlib.pyplot as plt
import uproot
import json
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
import fit_models

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
    
    with open("config.json", "r") as f:
        cfg = json.load(f)
    
    for filename in args.filenames:
        print("\n ################################ \n Analyzing file " + str(filename))
        df = data_io.import_tree(filename)
        
        if_calibrate = cfg["analysis"]["if_calibrate"]
        
        if if_calibrate:
        
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

            calib_popt_tot, calib_perr_tot = data_calibration.fit_hist(centers, histo_areas[0], peaks[0], peaks[2], cfg)
            print("Best fit params: " + str(calib_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(calib_perr_tot, calib_popt_tot)]))

            data_plot.plot_calib_spectrum(centers, histo_areas[0], calib_popt_tot, calib_perr_tot, functions.make_output_name(filename), ylim=(1e1, 2e6))
            with open("observables/"+functions.make_output_name(filename, prefix="obs", ext=".txt"), "a") as f:
                f.write("SPE \t" + str(calib_popt_tot[4]) + "\t +- \t" + str(calib_perr_tot[4]) + "\n")
                
                
        calib_SPE = functions.read_from_file("observables/"+functions.make_output_name(filename, prefix="obs", ext=".txt"), obs="SPE", N=1)
        bins= cfg["alpha"]["bins"]
        _range=(cfg["alpha"]["range_low"], cfg["alpha"]["range_high"])
        histo_alpha = functions.make_histo(df["Integral"]/calib_SPE, bins=bins, _range=_range)
        centers = (histo_alpha[1][:-1] + histo_alpha[1][1:]) / 2
                
                
        if_analyse_alpha_bump_pre_PID = cfg["analysis"]["if_alpha_pre_PID"]
        
        if if_analyse_alpha_bump_pre_PID:
            ## Fitting the alpha bump to obtain light-yield
            #histo_alpha_smooth = gaussian_filter1d(histo_alpha[0], sigma=2) # Smooth the spectrum and ease the peak finding
            alpha_peak = data_calibration.peak_position(histo_alpha[0], height=(0,1e3), prominence=20)
            alpha_peak_PE = alpha_peak[0][0]*(_range[1]-_range[0])/bins+_range[0]

            print("\n ################################ \n Fitting histogram to find light-yield")
            print("Guess peak at " + str(alpha_peak_PE) + " PE, with heights " +str(alpha_peak[2][0]) +" counts")
            interval = (_range[1]-_range[0])/bins
            alpha_popt_tot, alpha_perr_tot = data_calibration.fit_hist_alpha_bump(centers[int(cfg["alpha"]["fit_from"]/interval):], histo_alpha[0][int(cfg["alpha"]["fit_from"]/interval):], alpha_peak_PE, alpha_peak[2][0], cfg)
            print("Best fit params: " + str(alpha_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(alpha_perr_tot, alpha_popt_tot)]))

            data_plot.plot_alpha_spectrum(centers, histo_alpha[0], alpha_popt_tot, alpha_perr_tot, functions.make_output_name(filename, prefix="light_yield_pre_PID", ext=".png"), ylim=(1, 1e4), interval=interval, filename_hist=functions.make_output_name(filename, prefix="light_yield_pre_PID", ext=".txt"), save_to_file=cfg["alpha"]["save_to_file"])
            
            with open("observables/"+functions.make_output_name(filename, prefix="obs", ext=".txt"), "a") as f:
                f.write("light_yield_pre_PID \t" + str(alpha_popt_tot[1]) + "\t +- \t" + str(alpha_perr_tot[1])  + "\n")
                
                
        if_analyse_alpha_bump_post_PID = cfg["analysis"]["if_alpha_post_PID"]
        
        if if_analyse_alpha_bump_post_PID:
            print("\n ################################ \n Performing Particle Identification analysis")
            data_plot.plot_hist2d_Integral_Fprompt(df["Integral"]/calib_SPE, df["Prompt"], _range, functions.make_output_name(filename, prefix="hist2d_Integral_Fprompt", ext=".png"))
            df_PE_cut = df[(df["Integral"]/calib_SPE > cfg["PID_analysis"]["PE_cut"])]
            
            bins = cfg["PID_analysis"]["prompt_bins"]
            ## Fitting the fraction of prompt light spectrum to obtain the PID cut
            histo_prompt = functions.make_histo(df_PE_cut["Prompt"], bins=bins, _range=(0, 1))
            centers = (histo_prompt[1][:-1] + histo_prompt[1][1:]) / 2
            guess_peaks = data_calibration.peak_position(histo_prompt[0], height=0, prominence=20)
            guess_peaks_pos = [x*1./bins for x in guess_peaks[0]]
            print("Guess peak at " + str(guess_peaks_pos) + " Fprompt, with heights " +str(guess_peaks[2]) +" counts")
            
            #data_plot.plot_Fprompt(df_PE_cut, functions.make_output_name(filename, prefix="hist_Fprompt", ext=".png"))
            
            popt_ER, perr_ER, popt_alpha, perr_alpha = data_calibration.fit_fprompt(centers, histo_prompt[0], guess_peaks_pos, guess_peaks[2], cfg, interval=1./bins)
            print("Best fit params: ER:" + str(popt_ER) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_ER, popt_ER)]) + "\n")
            print("Best fit params: alpha:" + str(popt_alpha) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_alpha, popt_alpha)]) + "\n")
            
            gaus_sum = fit_models.gaus_list(centers[50:190], popt_ER[0], popt_ER[1], popt_ER[2]) + fit_models.gaus_list(centers, popt_alpha[0], popt_alpha[1], popt_alpha[2])
            
            print(np.argmin(gaus_sum)/bins)
            

if __name__ == "__main__":
    main()