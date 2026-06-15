## Program for LArS PMT post-processed waveforms and visible-light-yield estimation.
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
        
    print("\n ################################################################### \n LArS PMT post-processed waveform analysis for visible-light-yield estimation program - Gloria Senatore UZH \n ################################################################### \n")
        
    print("\n Number of files: " + str(len(args.filenames)))
    
    if cfg["analysis"]["concatenate_files"]:
        name = functions.sum_run_string(args.filenames)
        dfs = pd.concat([data_io.import_tree(filename, store_traces=cfg["import_tree"]["store_traces"]) for filename in args.filenames], axis=0, ignore_index=True)
        dfs["Evtnb"] = dfs.index.to_list()
        print("\n Total number of waveforms to analyze: " + str(len(dfs)))
        dfs = [dfs]
    else:
        dfs = [data_io.import_tree(filename, store_traces=cfg["import_tree"]["store_traces"]) for filename in args.filenames]
    
    
    for df, filename in zip(dfs, args.filenames):
        if cfg["analysis"]["concatenate_files"]:
            print("\n ################################ \n Summing events in all files. Files: " + str(name))
        else:
            print("\n ################################ \n Treating each file saparately. Analyzing file " + str(filename))
            name = filename
        
        if_calibrate = cfg["analysis"]["if_calibrate"]
        
        if if_calibrate:
        
            areas = data_calibration.small_pulses(df)
            histo_areas = functions.make_histo(areas, bins=200, _range=(0,200))

            ## Obtain a rough estimation of the peak positions (pedestal and SPE):
            print("Initial estimation of peak position")
            histo_areas_smooth = gaussian_filter1d(histo_areas[0], sigma=2) # Smooth the spectrum and ease the peak finding
            peaks, widths, heights = data_calibration.peak_position(histo_areas_smooth, prominence=np.max(histo_areas_smooth)*0.005)
            print("First peaks at " + str(peaks) + " ADC*ns, with heights " +str(heights) +" counts")

            ## Fitting the histogram in order to obtain SPE peak and calibrate the spectrum
            print("Low histogram fitting")
            centers = (histo_areas[1][:-1] + histo_areas[1][1:]) / 2

            calib_popt_tot, calib_perr_tot = data_calibration.fit_hist(centers, histo_areas[0], peaks, heights, cfg)
            print("Best fit params: " + str(calib_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(calib_perr_tot, calib_popt_tot)]))

            data_plot.plot_calib_spectrum(centers, histo_areas[0], calib_popt_tot, calib_perr_tot, functions.make_output_name(name, others="LED_high"), ylim=(1e1, 2e6))
            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt", others="LED_high"), "a") as f:
                f.write("SPE \t" + str(calib_popt_tot[4]) + "\t +- \t" + str(calib_perr_tot[4]) + "\n")
                
                
        using_LED_calib = cfg["analysis"]["using_LED_calib"]
        file_LED_calib = cfg["analysis"]["file_LED_calib"]
        if using_LED_calib == False:
            calib_SPE = data_io.read_from_file("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt", others="LED_high"), obs="SPE", N=1)
        else:
            calib_SPE = data_io.read_from_file("observables/"+functions.make_output_name(file_LED_calib, prefix="obs", ext=".txt", others="LED_calib"), obs="SPE_100_115", N=1)
            
        bins= cfg["LED_high"]["bins"]
        _range=(cfg["LED_high"]["range_low"], cfg["LED_high"]["range_high"])
        
        
        histo_integral = functions.make_histo(df["Integral"]/calib_SPE, bins=bins, _range=_range)
        centers = (histo_integral[1][:-1] + histo_integral[1][1:]) / 2
        #histo_integral_smooth = gaussian_filter1d(histo_integral[0], sigma=2) # Smooth the spectrum and ease the peak finding
        integral_peak = data_calibration.peak_position(histo_integral[0], height=(0,5e5), prominence=20)
        integral_peak_PE = integral_peak[0][0]*(_range[1]-_range[0])/bins+_range[0]

        print("\n ################################ \n Fitting histogram to find light-yield")
        print("Guess peak at " + str(integral_peak_PE) + " PE, with heights " +str(integral_peak[2][0]) +" counts")
        interval = (_range[1]-_range[0])/bins
        
        integral_popt_tot, integral_perr_tot = data_calibration.fit_hist_visible_light_bump(centers[int((integral_peak_PE-10.)/interval):int((integral_peak_PE+10.)/interval)], histo_integral[0][int((integral_peak_PE-10.)/interval):int((integral_peak_PE+10.)/interval)], integral_peak_PE, integral_peak[2][0], cfg)
        print("Best fit params: " + str(integral_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(integral_perr_tot, integral_popt_tot)]))
        
        if using_LED_calib == False:
            data_plot.plot_visible_light_spectrum(centers, histo_integral[0], integral_popt_tot, integral_perr_tot, functions.make_output_name(name, prefix="visible", ext=".png", others="LED_high"), ylim=(1, 5e5), interval=interval, filename_hist=functions.make_output_name(name, prefix="visible_light", ext=".txt", others="LED_high"), save_to_file=cfg["LED_high"]["save_to_file"])

            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt", others="LED_high"), "a") as f:
                f.write("light_yield \t" + str(integral_popt_tot[1]) + "\t +- \t" + str(integral_perr_tot[1])  + "\n")

        else:
            data_plot.plot_visible_light_spectrum(centers, histo_integral[0], integral_popt_tot, integral_perr_tot, functions.make_output_name(name, prefix="visible", ext=".png", others="LED_high_LED_calib"), ylim=(1, 1e5), interval=interval, filename_hist=functions.make_output_name(name, prefix="visible_light", ext=".txt"), save_to_file=cfg["LED_high"]["save_to_file"])

            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt", others="LED_high"), "a") as f:
                f.write("light_yield_LED_calib \t" + str(integral_popt_tot[1]) + "\t +- \t" + str(integral_perr_tot[1])  + "\n")
                
        
            

if __name__ == "__main__":
    main()