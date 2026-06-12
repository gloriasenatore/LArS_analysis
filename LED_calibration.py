## Program for LArS PMT calibration with LED.
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
        
    print("\n ################################################################### \n LArS LED calibration analysis program - Gloria Senatore UZH \n ################################################################### \n")
        
    print("\n Number of files: " + str(len(args.filenames)))
    
    dfs = [data_io.import_tree(filename, store_traces=True) for filename in args.filenames]
    
    popts = []
    perrs = []
    all_traces = []
    peak_valley_ratios = []
    
    for df, filename in zip(dfs, args.filenames):
        
        sum_traces = functions.integrate_interval(df, cfg)
        all_traces.append(sum_traces)
        
        bins = cfg["LED_calibration"]["bins"]
        _range = (cfg["LED_calibration"]["range_low"], cfg["LED_calibration"]["range_up"])
        hist = plt.hist(sum_traces, bins=bins, range=_range)
        centers = (hist[1][:-1] + hist[1][1:]) / 2
        
        
        popt, perr, peak_valley_ratio = data_calibration.fit_LED_calibration(centers, hist[0], cfg, bins, _range) 
        print("Best fit params: a0, x00, sigma0, a1, delta, sigma1, a2: " + str(popt) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr, popt)]))
        print("\mu_{SPE}: " + str(popt[1]+popt[4]) + "  " + "\mu_{DPE}: " + str(popt[1]+2.*popt[4]) + "  " + "\sigma_{DPE}: " + str(np.sqrt(2.)*popt[5]))
        
        popts.append(popt)
        perrs.append(perr)
        peak_valley_ratios.append(peak_valley_ratio)
        
        
        with open("observables/"+functions.make_output_name(filename, prefix="obs", ext=".txt", others="LED_calib"), "a") as f:
                f.write("SPE_"+str(cfg["LED_calibration"]["lower_boundary_integration"])+"_"+str(cfg["LED_calibration"]["upper_boundary_integration"])+ "\t" + str(popt[1]+popt[4]) + "\t +- \t" + str(np.sqrt(perr[1]**2.+perr[4]**2.)) + "\n" +
                       "peak_valley_"+str(cfg["LED_calibration"]["lower_boundary_integration"])+"_"+str(cfg["LED_calibration"]["upper_boundary_integration"])+ "\t" + str(peak_valley_ratio)+ "\n")
        
        '''
        The following fit uses a model in which all params are fixed. It was a cross-check. Results are compatible with the previous model.
        popt, perr = data_calibration.fit_LED_calibration(centers, hist[0], cfg, model=fit_models.combined_gaus_LED_calib_free) 
        print("Best fit params: a0, x00, sigma0, a1, x01, sigma1, a2, x02, sigma2: " + str(popt) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr, popt)]))
        '''
        
    
    peak_valley_ratio = data_plot.plot_LED_calibration(centers, all_traces, bins, _range, popts, perrs, functions.make_output_name(functions.sum_run_string(args.filenames), prefix="hist_calib", ext=".png", others=str(cfg["LED_calibration"]["lower_boundary_integration"])+"_"+str(cfg["LED_calibration"]["upper_boundary_integration"])), peak_valley_ratios, ylim=(7e-1, 6e4), fixed_delta=True)
            

if __name__ == "__main__":
    main()