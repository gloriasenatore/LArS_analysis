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
    
    for df, filename in zip(dfs, args.filenames):
        
        sum_traces = functions.integrate_interval(df, cfg)
        
        peaks, widths, heights = data_calibration.peak_position(sum_traces, prominence=np.max(sum_traces)*0.005)
        print("First peak at " + str(peaks) + " ADC*ns, with heights " + str(heights) +" counts")
        
        plt.hist(sum_traces, bins=200, range=(-30, 230), histtype='step', label='710 mV')
        plt.yscale('log')
        plt.savefig("plots/LED_calibration/"+functions.make_output_name(filename), bbox_inches='tight')
            

if __name__ == "__main__":
    main()