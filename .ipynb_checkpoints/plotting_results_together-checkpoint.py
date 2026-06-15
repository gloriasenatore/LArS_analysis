## Program for plotting together data of different runs for LArS.
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
from collections import defaultdict
import data_io
import pandas as pd
import data_calibration
import functions
import data_plot
import fit_models


import argparse

def main():
    
    print("\n ################################################################### \n LArS PMT analysis: plotting results together - Gloria Senatore UZH \n ################################################################### \n")
    
    with open("config.json", "r") as f:
        cfg = json.load(f)
    
    print("\n Plotting small pulses calibration plot")
    SPE_small_pulses = functions.get_values_from_folder("observables/black_test_cell", obs="SPE")
    print(SPE_small_pulses)
    data_plot.plot_together_obs(SPE_small_pulses, filename="black_test_cell_calib_small_pulses.png", with_time=True)
    
    print("\n Plotting LED calibration plot")
    LED_calib = functions.get_values_from_folder("observables/black_test_cell", obs="SPE_100_115", filelist="observables/black_test_cell/valid_LED_calib_filelist.txt")
    print(LED_calib)
    data_plot.plot_together_obs(LED_calib, filename="black_test_cell_calib_LED.png", with_time=True)
    
    if_put_together_small_pulses_and_LED_calib = True
    if if_put_together_small_pulses_and_LED_calib:
        calib_values = {**SPE_small_pulses, **LED_calib}
        data_plot.plot_together_obs(calib_values, filename="black_test_cell_calib_all_together.png", with_time=True)
        
    print("\n Plotting VUV light-yield")
    light_yield_pre_PID = functions.get_values_from_folder("observables/black_test_cell", obs="light_yield_pre_PID", include_LED_high=False)
    print(light_yield_pre_PID)
    
    light_yield_post_PID = functions.get_values_from_folder("observables/black_test_cell", obs="light_yield_post_PID", include_LED_high=False)
    print(light_yield_post_PID)
    
    data_plot.plot_together_obs(light_yield_pre_PID, with_time=True, obs="light_yield", filename="black_test_cell_VUV_light_yield_pre_PID.png")
    post_dataset = data_plot.plot_together_obs(light_yield_post_PID, with_time=True, obs="light_yield", filename="black_test_cell_VUV_light_yield_post_PID.png")
    data_plot.plot_together_obs(light_yield_pre_PID, with_time=True, obs="light_yield", filename="black_test_cell_VUV_light_yield_pre_&_post_PID.png", additional_dataset=post_dataset)
    
    print("\n Plotting visible light-yield")
    visible_light_yield = functions.get_values_from_folder("observables/black_test_cell", obs="light_yield", include_LED_high="only")
    print(visible_light_yield)
    data_plot.plot_together_obs(visible_light_yield, with_time=True, obs="light_yield", filename="black_test_cell_visible_light_yield.png")
    
    print("\n Plotting triplet lifetime")
    triplet_lifetime_ER_alpha = functions.get_values_from_folder("observables/black_test_cell", obs="triplet_lifetime_ER+alpha", include_LED_high=False)
    print(triplet_lifetime_ER_alpha)
    
    triplet_lifetime_ER = functions.get_values_from_folder("observables/black_test_cell", obs="triplet_lifetime_ER", include_LED_high=False)
    print(triplet_lifetime_ER_alpha)
    
    triplet_lifetime_alpha = functions.get_values_from_folder("observables/black_test_cell", obs="triplet_lifetime_alpha", include_LED_high=False)
    print(triplet_lifetime_ER_alpha)
    
    data_plot.plot_together_triplet_lifetime(triplet_lifetime_ER_alpha, triplet_lifetime_ER, triplet_lifetime_alpha, with_time=True, filename="black_test_cell_triplet_lifetime.png")
    
    
if __name__ == "__main__":
    main()