## Other useful functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import data_io

def make_histo(data, bins, _range):
    histo = plt.hist(data, bins=bins, range=_range)
    return histo

def residuals(obs, model, sigma=None):
    res = []
    for i in range (len(obs)):
        if sigma is None:
            res.append( (obs[i] - model[i]) / np.sqrt(model[i]) )
        else:
            res.append( (obs[i] - model[i]) / model[i] * 100. ) #This is used when counts are small, hence Poisson approx cannot apply, like for the normalized staked waveform exp fit.
        
    return res
        
        
def reduced_chi_square(obs, model, centers, popt_tot, sigma=None):
    chi2 = 0.

    for i in range (len(obs)):
        if sigma is None:
            chi2 += (obs[i] - model[i])**2./model[i]
        else:
            chi2 += (obs[i] - model[i])**2./sigma[i]**2.
        
    ndof = len(centers) - 1 - len(popt_tot)
    
    return chi2 / ndof


def make_output_name(filepath, prefix="hist_calib", ext=".png", others=None):
    """
        Output is the name for the png file with the calibrated SPE
    """
    base = os.path.basename(filepath)

    match = re.search(r"(R\d+(?:\+R\d+)*)", base)
    if not match:
        raise ValueError(f"No pattern like R<number> found in {filepath}")

    run_tag = match.group(1)

    if others is None:
        return f"{prefix}_{run_tag}{ext}"
    else:
        return f"{prefix}_{run_tag}_{others}{ext}"


def sum_run_string(filenames):
    runs = []

    for filename in filenames:
        basename = os.path.basename(filename)
        match = re.search(r'(R\d+)', basename)

        if match:
            runs.append(match.group(1))
        else:
            raise ValueError(f"Run number not found in filename: {filename}")

    return "+".join(runs)


def stack_waveforms(df, n_samples=800):
    sum_traces = np.zeros(shape=(n_samples))
    wvfs_bin = np.full((len(df), n_samples), np.nan)

    for row_idx, event in enumerate(df["Evtnb"]):
        lefte = df["Leftedge"].loc[event][0]
        trace = np.asarray(df["Traces"].loc[event])
        norm = np.max(df["Peaks_area"].loc[event])

        aligned = trace[lefte:] / norm
        
        L = len(aligned)

        sum_traces[:L] += aligned
        wvfs_bin[row_idx, :L] = aligned
            
    n_bin = np.sum(~np.isnan(wvfs_bin), axis=0)

    sigma_mean = (np.nanstd(wvfs_bin, axis=0, ddof=1) / np.sqrt(n_bin))

    return sum_traces, sigma_mean


def integrate_interval(df, cfg, entry="LED_calibration"):
    
    lower_boundary = cfg[entry]["lower_boundary_integration"]
    upper_boundary = cfg[entry]["upper_boundary_integration"]
            
    sum_traces = np.array([ np.sum(trace[lower_boundary:upper_boundary]) for trace in df["Traces"] ])
            
    return sum_traces


def get_values_from_folder(folder, obs="SPE", filelist=None, include_LED_high=True):
    if filelist is None and include_LED_high=="only": pattern = re.compile(r"^obs_(R\d+_LED_high)\.txt$")
    elif filelist is None and include_LED_high: pattern = re.compile(r"^obs_(R\d+(?:_LED_high)?)\.txt$")
    elif filelist is None and include_LED_high==False: pattern = re.compile(r"^obs_(R\d+)\.txt$")
    
    else:
        if isinstance(filelist, str) and filelist.endswith(".txt"):
            with open(filelist, "r") as f:
                filelist = [line.strip() for line in f if line.strip()]
                
        escaped = [re.escape(name) for name in filelist]
        pattern_str = r"^obs_(" + "|".join(escaped) + r")\.txt$"
        pattern = re.compile(pattern_str)

    values = {}
    time = {}

    for filename in os.listdir(folder):
        match = pattern.match(filename)

        if match:
            key = match.group(1)
            filepath = os.path.join(folder, filename)
            
            values[key] = data_io.read_from_file(filepath, obs, return_error=True)

    return values



def sort_items(values):
    sorted_items = sorted(
        values.items(),
        key=lambda kv: kv[1][2]
    )
    
    keys = [k for k, v in sorted_items]
    measurements = [v[0] for k, v in sorted_items]
    errs = [v[1] for k, v in sorted_items]
    times = [v[2] for k, v in sorted_items]
    
    return keys, measurements, errs, times