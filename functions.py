## Other useful functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd

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


def make_output_name(filepath, prefix="hist_calib", ext=".png"):
    """
        Output is the name for the png file with the calibrated SPE
    """
    base = os.path.basename(filepath)

    match = re.search(r"(R\d+(?:\+R\d+)*)", base)
    if not match:
        raise ValueError(f"No pattern like R<number> found in {filepath}")

    run_tag = match.group(1)

    return f"{prefix}_{run_tag}{ext}"


def sum_run_string(filenames):
    runs = []

    for filename in filenames:
        basename = os.path.basename(filename)
        match = re.search(r'(R\d+)\.root$', basename)

        if match:
            runs.append(match.group(1))
        else:
            raise ValueError(f"Run number not found in filename: {filename}")

    return "+".join(runs)


def stack_waveforms(df, n_samples=800):
    sum_traces = np.zeros(shape=(n_samples))
    zeros = np.full((len(df["Evtnb"]), n_samples), np.nan)
    wvfs_bin = pd.DataFrame(index = df["Evtnb"], columns=np.arange(n_samples), data=zeros)

    for event in df["Evtnb"]:
        lefte = df["Leftedge"].loc[event][0]
        k=0
        for i in range(len(sum_traces)-lefte):
            value = df["Traces"].loc[event][lefte+i]/np.max(df["Peaks_area"].loc[event])
            sum_traces[i] += value
            wvfs_bin.loc[event, k] = value
            k=k+1
            
    n_bin = wvfs_bin.count(axis=0)
    sigma_mean = wvfs_bin.std(axis=0) / np.sqrt(n_bin)
    
    '''
    bin_test = 100

    col = wvfs_bin.iloc[:, bin_test].dropna()

    print("N =", len(col))
    print("mean =", col.mean())
    print("std =", col.std())
    print("sem =", col.std()/np.sqrt(len(col)))
    '''
    
    sigma_mean = sigma_mean.to_numpy()
            
    return sum_traces, sigma_mean
    
