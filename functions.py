## Other useful functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re

def make_histo(data, bins, _range):
    histo = plt.hist(data, bins=bins, range=_range)
    return histo

def residuals(obs, model):
    res = []
    for i in range (len(obs)):
        res.append( (obs[i] - model[i]) / np.sqrt(model[i]) )
        
    return res


def residuals_staked_wvfs(obs, model, n_wvfs):
    sigma = np.std(waveforms, axis=0) / np.sqrt(N)
    res = []
    for i in range (len(obs)):
    pulls = (data - model)/sigma
        
        
def reduced_chi_square(obs, model, centers, popt_tot):
    chi2 = 0.

    for i in range (len(obs)):
        chi2 += (obs[i] - model[i])**2./model[i]
        
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

    for event in df["Evtnb"]:
        lefte = df["Leftedge"].loc[event][0]
        for i in range(len(sum_traces)-lefte):
            sum_traces[i] += df["Traces"].loc[event][lefte+i]/np.max(df["Peaks_area"].loc[event])
            
    return sum_traces
    
