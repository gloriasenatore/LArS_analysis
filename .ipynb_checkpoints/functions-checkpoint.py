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



def read_from_file(file_name, obs="SPE", N=1):   
    with open(file_name, "r") as f:
        for line in f:
            field = line.split()
            if field[0] == obs:
                var = float(field[N])
                break
            
    return var
    
