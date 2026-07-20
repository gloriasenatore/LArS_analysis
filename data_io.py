## Functions for the file IO

import uproot
import pandas as pd
from datetime import datetime

def open_file(filename):
    my_file = open(filename, "r")
    lines = my_file.readlines()
    data = [int(e) for e in lines]
    my_file.close()
    return data


def import_tree(path, store_traces=False, from_evt=None, to_evt=None):
    tree = uproot.open(path)
    
    branches = tree["T1"].arrays(
        ["IntegralWave", "RMS", "Area", "Leftedge",
         "Rightedge", "Height", "Position",
         "Width", "Prompt", "Evtnb"],
        library="np", entry_start=from_evt, entry_stop=to_evt
    )
    
    mask = branches["RMS"] < 3
    
    d = {
        "Integral": branches["IntegralWave"][mask], "RMS": branches["RMS"][mask], "Peaks_area": branches["Area"][mask], "Height":branches["Height"][mask], "Leftedge": branches["Leftedge"][mask], "Rightedge": branches["Rightedge"][mask], "Prompt": branches["Prompt"][mask], "Evtnb":branches["Evtnb"][mask], "Position": branches["Position"][mask], "Width":branches["Width"][mask]
    }
    print("Root read")
    
    if store_traces:
        traces = tree["T1"]["Trace"].array(library="np", entry_start=from_evt, entry_stop=to_evt) #Reading and storing the all pulses need some time
        d["Traces"] = traces[mask]
        print("Traces stored")
        
    df = pd.DataFrame(d)
    #print("Total waveforms number " + str(len(df)))
    #initial_len = len(df)
    #df = df[(df["RMS"] < 3)] #quality cut applied to remove noisy events
    #print("Waveforms number after quality cut: " + str(len(df)) + " (" + str(len(df)/initial_len*100.) + "%)")
    print("Dataframe created")
    
    return df


def read_from_file(file_name, obs="SPE", return_error=False):   
    last_value = None
    last_error = None
    time_to_return = None

    with open(file_name, "r") as f:
        for line in f:
            if line.startswith(obs):
                parts = line.split()
                last_value = float(parts[1])
                if return_error==True: last_error = float(parts[3])
                
            if line.startswith("time "):
                timestamp = line[len("time "):].strip()
                time_to_return = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")


    if return_error==True: return last_value, last_error, time_to_return
    else: return last_value

