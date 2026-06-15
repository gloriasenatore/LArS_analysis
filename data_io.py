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


def import_tree(path, store_traces=False):
    tree = uproot.open(path)

    integral = tree["T1"]["IntegralWave"].array(library="np")
    RMS = tree["T1"]["RMS"].array(library="np")
    area = tree["T1"]["Area"].array(library="np")
    leftedge = tree["T1"]["Leftedge"].array(library="np")
    height = tree["T1"]["Height"].array(library="np")
    prompts = tree["T1"]["Prompt"].array(library="np")
    events = tree["T1"]["Evtnb"].array(library="np")
    d = {"Integral":integral, "RMS":RMS, "Peaks_area":area, "Height":height, "Leftedge": leftedge, "Prompt": prompts, "Evtnb":events}
    
    if store_traces:
        traces = tree["T1"]["Trace"].array(library="np") #Reading and storing the all pulses need some time
        d["Traces"] = traces
        
    df_1 = pd.DataFrame(d)
    print("Total waveforms number " + str(len(df_1)))
    df = df_1[(df_1["RMS"] < 3)] #quality cut applied to remove noisy events
    print("Waveforms number after quality cut: " + str(len(df)) + " (" + str(len(df)/len(df_1)*100.) + "%)")
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

