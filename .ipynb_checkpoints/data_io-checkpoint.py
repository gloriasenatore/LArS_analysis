## Functions for the file IO

import uproot
import pandas as pd

def open_file(filename):
    my_file = open(filename, "r")
    lines = my_file.readlines()
    data = [float(e) for e in lines]
    my_file.close()
    return data


def import_tree(path):
    tree = uproot.open(path)

    integral = tree["T1"]["IntegralWave"].array(library="np")
    RMS = tree["T1"]["RMS"].array(library="np")
    area = tree["T1"]["Area"].array(library="np")
    leftedge = tree["T1"]["Leftedge"].array(library="np")
    height = tree["T1"]["Height"].array(library="np")
    #traces = tree["T1"]["Trace"].array(library="np") #Reading and storing the all pulses need some time, you only need it if you want to visualize them
    d = {"Integral":integral, "RMS":RMS, "Peaks_area":area, "Height":height, "Leftedge": leftedge} #, "Trace":traces}
    df_1 = pd.DataFrame(d)
    print("Total waveforms number " + str(len(df_1)))
    df = df_1[(df_1["RMS"] < 3)] #quality cut applied to remove noisy events
    print("Waveforms number after quality cut: " + str(len(df)) + " (" + str(len(df)/len(df_1)*100.) + "%)")
    return df