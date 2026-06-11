## Program for LArS PMT post-processed waveforms calibration and light-yield estimation.
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
        
    print("\n Number of files: " + str(len(args.filenames)))
    
    if cfg["analysis"]["concatenate_files"]:
        name = functions.sum_run_string(args.filenames)
        dfs = pd.concat([data_io.import_tree(filename, store_traces=cfg["import_tree"]["store_traces"]) for filename in args.filenames], axis=0, ignore_index=True)
        dfs["Evtnb"] = dfs.index.to_list()
        print("\n Total number of waveforms to analyze: " + str(len(dfs)))
        dfs = [dfs]
    else:
        dfs = [data_io.import_tree(filename, store_traces=cfg["import_tree"]["store_traces"]) for filename in args.filenames]
    
    #for filename in args.filenames:
    for df, filename in zip(dfs, args.filenames):
        if cfg["analysis"]["concatenate_files"]:
            print("\n ################################ \n Summing events in all files. Files: " + str(name))
        else:
            print("\n ################################ \n Treating each file saparately. Analyzing file " + str(filename))
            name = filename
        
        if_calibrate = cfg["analysis"]["if_calibrate"]
        
        if if_calibrate:
        
            areas = data_calibration.small_pulses(df)
            histo_areas = functions.make_histo(areas, bins=200, _range=(0,200))

            ## Obtain a rough estimation of the peak positions (pedestal and SPE):
            print("Initial estimation of peak position")
            histo_areas_smooth = gaussian_filter1d(histo_areas[0], sigma=2) # Smooth the spectrum and ease the peak finding
            peaks = data_calibration.peak_position(histo_areas_smooth, prominence=np.max(histo_areas_smooth)*0.005)
            print("First peaks at " + str(peaks[0]) + " ADC*ns, with heights " +str(peaks[2]) +" counts")

            ## Fitting the histogram in order to obtain SPE peak and calibrate the spectrum
            print("Low histogram fitting")
            centers = (histo_areas[1][:-1] + histo_areas[1][1:]) / 2

            calib_popt_tot, calib_perr_tot = data_calibration.fit_hist(centers, histo_areas[0], peaks[0], peaks[2], cfg)
            print("Best fit params: " + str(calib_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(calib_perr_tot, calib_popt_tot)]))

            data_plot.plot_calib_spectrum(centers, histo_areas[0], calib_popt_tot, calib_perr_tot, functions.make_output_name(name), ylim=(1e1, 2e6))
            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt"), "a") as f:
                f.write("SPE \t" + str(calib_popt_tot[4]) + "\t +- \t" + str(calib_perr_tot[4]) + "\n")
                
                
        calib_SPE = data_io.read_from_file("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt"), obs="SPE", N=1)
        bins= cfg["alpha"]["bins"]
        _range=(cfg["alpha"]["range_low"], cfg["alpha"]["range_high"])
                
                
        if_analyse_alpha_bump_pre_PID = cfg["analysis"]["if_alpha_pre_PID"]
        
        if if_analyse_alpha_bump_pre_PID:
            ## Fitting the alpha bump to obtain light-yield
            histo_alpha = functions.make_histo(df["Integral"]/calib_SPE, bins=bins, _range=_range)
            centers = (histo_alpha[1][:-1] + histo_alpha[1][1:]) / 2
            #histo_alpha_smooth = gaussian_filter1d(histo_alpha[0], sigma=2) # Smooth the spectrum and ease the peak finding
            alpha_peak = data_calibration.peak_position(histo_alpha[0], height=(0,1e3), prominence=20)
            alpha_peak_PE = alpha_peak[0][0]*(_range[1]-_range[0])/bins+_range[0]

            print("\n ################################ \n Fitting histogram to find light-yield")
            print("Guess peak at " + str(alpha_peak_PE) + " PE, with heights " +str(alpha_peak[2][0]) +" counts")
            interval = (_range[1]-_range[0])/bins
            alpha_popt_tot, alpha_perr_tot = data_calibration.fit_hist_alpha_bump(centers[int(cfg["alpha"]["fit_from"]/interval):], histo_alpha[0][int(cfg["alpha"]["fit_from"]/interval):], alpha_peak_PE, alpha_peak[2][0], cfg)
            print("Best fit params: " + str(alpha_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(alpha_perr_tot, alpha_popt_tot)]))

            data_plot.plot_alpha_spectrum(centers, histo_alpha[0], alpha_popt_tot, alpha_perr_tot, functions.make_output_name(name, prefix="light_yield_pre_PID", ext=".png"), ylim=(1, 1e4), interval=interval, filename_hist=functions.make_output_name(name, prefix="light_yield_pre_PID", ext=".txt"), save_to_file=cfg["alpha"]["save_to_file"])
            
            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt"), "a") as f:
                f.write("light_yield_pre_PID \t" + str(alpha_popt_tot[1]) + "\t +- \t" + str(alpha_perr_tot[1])  + "\n")
                
                
        if_analyse_alpha_bump_post_PID = cfg["analysis"]["if_alpha_post_PID"]
        
        if if_analyse_alpha_bump_post_PID:
            print("\n ################################ \n Performing Particle Identification analysis")
            data_plot.plot_hist2d_Integral_Fprompt(df["Integral"]/calib_SPE, df["Prompt"], _range, functions.make_output_name(name, prefix="hist2d_Integral_Fprompt", ext=".png"))
            df_PE_cut = df[(df["Integral"]/calib_SPE > cfg["PID_analysis"]["PE_cut"])]
            
            bins = cfg["PID_analysis"]["prompt_bins"]
            ## Fitting the fraction of prompt light spectrum to obtain the PID cut
            histo_prompt = functions.make_histo(df_PE_cut["Prompt"], bins=bins, _range=(0, 1))
            centers = (histo_prompt[1][:-1] + histo_prompt[1][1:]) / 2
            guess_peaks = data_calibration.peak_position(histo_prompt[0], height=0, prominence=20)
            guess_peaks_pos = [x*1./bins for x in guess_peaks[0]]
            print("Guess peak at " + str(guess_peaks_pos) + " Fprompt, with heights " +str(guess_peaks[2]) +" counts")
            
            #data_plot.plot_Fprompt(df_PE_cut, functions.make_output_name(name, prefix="hist_Fprompt", ext=".png"))
            
            popt_ER, perr_ER, popt_alpha, perr_alpha = data_calibration.fit_fprompt(centers, histo_prompt[0], guess_peaks_pos, guess_peaks[2], cfg, interval=1./bins)
            print("Best fit params: ER:" + str(popt_ER) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_ER, popt_ER)]) + "\n")
            print("Best fit params: alpha:" + str(popt_alpha) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_alpha, popt_alpha)]) + "\n")
            
            fprompt_fit_min = int(cfg["PID_analysis"]["fprompt_fit_min"]*bins)
            fprompt_fit_max = int(cfg["PID_analysis"]["fprompt_fit_max"]*bins)
            gaus_sum = np.array(fit_models.gaus_list(centers[fprompt_fit_min:fprompt_fit_max], popt_ER[0], popt_ER[1], popt_ER[2])) + np.array(fit_models.gaus_list(centers[fprompt_fit_min:fprompt_fit_max], popt_alpha[0], popt_alpha[1], popt_alpha[2]))
            
            PID_cut = (np.argmin(gaus_sum)+fprompt_fit_min)/bins
            print("PID cut at Fprompt " + str(PID_cut))
            alpha_evts = df_PE_cut[df_PE_cut["Prompt"] >= PID_cut]
            ER_evts = df_PE_cut[df_PE_cut["Prompt"] < PID_cut]
            
            print("Alpha events / tot events before energy cut: " + str(len(alpha_evts)/len(df)*100.) + "%")
            print("Alpha events / tot events after energy cut: " + str(len(alpha_evts)/len(df_PE_cut)*100.) + "%")
            
            data_plot.plot_Fprompt_fitted(centers, df_PE_cut, df, bins, gaus_sum, popt_ER, popt_alpha, functions.make_output_name(name, prefix="hist_Fprompt_fitted", ext=".png"), fprompt_fit_min=fprompt_fit_min, fprompt_fit_max=fprompt_fit_max, cut=PID_cut, ylims=(1, 3e5))
            
            
            ## Fitting the alpha bump to obtain light-yield
            #histo_alpha_smooth = gaussian_filter1d(histo_alpha[0], sigma=2) # Smooth the spectrum and ease the peak finding
            bins= cfg["alpha"]["bins"]
            histo_alpha = functions.make_histo(alpha_evts["Integral"]/calib_SPE, bins=bins, _range=_range)
            centers = (histo_alpha[1][:-1] + histo_alpha[1][1:]) / 2
            alpha_peak = data_calibration.peak_position(histo_alpha[0], height=(0,1e3), prominence=20)
            alpha_peak_PE = alpha_peak[0][0]*(_range[1]-_range[0])/bins+_range[0]

            print("\n ################################ \n Fitting histogram to find light-yield")
            print("Guess peak at " + str(alpha_peak_PE) + " PE, with heights " +str(alpha_peak[2][0]) +" counts")
            interval = (_range[1]-_range[0])/bins
            alpha_popt_tot, alpha_perr_tot = data_calibration.fit_hist_alpha_bump(centers[int(cfg["alpha"]["fit_from"]/interval):], histo_alpha[0][int(cfg["alpha"]["fit_from"]/interval):], alpha_peak_PE, alpha_peak[2][0], cfg, model=fit_models.gaus)
            print("Best fit params alpha bump:" + str(alpha_popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(alpha_perr_tot, alpha_popt_tot)]) + "\n")
            
            #data_plot.plot_Integral(alpha_evts, functions.make_output_name(name, prefix="hist_alpha_evts_Integral", ext=".png"), norm=calib_SPE, bins=bins, _range=_range)
            
            data_plot.plot_alpha_spectrum_after_PID(centers, alpha_evts["Integral"]/calib_SPE, df["Integral"]/calib_SPE, bins, _range, alpha_popt_tot, alpha_perr_tot, functions.make_output_name(name, prefix="light_yield_post_PID", ext=".png"), ylim=(1, 8e5), interval=interval)
            
            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt"), "a") as f:
                f.write("light_yield_post_PID \t" + str(alpha_popt_tot[1]) + "\t +- \t" + str(alpha_perr_tot[1])  + "\n")
                
            if cfg["PID_analysis"]["save_alpha_evts_to_file"]:
                with open("observables/selected_alpha_events/"+functions.make_output_name(name, prefix="alpha_events", ext=".txt"), "x") as f:
                    for event in alpha_evts["Evtnb"]:
                        f.write(str(event)+"\n")
                        
                        
        if_triplet_lifetime = cfg["analysis"]["if_triplet_lifetime"]
        
        if if_triplet_lifetime:
            print("\n ################################ \n Fitting stacked waveforms to find triplet lifetime")
            evtnb = data_io.open_file("observables/selected_alpha_events/"+functions.make_output_name(name, prefix="alpha_events", ext=".txt"))
            df_PE_cut = df[(df["Integral"]/calib_SPE > cfg["PID_analysis"]["PE_cut"])]
            ER_evts = df_PE_cut.drop(evtnb)
            alpha_evts = df_PE_cut[df_PE_cut["Evtnb"].isin(evtnb)]
            
            alpha_wvf = functions.stack_waveforms(alpha_evts, n_samples=cfg["triplet_lifetime"]["n_samples"])
            ER_wvf = functions.stack_waveforms(ER_evts, n_samples=cfg["triplet_lifetime"]["n_samples"])
            tot_wvf = functions.stack_waveforms(df_PE_cut, n_samples=cfg["triplet_lifetime"]["n_samples"])
            
            popt_alpha, perr_alpha = data_calibration.fit_stacked_waveforms(alpha_wvf[0], cfg, n_samples=cfg["triplet_lifetime"]["n_samples"], sigma=alpha_wvf[1])
            popt_ER, perr_ER = data_calibration.fit_stacked_waveforms(ER_wvf[0], cfg, n_samples=cfg["triplet_lifetime"]["n_samples"], sigma=ER_wvf[1])
            popt_tot, perr_tot = data_calibration.fit_stacked_waveforms(tot_wvf[0], cfg, n_samples=cfg["triplet_lifetime"]["n_samples"], sigma=tot_wvf[1])
            
            print("Best fit params: ER+alpha: " + str(popt_tot) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_tot, popt_tot)]) + "\n")
            print("Best fit params: ER: " + str(popt_ER) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_ER, popt_ER)]) + "\n")
            print("Best fit params: alpha: " + str(popt_alpha) + "\n Relative errors %: " + str([err / parm * 100 for err, parm in zip(perr_alpha, popt_alpha)]) + "\n")
            
            data_plot.plot_fitted_stacked_wvfs(tot_wvf, ER_wvf, alpha_wvf, popt_tot, perr_tot, popt_ER, perr_ER, popt_alpha, perr_alpha, cfg, functions.make_output_name(name, prefix="staked_wvfs_fitted", ext=".png"), n_samples=cfg["triplet_lifetime"]["n_samples"], xlims=(-50,cfg["triplet_lifetime"]["n_samples"]*10))
            
            with open("observables/"+functions.make_output_name(name, prefix="obs", ext=".txt"), "a") as f:
                f.write("triplet_lifetime_ER+alpha \t" + str(popt_tot[1]) + "\t +- \t" + str(perr_tot[1])  + "\n")
                f.write("triplet_lifetime_ER \t" + str(popt_ER[1]) + "\t +- \t" + str(perr_ER[1])  + "\n")
                f.write("triplet_lifetime_alpha \t" + str(popt_alpha[1]) + "\t +- \t" + str(perr_alpha[1])  + "\n")
            

if __name__ == "__main__":
    main()