Usage:

  - To calibrate the several spectra using the LED calibration files: python LED_calibration.py 
  - To fit the alpha-decay-induced-LAr scintillation spectrum: python run_analysis_alpha.py *waveforms root file or list of files*
  - To fit the blue LED light spectrum: python run_analysis_visible.py *waveforms root file or list of files*
  
config.json file explanation:

  - "import_tree"
    - "store_traces": true/false, whether to import the waveform (already post-processed, meaning corrected for the baseline), or just the quantities associated, like integral, fPrompt, ... It can slow down the root file import, but it's needed in case you want to calibrate and/or fitting the stacked waveforms to estimate the triplet lifetime.
  - "analysis"
    - "concatenate_files": true/false, whether to concatenate more than one root file (hence treat them as a single data-taking) or keep them separate. For alpha source data-taking I usually concatenate more files to have around 2-3 million waveforms, so to have enough statistic to do a nice fit.
    - "if_calibrate": true/false, whether to calibrate using small pulses (i.e. dark counts) or not. We prefer to calibrate using LED calibration runs, but we somethimes check if the small pulses calibration gives more or less compatible results and/or if there is a gain drift. Note that small pulses calibration might be a bit biased towards more positive values, because of the analysis threshold that the user set in WARP to select the small pulses.
    - "file_LED_calib": "R8", for example, data-taking name (just R+number) of the file containing SPE_100_115 value found with a LED calib run.
    - "if_alpha_pre_PID": true/false, whether to perform the fitting of the histogram of the waveforms integral in order to estimate VUV-light yield. This fitting is done "pre PID", meaning before performing the particle identification (also called pulse shape discrimination) with the fPrompt, by fitting the alpha bump with a gaussian + exponential model (exponential as a proxy for the background).
    - "if_alpha_post_PID": true/false, whether to perform the PID, which consists in selecting only waveform over a certain integral value (see "PE_cut" later) and then in finding the PID cut by fitting the fPrompt histogram (fraction of prompt light) with a double gaussian function and finding the fPrompt value that better separate the two populations. The waveforms above that fPrompt cut is formed by alpha events, and the population below cut by electron-recoil events.
    - "if_triplet_lifetime": true/false, whether to stack the waveforms and fit them with an exponential to find triplet lifetime of LAr. We fit alpha only, ER only, and alpha+ER events (always above the "PE_cut"). We noticed that probably the fit with ER events is more reliable because higher statistic.
  - "dark_counts_calib"
    - "lower_boundary": -5, number of samples (1 samples = 10ns) before the position of the maximum height of each peak found by WARP. We start the integration of the small pulse from there until "upper_boundary". We use total number of integrated samples equal to 15, which is the same used for LED calibration.
    - "upper_boundary": +10, number of samples after the position of the maximum height of each peak. As said, we use "upper_boundary" - "lower_boundary" = 15 samples to keep it consistent with LED calib.
    
