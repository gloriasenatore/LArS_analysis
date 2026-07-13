Usage:

  - To calibrate the several spectra using the LED calibration files: python LED_calibration.py 
  - To fit the alpha-decay-induced-LAr scintillation spectrum: python run_analysis_alpha.py *waveforms root file or list of files*
  - To fit the blue LED light spectrum: python run_analysis_visible.py *waveforms root file or list of files*
  
config.json file explanation:

  - "import_tree"
    - "store_traces": true/false, wheter to import the waveform (already post-processed, meaning corrected for the baseline), or just the quantities associated, like integral, fPrompt, ... It can slow down the root file import, but it's needed in case you want to calibrate and/or fitting the stacked waveforms to estimate the triplet lifetime.
