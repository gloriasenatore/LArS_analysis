import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import legendstyles
plt.style.use(legendstyles.LEGEND)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
import functions
from fit_models import calib_fit_model, gaus_list, gaus_fixed, exp_list, comb_list_gaus_with_exp, combined_gaus_LED_calib_delta, combined_gaus_LED_calib_free

def plot_calib_spectrum(centers, hist, popt_tot, perr_tot, filename, ylim=(1e1, 2e6)):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    #Plot Data-model
    axs[0].scatter(centers, hist, color='black', s=5, label='data')
    axs[0].plot(centers,calib_fit_model(centers,*popt_tot),'-', color='green', label='combined fit')
    axs[0].plot(centers,gaus_list(centers,popt_tot[0],popt_tot[1], popt_tot[2]),'-', color='grey', label=r'noise $\mu=$'+str('%.3f' % popt_tot[1])+"$\pm$"+str('%.3f' %perr_tot[1])+"  $\sigma=$"+str('%.3f' % popt_tot[2])+"$\pm$"+str('%.3f' % perr_tot[2]))
    #plt.plot(centers,gaus_list(centers,popt_erga_c[0]-expy[np.argwhere(centers_a==5.5)[0][0]],popt_erga_c[1], popt_erga_c[2]),'-', color='grey', label='noise')
    #plt.plot(centers,gaus(centers,*popt_erga_c),'-', color='grey')
    axs[0].plot(centers,gaus_list(centers,popt_tot[3],popt_tot[4],popt_tot[5]),'-', color='darkorange', label='SPE $\mu=$'+str('%.2f' % popt_tot[4])+"$\pm$"+str('%.2f' %perr_tot[4])+"  $\sigma=$"+str('%.2f' % popt_tot[5])+"$\pm$"+str('%.2f' % perr_tot[5]))
    axs[0].plot(centers,gaus_list(centers,popt_tot[6],popt_tot[7],popt_tot[8]),'-', color='blue', label='multiPE, DPE $\mu=$'+str('%.1f' % popt_tot[7])+"$\pm$"+str('%.1f' %perr_tot[7])+"  $\sigma=$"+str('%.1f' % popt_tot[8])+"$\pm$"+str('%.1f' % perr_tot[8]))
    #plt.plot(centers,gaus_fixed_3(centers,popt_tot[7]),'-', color='blue', label='TPE')
    #plt.plot(centers,gaus(centers,2400,174,50),'-', color='blue')
    axs[0].plot(centers,gaus_fixed(centers,popt_tot[9], popt_tot[4], 3, popt_tot[5]),'-', color='blue')
    axs[0].plot(centers,gaus_fixed(centers,popt_tot[10], popt_tot[4], 4, popt_tot[5]),'-', color='blue')
    axs[0].plot(centers,gaus_fixed(centers,popt_tot[11], popt_tot[4], 5, popt_tot[5]),'-', color='blue')
    axs[0].plot(centers,exp_list(centers,popt_tot[12], popt_tot[13]),'--', color='lightblue', label='exponential')

    axs[0].set_ylabel(f'Counts/{(centers[1]-centers[0]):.1f} ADC$\cdot$ns', fontsize=19)
    axs[0].legend(fontsize=15, frameon=False)

    axs[0].set_yscale('log')
    axs[0].set_xlim(0, 200)
    axs[0].set_ylim(ylim)

    axs[1].set_xlabel(f'Charge [ADC$\cdot$ns]', fontsize=15)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    
    res = functions.residuals(hist, calib_fit_model(centers,*popt_tot))

    
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)
    axs[1].plot(res, marker='.', linestyle='None', color='black')

    chi2_red = functions.reduced_chi_square(hist, calib_fit_model(centers,*popt_tot), centers, popt_tot)
    print("Fitting done, reduced chi2: " + str(chi2_red))
    if chi2_red < 4.:
        plt.savefig("plots/calibration/"+filename, bbox_inches='tight')
    else:
        print("WARNING: not very good fit, saved in bad")
        plt.savefig("plots/calibration/bad/"+filename, bbox_inches='tight')
        

def plot_alpha_spectrum(centers, hist, popt_tot, perr_tot, filename, filename_hist, ylim=(7e-1, 6e5), interval=5, save_to_file=False):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    axs[0].scatter(centers, hist, color='black', s=20)
    axs[0].set_ylim(ylim)
    
    mini=int((popt_tot[1]-200)/interval)
    maxi=int((popt_tot[1]+200)/interval)
    
    axs[0].plot(centers[mini:maxi], comb_list_gaus_with_exp(centers[mini:maxi],*popt_tot),'-', color='green')
    axs[0].plot(centers[mini:maxi],gaus_list(centers[mini:maxi],popt_tot[0],popt_tot[1],popt_tot[2]),'-', color='red', label=r'alpha peak' +"\n"+ '$\mu=$('+str('%.1f' % popt_tot[1])+"$\pm$"+str('%.1f' %perr_tot[1])+") PE" +"\n"+  "$\sigma=$("+str('%.1f' % popt_tot[2])+"$\pm$"+str('%.1f' % perr_tot[2])+") PE")
    axs[0].plot(centers[mini:maxi],exp_list(centers[mini:maxi],popt_tot[3], popt_tot[4]),'--', color='lightblue')
    axs[0].set_yscale('log')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].set_ylabel('Counts/'+str(interval)+'PE', fontsize=19)
    axs[0].legend(fontsize=15, frameon=False, loc='lower left')

    axs[1].set_xlabel('Waveform Area [PE]', fontsize=19)
    
    res = functions.residuals(hist[mini:maxi], comb_list_gaus_with_exp(centers[mini:maxi],*popt_tot))
    
    
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)
    axs[1].scatter(centers[mini:maxi], res, marker='.', linestyle='None', color='black')
    
    axs[1].set_xlim(min(centers), max(centers))

    chi2_red = functions.reduced_chi_square(hist[mini:maxi], comb_list_gaus_with_exp(centers[mini:maxi],*popt_tot), centers, popt_tot)
    print("Fitting done, reduced chi2: " + str(chi2_red))
    if chi2_red < 4.:
        plt.savefig("plots/light_yield_before_PID/"+filename, bbox_inches='tight')
    else:
        print("WARNING: not very good fit, saved in bad")
        plt.savefig("plots/light_yield_before_PID/bad/"+filename, bbox_inches='tight')
        
    if save_to_file:
        gaus_fit = gaus_list(centers[mini:maxi],popt_tot[0],popt_tot[1],popt_tot[2])
        with open("plots/light_yield_before_PID/"+filename_hist,"x") as f:
            j=0
            for i in range(len(hist)):
                f.write(str(centers[i])+"\t"+str(hist[i])+"\t"+str(gaus_fit[i])+"\t"+str(popt_tot[1])+
                        "\t"+str(perr_tot[1])+"\t"+str(popt_tot[2])+
                        "\t"+str(perr_tot[2]))
                if(i >= mini and i < maxi):
                    f.write("\t"+str(res[j])+"\t"+str(mini)+"\t"+str(maxi))
                    j=j+1

                f.write("\n")
        
        
def plot_hist2d_Integral_Fprompt(xvalues, yvalues, _range, filename):
    fig = plt.figure(figsize=(10,6))
    h = plt.hist2d(xvalues, yvalues, bins=[200, 100], range=[[_range[0], _range[1]],[0,1]], norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3])
    plt.ylabel("Fprompt", fontsize=19)
    plt.xlabel(r"Waveform integral [PE]", fontsize=19)
    
    plt.savefig("plots/light_yield_after_PID/"+filename, bbox_inches='tight')
    

def plot_Fprompt(df, filename):
    fig = plt.figure(figsize=(10,6))
    plt.hist(df["Prompt"], bins=200, range=(0, 1), histtype='step')
    plt.savefig("plots/light_yield_after_PID/"+filename, bbox_inches='tight')
    
    
def plot_Fprompt_fitted(centers, hist, hist_tot, bins, gaus_sum, popt_ER, popt_alpha, filename, cut, fprompt_fit_min, fprompt_fit_max, ylims=(1, 1e5)):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    
    axs[0].hist(hist_tot["Prompt"], bins=200, range=(0, 1), histtype='step', label='Before bkg cut', color='blue')
    hist = axs[0].hist(hist["Prompt"], bins=200, range=(0, 1), histtype='step', label='After bkg cut', color='orange')
    
    axs[0].plot(centers[fprompt_fit_min:fprompt_fit_max],gaus_sum,'-', color='green',label=r'Gaussian fits: $\mu_{ER}=$'+str('%.3f' %popt_ER[1])+" $\mu_{NR}=$"+str('%.3f' %popt_alpha[1]))
    axs[0].vlines(cut, 1, max(popt_ER[0], popt_alpha[0]), color='fuchsia', label='PID cut at '+str('%.2f' %cut))

    axs[0].set_yscale('log')
    axs[0].set_ylim(ylims)
    axs[0].legend(loc='upper right', frameon=False, fontsize=17)
    axs[0].set_ylabel("Counts", fontsize=19)
    
    axs[1].set_xlabel("Fprompt", fontsize=19)
    res_ER = functions.residuals(hist[0][int(popt_ER[1]*bins)-20:int(popt_ER[1]*bins)+20], gaus_sum[int(popt_ER[1]*bins)-fprompt_fit_min-20:int(popt_ER[1]*bins)-fprompt_fit_min+20])
    res_alpha = functions.residuals(hist[0][int(popt_alpha[1]*bins)-15:int(popt_alpha[1]*bins)+15], gaus_sum[int(popt_alpha[1]*bins)-fprompt_fit_min-15:int(popt_alpha[1]*bins)-fprompt_fit_min+15])
    
    
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)
    axs[1].scatter(centers[int(popt_ER[1]*bins)-20:int(popt_ER[1]*bins)+20], res_ER, marker='.', linestyle='None', color='black')
    axs[1].scatter(centers[int(popt_alpha[1]*bins)-15:int(popt_alpha[1]*bins)+15], res_alpha, marker='.', linestyle='None', color='black')
    
    axs[1].set_xlim(0, 1)
    
    plt.savefig("plots/light_yield_after_PID/"+filename, bbox_inches='tight')
    

def plot_Integral(df, filename, norm, bins, _range):
    fig = plt.figure(figsize=(10,6))
    plt.hist(df["Integral"]/norm, bins=bins, range=(_range), histtype='step')
    plt.yscale('log')
    plt.savefig("plots/light_yield_after_PID/"+filename, bbox_inches='tight')
    
    
    
def plot_alpha_spectrum_after_PID(centers, hist, hist_tot, bins, _range, popt_tot, perr_tot, filename, ylim=(7e-1, 6e5), interval=5):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    
    axs[0].hist(hist_tot, bins=bins, range=_range, histtype='step', label='before bkg & PID cuts', color='blue')
    hist_alpha = axs[0].hist(hist, bins=bins, range=_range, histtype='step', label='after bkg & PID cuts', color='orange')
    axs[0].set_ylim(ylim)
    
    mini=int((popt_tot[1]-150)/interval)
    maxi=int((popt_tot[1]+150)/interval)
    
    axs[0].plot(centers[mini:maxi], gaus_list(centers[mini:maxi],popt_tot[0],popt_tot[1],popt_tot[2]),'-', color='red', label=r'alpha peak' +"\n"+ '$\mu=$('+str('%.1f' % popt_tot[1])+"$\pm$"+str('%.1f' %perr_tot[1])+") PE" +"\n"+  "$\sigma=$("+str('%.1f' % popt_tot[2])+"$\pm$"+str('%.1f' % perr_tot[2])+") PE")
    axs[0].set_yscale('log')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].set_ylabel('Counts/'+str(interval)+'PE', fontsize=19)
    axs[0].legend(fontsize=15, frameon=False, loc='upper right')

    axs[1].set_xlabel('Waveform Area [PE]', fontsize=19)
    
    res = functions.residuals(hist_alpha[0][mini:maxi], gaus_list(centers[mini:maxi],*popt_tot))
    
    
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)
    axs[1].scatter(centers[mini:maxi], res, marker='.', linestyle='None', color='black')
    
    axs[1].set_xlim(min(centers), max(centers))

    chi2_red = functions.reduced_chi_square(hist_alpha[0][mini:maxi], gaus_list(centers[mini:maxi],*popt_tot), centers, popt_tot)
    print("Fitting done, reduced chi2: " + str(chi2_red))
    if chi2_red < 4.:
        plt.savefig("plots/light_yield_after_PID/"+filename, bbox_inches='tight')
    else:
        print("WARNING: not very good fit, saved in bad")
        plt.savefig("plots/light_yield_after_PID/bad/"+filename, bbox_inches='tight')
        
        
def plot_fitted_stacked_wvfs(traces_all, traces_ER, traces_alpha, popt_all, perr_all, popt_ER, perr_ER, popt_alpha, perr_alpha, cfg, filename, n_samples=800, xlims=(-50,7200)):
    
    lower_boundary = cfg["triplet_lifetime"]["lower_boundary_fit"]
    upper_boundary = cfg["triplet_lifetime"]["upper_boundary_fit"]
    
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    
    samples = np.linspace(0, int(n_samples*10), int(n_samples), endpoint=False)
    
    axs[0].hist(samples, bins = len(samples), weights = traces_all[0], histtype='step', label="ER+alpha stacked waveform", color='grey', alpha=0.7)
    axs[0].hist(samples, bins = len(samples), weights = traces_ER[0], histtype='step', label="ER stacked waveform", color='blue')
    axs[0].hist(samples, bins = len(samples), weights = traces_alpha[0], histtype='step', label="alpha stacked waveform", color='orange')

    axs[0].plot(samples[lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_all),'--', color='black', alpha=0.7, label=r'exponential fit ER+alpha $\tau_t$=('+str('%.0f' % popt_all[1])+"$\pm$"+str('%.0f' % perr_all[1])+") ns")
    axs[0].plot(samples[lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_ER),'--', color='red', label=r'exponential fit ER $\tau_t$=('+str('%.0f' % popt_ER[1])+"$\pm$"+str('%.0f' % perr_ER[1])+") ns")
    axs[0].plot(samples[lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_alpha),'--', color='green', label=r'exponential fit alpha $\tau_t$=('+str('%.0f' % popt_alpha[1])+"$\pm$"+str('%.0f' % perr_alpha[1])+") ns")

    axs[0].set_yscale("log")
    axs[0].set_ylabel("Normalized pulse amplitude", fontsize=19)
    axs[0].legend(fontsize=15, frameon=False)
    
    axs[1].set_xlabel("Time [ns]", fontsize=19)
    axs[1].set_xlim(xlims)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    
    res_all = functions.residuals(traces_all[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_all), sigma=traces_all[1][lower_boundary:upper_boundary])
    res_ER = functions.residuals(traces_ER[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_ER), sigma=traces_ER[1][lower_boundary:upper_boundary])
    res_alpha = functions.residuals(traces_alpha[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_alpha), sigma=traces_alpha[1][lower_boundary:upper_boundary])
   

    axs[1].fill_between(samples, y1= 0 - 10, y2= 0 + 10, color='green', alpha=.5)
    axs[1].fill_between(samples, y1= - 20, y2= -10, color='orange', alpha=.5)
    axs[1].fill_between(samples, y1= 10, y2= 20, color='orange', alpha=.5)
    axs[1].fill_between(samples, y1= - 30, y2= -20 , color='red', alpha=.5)
    axs[1].fill_between(samples, y1= 20, y2= 30, color='red', alpha=.5)
    axs[1].set_ylabel(r'Res [%]', fontsize=19)
    
    axs[1].scatter(samples[lower_boundary:upper_boundary], res_all, marker='.', linestyle='None', color='black', alpha=0.7)
    axs[1].scatter(samples[lower_boundary:upper_boundary], res_ER, marker='.', linestyle='None', color='red')
    axs[1].scatter(samples[lower_boundary:upper_boundary], res_alpha, marker='.', linestyle='None', color='green')
    
    chi2_red_all = functions.reduced_chi_square(traces_all[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_all), samples, popt_all, sigma=traces_all[1][lower_boundary:upper_boundary])
    chi2_red_ER = functions.reduced_chi_square(traces_ER[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_ER), samples, popt_ER, sigma=traces_ER[1][lower_boundary:upper_boundary])
    chi2_red_alpha = functions.reduced_chi_square(traces_alpha[0][lower_boundary:upper_boundary], exp_list(samples[lower_boundary:upper_boundary], *popt_alpha), samples, popt_alpha, sigma=traces_alpha[1][lower_boundary:upper_boundary])
    print("ER+alpha fit done, reduced chi2: " + str(chi2_red_all))
    print("ER fit done, reduced chi2: " + str(chi2_red_ER))
    print("alpha fit done, reduced chi2: " + str(chi2_red_alpha))
    
    plt.savefig("plots/triplet_lifetime/"+filename, bbox_inches='tight')
    
    
    
def plot_LED_calibration(centers, data_all, bins, _range, popt_tot, perr_tot, filename, peak_valley_ratios, ylim=(7e-1, 6e4), fixed_delta=True):
    if fixed_delta: model = combined_gaus_LED_calib_delta
    else: model = combined_gaus_LED_calib_free
    
    fig = plt.figure(figsize=(15,9))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[4,1])
    axs = gs.subplots(sharex=True, sharey=False)
    
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)
    
    interval = (_range[1]-_range[0])/bins
    
    for data, popt, perr, peak_valley_ratio in zip(data_all, popt_tot, perr_tot, peak_valley_ratios):
        n, bins, patches = axs[0].hist(data, bins=bins, range=_range, histtype='step')
        color = patches[0].get_edgecolor()
        
        if fixed_delta:
            axs[0].plot(centers, model(centers,*popt), '-', color=color, label=r'noise $\mu=$'+str('%.2f' % popt[1])+"$\pm$"+str('%.2f' % perr[1])+"  $\sigma=$"+str('%.2f' % popt[2])+"$\pm$"+str('%.2f' % perr[2])+'\n SPE $\mu=$'+str('%.2f' % (popt[1]+popt[4]))+"$\pm$"+str('%.2f' %np.sqrt(perr[1]**2.+perr[4]**2.))+"  $\sigma=$"+str('%.2f' % popt[5])+"$\pm$"+str('%.2f' % perr[5])+'\n DPE $\mu=$'+str('%.2f' % (popt[1]+2.*popt[4]))+"$\pm$"+str('%.2f' %np.sqrt(perr[1]**2.+(2.*perr[4])**2.))+"  $\sigma=$"+str('%.2f' % (np.sqrt(2.)*popt[5]))+"$\pm$"+str('%.2f' % (np.sqrt(2.)*perr[5])) + "\n peak/valley=" + str('%.2f' % peak_valley_ratio))
        else:
            axs[0].plot(centers, model(centers,*popt), '-', color=color, label=r'noise $\mu=$'+str('%.2f' % popt[1])+"$\pm$"+str('%.2f' % perr[1])+"  $\sigma=$"+str('%.2f' % popt[2])+"$\pm$"+str('%.2f' % perr[2])+'\n SPE $\mu=$'+str('%.2f' % popt[4])+"$\pm$"+str('%.2f' % perr[4])+"  $\sigma=$"+str('%.2f' % popt[5])+"$\pm$"+str('%.2f' % perr[5])+'\n DPE $\mu=$'+str('%.2f' % popt[7])+"$\pm$"+str('%.2f' % perr[7])+"  $\sigma=$"+str('%.2f' % popt[8])+"$\pm$"+str('%.2f' % perr[8]) + "\n peak/valley=" + str('%.2f' % peak_valley_ratio))
        
        res = functions.residuals(n, model(centers,*popt))
        axs[1].scatter(centers, res, marker='.', linestyle='None', color=color)
        
        chi2_red = functions.reduced_chi_square(n, model(centers,*popt), centers, popt)
        print("Fitting done, reduced chi2: " + str(chi2_red))

    
    axs[0].set_ylim(ylim)
    axs[0].set_yscale('log')
    
    axs[0].set_ylabel('Counts/'+str('%.1f' % interval)+f'ADC$\cdot$ns', fontsize=19)
    axs[0].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    axs[1].set_xlabel(f'Charge [ADC$\cdot$ns]', fontsize=19)
    axs[1].set_xlim(min(centers), max(centers))
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig("plots/LED_calibration/"+filename, bbox_inches='tight')