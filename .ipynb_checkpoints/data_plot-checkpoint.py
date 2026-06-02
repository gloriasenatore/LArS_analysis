import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import legendstyles
plt.style.use(legendstyles.LEGEND)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
import functions
from fit_models import calib_fit_model, gaus_list, gaus_fixed, exp_list, comb_list_gaus_with_exp

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

    axs[1].plot(res, marker='.', linestyle='None', color='black')
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)

    chi2_red = functions.reduced_chi_square(hist, calib_fit_model(centers,*popt_tot), centers, popt_tot)
    print("Fitting done, reduced chi2: " + str(chi2_red))
    if chi2_red < 4.:
        plt.savefig("plots/calibration/"+filename, bbox_inches='tight')
    else:
        print("WARNING: not very good fit, saved in bad")
        plt.savefig("plots/calibration/bad/"+filename, bbox_inches='tight')
        

def plot_alpha_spectrum(centers, hist, popt_tot, perr_tot, filename, ylim=(7e-1, 6e5), interval=5):
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
    axs[0].set_ylabel('Counts/5 PE', fontsize=19)
    axs[0].legend(fontsize=15, frameon=False, loc='lower left')

    axs[1].set_xlabel('Waveform Area [PE]', fontsize=19)
    
    res = functions.residuals(hist[mini:maxi], comb_list_gaus_with_exp(centers[mini:maxi],*popt_tot))
    
    axs[1].scatter(centers[mini:maxi], res, marker='.', linestyle='None', color='black')
    axs[1].fill_between(centers, y1= 0 - 1, y2= 0 + 1, color='green', alpha=.5)
    axs[1].fill_between(centers, y1= - 2, y2= -1, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= 1, y2= 2, color='orange', alpha=.5)
    axs[1].fill_between(centers, y1= - 3, y2= -2 , color='red', alpha=.5)
    axs[1].fill_between(centers, y1= 2, y2= 3, color='red', alpha=.5)
    axs[1].set_ylabel(r'Pulls [$\sigma$]', fontsize=19)

    chi2_red = functions.reduced_chi_square(hist[mini:maxi], comb_list_gaus_with_exp(centers[mini:maxi],*popt_tot), centers, popt_tot)
    print("Fitting done, reduced chi2: " + str(chi2_red))
    if chi2_red < 4.:
        plt.savefig("plots/light_yield_before_PID/"+filename, bbox_inches='tight')
    else:
        print("WARNING: not very good fit, saved in bad")
        plt.savefig("plots/light_yield_before_PID/bad/"+filename, bbox_inches='tight')