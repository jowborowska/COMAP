import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys
import scipy.interpolate

import tools
import map_cosmo
import my_class
import PS_function
from scipy.optimize import curve_fit

#theory spectrum
k_th = np.load('k.npy')
ps_th = np.load('ps.npy')
ps_th_nobeam = np.load('psn.npy') #instrumental beam, less sensitive to small scales line broadening, error bars go up at high k, something with the intrinsic resolution of the telescope (?)

#values from COPPS
ps_copps = 8.746e3 * ps_th / ps_th_nobeam #shot noise level
ps_copps_nobeam = 8.7e3


def xs_feed_feed_grid(path_to_xs, figure_name,split1, split2, xs_summ, xs_divv):
   n_sim = 100
   n_k = 14
   n_feed = 19
   xs = np.zeros((n_feed, n_feed, n_k))
   rms_xs_std = np.zeros_like(xs)
   chi2 = np.zeros((n_feed, n_feed))
   noise = np.zeros_like(chi2)
   n_sum = 0
   k = np.zeros(n_k)
   xs_sum = xs_summ
   rms_xs_sum = np.zeros((n_k, n_sim))
   xs_div = xs_divv
   for i in range(n_feed):
       for j in range(n_feed):
           #if i != 7 and j != 7:
              try:
                  filepath = path_to_xs %(i+1, j+1)
                  with h5py.File(filepath, mode="r") as my_file:
                      xs[i, j] = np.array(my_file['xs'][:])
                      rms_xs_std[i, j] = np.array(my_file['rms_xs_std'][:])
                      k[:] = np.array(my_file['k'][:])
              except:
                  xs[i, j] = np.nan
                  rms_xs_std[i, j] = np.nan
            
              w = np.sum(1 / rms_xs_std[i,j])
              noise[i,j] = 1 / np.sqrt(w)
              chi3 = np.sum((xs[i,j] / rms_xs_std[i,j]) ** 3) #we need chi3 to take the sign into account - positive or negative correlation

              chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k)) #chi2 gives magnitude - how far it is from the white noise
              #print ("chi2: ", chi2[i, j]) #this chi2 is very very big, so it never comes through the if-test - check how to generate maps with smaller chi2 maybe :)
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal   
              #if i != j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
              if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:
                  xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                  #print ("if test worked")
                  xs_div += 1 / rms_xs_std[i,j] ** 2
                  n_sum += 1


   plt.figure()
   vmax = 15
   plt.imshow(chi2, interpolation='none', vmin=-vmax, vmax=vmax, extent=(0.5, n_feed + 0.5, n_feed + 0.5, 0.5))
   new_tick_locations = np.array(range(n_feed)) + 1
   plt.xticks(new_tick_locations)
   plt.yticks(new_tick_locations)
   plt.xlabel('Feed'+split1)
   plt.ylabel('Feed'+split2)
   cbar = plt.colorbar()
   cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()
   #print ("xs_div:", xs_div)
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div), xs_sum, xs_div

def xs_with_model(figure_name, k, xs_mean, xs_sigma):
  
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
   lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8

   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.errorbar(k, k * xs_mean / transfer(k), k * xs_sigma / transfer(k), fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   ax1.plot(k_th, k_th * ps_th_nobeam * 10, 'r--', label=r'$10 \times kP_{Theory}(k)$')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]')
   ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
   ax1.set_xscale('log')
   ax1.grid()
   plt.legend()

   ax2 = fig.add_subplot(212)
   ax2.errorbar(k, xs_mean / xs_sigma, xs_sigma / xs_sigma, fmt='o', label=r'$\tilde{C}_{data}(k)$')
   ax2.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
   ax2.set_ylim(-12, 12)
   ax2.set_xscale('log')
   ax2.grid()
   plt.tight_layout()
   plt.legend()
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()


k_12, xs_mean_12, xs_sigma_12, xs_sum, xs_div = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_earlysid_1st_sidr_feed%01i_and_co7_map_complete_night_earlysid_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_12.png', ' of split 1', ' of split 2', np.zeros(14), np.zeros(14))
print ('12: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_12.png', k_12, xs_mean_12, xs_sigma_12)

k_13, xs_mean_13, xs_sigma_13, xs_sum, xs_div = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_earlysid_1st_sidr_feed%01i_and_co7_map_complete_night_latesid_1st_sidr_feed%01i.h5', 'xs_grid_sidr_13.png', ' of split 1', ' of split 3',  np.zeros(14), np.zeros(14))
print ('13: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_13.png', k_13, xs_mean_13, xs_sigma_13)

k_14, xs_mean_14, xs_sigma_14, xs_sum, xs_div = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_earlysid_1st_sidr_feed%01i_and_co7_map_complete_night_latesid_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_14.png', ' of split 1', ' of split 4',  np.zeros(14), np.zeros(14))
print ('14: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_14.png', k_14, xs_mean_14, xs_sigma_14)

k_23, xs_mean_23, xs_sigma_23, xs_sum, xs_div = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_earlysid_2nd_sidr_feed%01i_and_co7_map_complete_night_latesid_1st_sidr_feed%01i.h5', 'xs_grid_sidr_23.png', ' of split 2', ' of split 3',  np.zeros(14), np.zeros(14))
print ('23: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_23.png', k_23, xs_mean_23, xs_sigma_23)

k_24, xs_mean_24, xs_sigma_24, xs_sum, xs_div  = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_earlysid_2nd_sidr_feed%01i_and_co7_map_complete_night_latesid_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_24.png', ' of split 2', ' of split 4',  np.zeros(14), np.zeros(14))
print ('24: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_24.png', k_24, xs_mean_24, xs_sigma_24)

k_34, xs_mean_34, xs_sigma_34, xs_sum, xs_div  = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_latesid_1st_sidr_feed%01i_and_co7_map_complete_night_latesid_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_34.png', ' of split 3', ' of split 4',  np.zeros(14), np.zeros(14))
print ('34: ', xs_sum, xs_div)
xs_with_model('xs_mean_sidr_34.png', k_34, xs_mean_34, xs_sigma_34)
#xs_with_model('xs_mean_sidr_4splits.png', k_34, xs_mean_34, xs_sigma_34)

'''
k_co7, xs_mean_co7, xs_sigma_co7 = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_1st_sidr_feed%01i_and_co7_map_complete_night_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_co7_night_no8.png')
xs_with_model('xs_mean_sidr_co7_night_no8.png', k_co7, xs_mean_co7, xs_sigma_co7)

'''

