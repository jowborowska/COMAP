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

def read_Nils_transfer(filename):
   infile = open(filename, 'r')
   k = np.zeros(14)
   T = np.zeros(14)
   i = 0
   infile.readline()
   for line in infile:
      values = line.split()
      k[i] = float(values[0])
      T[i] = float(values[1])
      i += 1
   infile.close()
   return k,T

k_Nils, T_Nils = read_Nils_transfer('TF.txt')

def xs_feed_feed_grid(path_to_xs, figure_name, split1, split2):
   n_sim = 100
   n_k = 14
   n_feed = 19
   xs = np.zeros((n_feed, n_feed, n_k))
   rms_xs_std = np.zeros_like(xs)
   chi2 = np.zeros((n_feed, n_feed))
   noise = np.zeros_like(chi2)
   n_sum = 0
   k = np.zeros(n_k)
   xs_sum = np.zeros(n_k)
   rms_xs_sum = np.zeros((n_k, n_sim))
   xs_div = np.zeros(n_k)
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
              print ("chi2: ", chi2[i, j]) #this chi2 is very very big, so it never comes through the if-test - check how to generate maps with smaller chi2 maybe :)
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal   
              #if i != j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
              if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:
                  xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                  print ("if test worked")
                  xs_div += 1 / rms_xs_std[i,j] ** 2
                  n_sum += 1


   plt.figure()
   vmax = 15
   plt.imshow(chi2, interpolation='none', vmin=-vmax, vmax=vmax, extent=(0.5, n_feed + 0.5, n_feed + 0.5, 0.5))
   new_tick_locations = np.array(range(n_feed)) + 1
   plt.xticks(new_tick_locations)
   plt.yticks(new_tick_locations)
   plt.xlabel('Feed' + split1)
   plt.ylabel('Feed' + split2)
   cbar = plt.colorbar()
   cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()
   print ("xs_div:", xs_div)
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div)

def xs_with_model(figure_name, k, xs_mean, xs_sigma, PS_estimate, PS_error, better):
  
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
   transfer_Nils = scipy.interpolate.interp1d(k_Nils, T_Nils) 
   P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)
   lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8

   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   if better == False:
      ax1.plot(k, k*PS_estimate, label=r'$kA_1$', color='teal')
      ax1.fill_between(x=k, y1=k*PS_estimate-k*PS_error, y2=k*PS_estimate+k*PS_error, facecolor='paleturquoise', edgecolor='paleturquoise')
   if better == True:
      ax1.plot(k, k*PS_estimate*P_theory(k), label=r'$kA_2\times P_{theory}$', color='teal')
      ax1.fill_between(x=k, y1=k*PS_estimate*P_theory(k)-k*PS_error*P_theory(k), y2=k*PS_estimate*P_theory(k)+k*PS_error*P_theory(k), facecolor='paleturquoise', edgecolor='paleturquoise')
   ax1.errorbar(k, k * xs_mean / (transfer(k)*transfer_Nils(k)), k * xs_sigma / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{data}(k)$', color='purple')
   
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   ax1.plot(k_th, k_th * ps_th_nobeam * 5, '--', label=r'$5 \times kP_{Theory}(k)$', color='navy')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]')
   ax1.set_ylim(-lim*2.5, lim*2.5)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.03,k[-1]+0.1)
   ax1.set_xscale('log')
   ax1.grid()
   plt.legend(bbox_to_anchor=(0, 0.61))

   ax2 = fig.add_subplot(212)
   ax2.errorbar(k, xs_mean / xs_sigma, xs_sigma / xs_sigma, fmt='o', label=r'$\tilde{C}_{data}(k)$')
   ax2.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
   ax2.set_ylim(-12, 12)
   ax2.set_xlim(0.03,k[-1]+0.1)
   ax2.set_xscale('log')
   ax2.grid()
   plt.tight_layout()
   plt.legend()
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()

def calculate_PS_amplitude(k, xs_mean, xs_sigma):
   PS_estimate = 0
   w_sum = 0
   no_of_k = len(k)
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) 
   transfer_Nils = scipy.interpolate.interp1d(k_Nils, T_Nils) 
   xs_mean = xs_mean/(transfer(k)*transfer_Nils(k))
   xs_sigma = xs_sigma/(transfer(k)*transfer_Nils(k))
   for i in range(2,no_of_k-3): #we exclude 2 first points and 3 last points
      print (k[i]*xs_mean[i], k[i]*xs_sigma[i])
      w = 1./ xs_sigma[i]**2.
      w_sum += w
      PS_estimate += w*xs_mean[i]
   PS_estimate = PS_estimate/w_sum
   PS_error = w_sum**(-0.5)
   return PS_estimate, PS_error

def calculate_PS_amplitude_better(k, xs_mean, xs_sigma):
   PS_estimate = 0
   w_sum = 0
   no_of_k = len(k)
   P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) 
   transfer_Nils = scipy.interpolate.interp1d(k_Nils, T_Nils) 
   xs_mean = xs_mean/(transfer(k)*transfer_Nils(k)*P_theory(k))
   xs_sigma = xs_sigma/(transfer(k)*transfer_Nils(k)*P_theory(k))
   for i in range(2,no_of_k-3): #we exclude 2 first points and 3 last points
      w = 1./ xs_sigma[i]**2.
      w_sum += w
      PS_estimate += w*xs_mean[i]
   PS_estimate = PS_estimate/w_sum
   PS_error = w_sum**(-0.5)
   return PS_estimate, PS_error


k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_1st_dayn_feed%01i_and_co7_map_complete_night_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co7_night.png', ' of 1st dayn split', ' of 2nd dayn split')
PS_estimate_1, PS_error_1 = calculate_PS_amplitude(k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn)
PS_estimate_2, PS_error_2 = calculate_PS_amplitude_better(k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn)
print (PS_estimate_1, PS_error_1)
print ('PS2', PS_estimate_2, PS_error_2)
PS_estimate_arr = np.zeros(14) + PS_estimate_1
PS_error_arr = np.zeros(14) + PS_error_1
PS_estimate_arr2 = np.zeros(14) + PS_estimate_2
PS_error_arr2 = np.zeros(14) + PS_error_2
xs_with_model('xs_mean_dayn_co7_night_wbetterestimate.png', k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn, PS_estimate_arr2, PS_error_arr2, better=True)
xs_with_model('xs_mean_dayn_co7_night_westimate.png', k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn, PS_estimate_arr, PS_error_arr, better=False)


