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
      ax1.plot(k, k*PS_estimate, label=r'$A_1k$', color='teal')
      ax1.fill_between(x=k, y1=k*PS_estimate-k*PS_error, y2=k*PS_estimate+k*PS_error, facecolor='paleturquoise', edgecolor='paleturquoise')
   if better == True:
      ax1.plot(k, k*PS_estimate*P_theory(k), label=r'$A_2kP_{theory}(k)$', color='teal')
      ax1.fill_between(x=k, y1=k*PS_estimate*P_theory(k)-k*PS_error*P_theory(k), y2=k*PS_estimate*P_theory(k)+k*PS_error*P_theory(k), facecolor='paleturquoise', edgecolor='paleturquoise')
   ax1.errorbar(k, k * xs_mean / (transfer(k)*transfer_Nils(k)), k * xs_sigma / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{data}(k)$', color='purple')
   
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10 kP_{Theory}(k)$', color='navy')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=13)
   ax1.set_ylim(-lim*2.5, lim*2.5)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.03,k[-1]+0.1)
   ax1.set_xscale('log')
   ax1.grid()
   plt.legend(ncol=3, fontsize=10)

   ax2 = fig.add_subplot(212)
   ax2.errorbar(k, xs_mean / xs_sigma, xs_sigma / xs_sigma, fmt='o',color='purple')
   ax2.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=13)
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=13)
   ax2.set_ylim(-5, 5)
   ax2.set_xlim(0.03,k[-1]+0.1)
   ax2.set_xscale('log')
   ax2.grid()
   plt.tight_layout()
   
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
   for i in range(4,no_of_k-3): #we exclude 4 first points and 3 last points, previously excluded 2 first
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
   for i in range(4,no_of_k-3): #we exclude 4 first points and 3 last points, previously excluded 2 first
      w = 1./ xs_sigma[i]**2.
      w_sum += w
      PS_estimate += w*xs_mean[i]
   PS_estimate = PS_estimate/w_sum
   PS_error = w_sum**(-0.5)
   return PS_estimate, PS_error

def call_all(mapname, split):
   xs_files = 'spectra/xs_' + mapname + '_1st_' + split + '_feed%01i_and_' + mapname +'_2nd_' + split +'_feed%01i.h5'
   k, xs_mean, xs_sigma = xs_feed_feed_grid(xs_files, 'xs_grid_' + mapname + '_' + split + '.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split')
   #xs_with_model('xs_mean_' + mapname + '_' + split + '.png', k, xs_mean, xs_sigma)
   return k, xs_mean, xs_sigma

k2c, mean2c, sigma2c = call_all('co2_map_complete_sunel_ces', 'dayn')
#k6c, mean6c, sigma6c = call_all('co6_map_complete_sunel_ces', 'dayn')
#k7c, mean7c, sigma7c = call_all('co7_map_complete_sunel_ces', 'dayn')
no_k = len(k2c)
print ('k2c', k2c)
print ('mean2c', mean2c)
print ('sigma2c', sigma2c)
np.save('k_co2_ces.npy', k2c)
np.save('xs_co2_ces.npy', mean2c)
np.save('sigma_co2_ces.npy', sigma2c)
'''
xs_sigma_arr = np.array([sigma2c,sigma6c,sigma7c])
xs_mean_arr = np.array([mean2c, mean6c, mean7c])
mean_combined = 0
w_sum = 0
no_maps = len(xs_sigma_arr)
no_k = len(k2c)
for i in range(no_maps): 
   w = 1./ xs_sigma_arr[i]**2.
   w_sum += w
   mean_combined += w*xs_mean_arr[i]
mean_combined = mean_combined/w_sum
sigma_combined = w_sum**(-0.5)
#xs_with_model('xs_mean_co2ces_co6ces_co7ces_new.png', k2c, mean_combined, sigma_combined)

PS_estimate_1, PS_error_1 = calculate_PS_amplitude(k2c, mean_combined, sigma_combined)
PS_estimate_2, PS_error_2 = calculate_PS_amplitude_better(k2c, mean_combined, sigma_combined)
print ('PS1', PS_estimate_1, PS_error_1)#PS1 28560.3860758366 12383.365385164725
print ('PS2', PS_estimate_2, PS_error_2) #PS2 12.881779694882994 4.916077608835785

'''
PS_estimate_1, PS_error_1 = calculate_PS_amplitude(k2c, mean2c, sigma2c)
PS_estimate_2, PS_error_2 = calculate_PS_amplitude_better(k2c, mean2c, sigma2c)
print ('PS1', PS_estimate_1, PS_error_1) 
print ('PS2', PS_estimate_2, PS_error_2) 

#PS1 8496.58335475071 22182.791182467696
#PS2 -0.5425939574210513 8.608316607884488


PS_estimate_arr = np.zeros(no_k) + PS_estimate_1
PS_error_arr = np.zeros(no_k) + PS_error_1
PS_estimate_arr2 = np.zeros(no_k) + PS_estimate_2
PS_error_arr2 = np.zeros(no_k) + PS_error_2
xs_with_model('xs_mean_co2ces_v2.pdf', k2c, mean2c, sigma2c, PS_estimate_arr2, PS_error_arr2, better=True)
xs_with_model('xs_mean_co2ces_v1.pdf', k2c, mean2c, sigma2c, PS_estimate_arr, PS_error_arr, better=False)

'''
k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_1st_dayn_feed%01i_and_co7_map_complete_night_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co7_night.png', ' of 1st dayn split', ' of 2nd dayn split')
PS_estimate_1, PS_error_1 = calculate_PS_amplitude(k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn)
PS_estimate_2, PS_error_2 = calculate_PS_amplitude_better(k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn)
print (PS_estimate_1, PS_error_1) #14622.089136473009 12593.846308010017
print ('PS2', PS_estimate_2, PS_error_2) #PS2 4.061535273875437 5.053176958684818

PS_estimate_arr = np.zeros(14) + PS_estimate_1
PS_error_arr = np.zeros(14) + PS_error_1
PS_estimate_arr2 = np.zeros(14) + PS_estimate_2
PS_error_arr2 = np.zeros(14) + PS_error_2
xs_with_model('xs_mean_dayn_co7_night_wbetterestimate.pdf', k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn, PS_estimate_arr2, PS_error_arr2, better=True)
xs_with_model('xs_mean_dayn_co7_night_westimate.pdf', k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn, PS_estimate_arr, PS_error_arr, better=False)
'''

