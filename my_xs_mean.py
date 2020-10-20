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
              if i < j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and j>i:#i != j:
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


def xs_split_split_grid(path_to_xs, figure_name, n_splits):
   n_sim = 100
   n_k = 14
   n_feed = 19
   xs = np.zeros((n_splits, n_splits, n_k))
   rms_xs_std = np.zeros_like(xs)
   chi2 = np.zeros((n_splits, n_splits))
   noise = np.zeros_like(chi2)
   n_sum = 0
   k = np.zeros(n_k)
   xs_sum = np.zeros(n_k)
   rms_xs_sum = np.zeros((n_k, n_sim))
   xs_div = np.zeros(n_k)
   vmax = 0.
   for i in range(n_splits):
       for j in range(n_splits):
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
           if chi2[i,j] > vmax and not np.isnan(chi2[i,j]):
              vmax = chi2[i,j]
           #print ("chi2: ", chi2[i, j]) #this chi2 is very very big, so it never comes through the if-test - check how to generate maps with smaller chi2 maybe :)
 
           #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal   
           if i != j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
               xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
               #print ("if test worked")
               xs_div += 1 / rms_xs_std[i,j] ** 2
               n_sum += 1


   plt.figure()
   
   #print (vmax)
   plt.imshow(chi2, interpolation='none', vmin=-vmax, vmax=vmax, extent=(0.5, n_splits + 0.5, n_splits + 0.5, 0.5))
   new_tick_locations = np.array(range(n_splits)) + 1
   plt.xticks(new_tick_locations)
   plt.yticks(new_tick_locations)
   plt.xlabel('Split')
   plt.ylabel('Split')
   cbar = plt.colorbar()
   cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()
   #print ("xs_div:", xs_div)
   plt.close()
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div)

def xs_mean_from_splits(figure_name, k, xs_mean, xs_sigma):
  
  
   #lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8

   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]')
   #ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
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
   plt.close()
   #plt.show()

def xs_mean_autoPS(filename):
  
   with h5py.File(filename, mode="r") as my_file:
      xs = np.array(my_file['xs'][:])
      rms_xs_std = np.array(my_file['rms_xs_std'][:])
      k = np.array(my_file['k'][:])
 
   xs_sum = xs/ rms_xs_std ** 2
   xs_div = 1 / rms_xs_std ** 2
   xs_mean_auto = xs_sum / xs_div
   xs_sigma_auto = 1. / np.sqrt(xs_div)
   return k, xs_mean_auto, xs_sigma_auto




'''
#The task for now:
maps/co2_map_complete_night_ces.h5
python my_script.py /mn/stornext/d16/cmbco/comap/protodir/maps/co2_map_complete_night_ces.h5 all dayn


maps/co2_map_complete_night_liss.h5
python my_script.py /mn/stornext/d16/cmbco/comap/protodir/maps/co2_map_complete_night_liss.h5 all dayn


maps/co6_map_complete_night_ces.h5

python my_script.py /mn/stornext/d16/cmbco/comap/protodir/maps/co6_map_complete_night_ces.h5 all dayn

maps/co6_map_complete_night_liss.h5

python my_script.py /mn/stornext/d16/cmbco/comap/protodir/maps/co6_map_complete_night_liss.h5 all dayn

calculate the feed-feed cross-spectra for all of these maps (start with using the dayn-split, but do all of them eventually)

'''
k_sim, xs_mean_sim, xs_sigma_sim = xs_feed_feed_grid('spectra/xs_20oct_1test_2splits_1st_sim_feed%01i_and_20oct_1test_2splits_2nd_sim_feed%01i.h5', 'xs_grid_test.png', ' of 1st sim split', ' of 2nd sim split')
xs_with_model('xs_mean_sim_null_1half.png', k_sim, xs_mean_sim, xs_sigma_sim)



'''
k_co2_night_ces_dayn, xs_mean_co2_night_ces_dayn, xs_sigma_co2_night_ces_dayn = xs_feed_feed_grid('spectra/xs_co2_map_complete_night_ces_1st_dayn_feed%01i_and_co2_map_complete_night_ces_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co2_night_ces.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co2_night_ces.png',k_co2_night_ces_dayn, xs_mean_co2_night_ces_dayn, xs_sigma_co2_night_ces_dayn)

k_co2_night_liss_dayn, xs_mean_co2_night_liss_dayn, xs_sigma_co2_night_liss_dayn = xs_feed_feed_grid('spectra/xs_co2_map_complete_night_liss_1st_dayn_feed%01i_and_co2_map_complete_night_liss_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co2_night_liss.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co2_night_liss.png',k_co2_night_liss_dayn, xs_mean_co2_night_liss_dayn, xs_sigma_co2_night_liss_dayn)

k_co6_night_ces_dayn, xs_mean_co6_night_ces_dayn, xs_sigma_co6_night_ces_dayn = xs_feed_feed_grid('spectra/xs_co6_map_complete_night_ces_1st_dayn_feed%01i_and_co6_map_complete_night_ces_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co6_night_ces.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co6_night_ces.png',k_co6_night_ces_dayn, xs_mean_co6_night_ces_dayn, xs_sigma_co6_night_ces_dayn)

k_co6_night_liss_dayn, xs_mean_co6_night_liss_dayn, xs_sigma_co6_night_liss_dayn = xs_feed_feed_grid('spectra/xs_co6_map_complete_night_liss_1st_dayn_feed%01i_and_co6_map_complete_night_liss_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co6_night_liss.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co6_night_liss.png',k_co6_night_liss_dayn, xs_mean_co6_night_liss_dayn, xs_sigma_co6_night_liss_dayn)


k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_1st_dayn_feed%01i_and_co7_map_complete_night_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co7_night2.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co7_night2.png', k_co7_night_dayn, xs_mean_co7_night_dayn, xs_sigma_co7_night_dayn)

k_co72, xs_mean_co72, xs_sigma_co72 = xs_feed_feed_grid('spectra/xs_co7_map_complete_wday_1st_dayn_feed%01i_and_co7_map_complete_wday_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_co7_wday.png', ' of 1st dayn split', ' of 2nd dayn split')
xs_with_model('xs_mean_dayn_co7_wday.png', k_co72, xs_mean_co72, xs_sigma_co72)

k_co73, xs_mean_co73, xs_sigma_co73 = xs_feed_feed_grid('spectra/xs_co7_map_complete_night_1st_half_feed%01i_and_co7_map_complete_night_2nd_half_feed%01i.h5', 'xs_grid_half_co7_night.png', ' of 1st half split', ' of 2nd half split')
xs_with_model('xs_mean_half_co7_night.png', k_co73, xs_mean_co73, xs_sigma_co73)

k_co74, xs_mean_co74, xs_sigma_co74 = xs_feed_feed_grid('spectra/xs_co7_map_complete_wday_1st_half_feed%01i_and_co7_map_complete_wday_2nd_half_feed%01i.h5', 'xs_grid_half_co7_wday.png', ' of 1st half split', ' of 2nd half split')
xs_with_model('xs_mean_half_co7_wday.png', k_co74, xs_mean_co74, xs_sigma_co74)


k_auto, xs_mean_auto, xs_sigma_auto = xs_mean_autoPS('spectra/xs_1octd_1test_2splits_coadded_and_1octd_1test_2splits_coadded.h5')

date = '1octd'
splits_collection = np.array(['2','4','8','16'])
#splits_collection = np.array(['2'])
splits_array = np.zeros(len(splits_collection))
for p in range(len(splits_collection)):
   splits_array[p] = int(splits_collection[p])

errorbars = []
for splits in splits_collection:
   path = 'spectra/xs_split%01i_' + date +'_1test_' + splits + 'splits_and_split%01i_' + date + '_1test_' + splits +'splits.h5'
   figure1 = date + '_xs_grid_' + splits + 'splits.png'
   figure2 = date + '_xs_mean_' + splits + 'splits.png'
   k_split, xs_mean_split, xs_sigma_split = xs_split_split_grid(path, figure1, int(splits))
   xs_mean_from_splits(figure2, k_split, xs_mean_split, xs_sigma_split)
   errorbars.append(xs_sigma_split)
errorbars = np.array(errorbars)

plt.figure()
for g in range(len(splits_array)):
   plt.plot(k_split,errorbars[g], label='%01i splits' %(splits_array[g]))
   plt.scatter(k_split,errorbars[g],s=5)
plt.plot(k_auto, xs_sigma_auto, label = 'auto PS', color='black')
plt.scatter(k_auto, xs_sigma_auto,s=5, color='black')
plt.legend(fontsize=12)
plt.xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=12)
plt.ylabel('xs sigma', fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.savefig(date + '_splits_allscales_error.png')
plt.show()

sums_of_errors = []
for g in range(len(splits_array)):
   sum_of_sigmas = 1. / np.sqrt(sum( 1./errorbars[g]**2.))
   sums_of_errors.append(sum_of_sigmas)


def sigma_of_N(x,a,b):
   return a*x+b

popt,pcov = curve_fit(sigma_of_N, splits_array, sums_of_errors)
print (popt) #[-0.04074033  0.27340673] before changes 


plt.figure()
plt.plot(splits_array, sigma_of_N(splits_array,*popt), label=r'best fit, $%.3fx + %.3f$' %(popt[0], popt[1]), color='paleturquoise',zorder=1)
plt.scatter(splits_array, sums_of_errors, color='lightseagreen',zorder=2)
plt.scatter(1, 1. / np.sqrt(sum( 1./xs_sigma_auto**2.)), label='auto PS', color='navy')
plt.xlabel('Number of map-splits', fontsize=12)
plt.ylabel(r'xs sigma across all scales, $\left( \sqrt{\sum_k \frac{1}{\sigma_k^2}} \right)^{-1}$ ', fontsize=12)
plt.legend(fontsize=12)
plt.savefig(date + '_splits_sum_error.png')
plt.show()
'''


'''
k_test, xs_mean_test, xs_sigma_test = xs_feed_feed_grid('spectra/xs_17sept_1test_feed%01i_and_17sept_1test_feed%01i.h5', 'xs_grid_test.png')
print ("xs mean: ", xs_mean_test)
xs_with_model('xs_mean_test.png', k_test, xs_mean_test, xs_sigma_test)

k_co2, xs_mean_co2, xs_sigma_co2 = xs_feed_feed_grid('spectra/xs_co2_map_complete_1st_half_feed%01i_and_co2_map_complete_2nd_half_feed%01i.h5', 'xs_grid_halfs_co2.png')
xs_with_model('xs_mean_full_co2.png', k_co2, xs_mean_co2, xs_sigma_co2)

k_co6, xs_mean_co6, xs_sigma_co6 = xs_feed_feed_grid('spectra/xs_co6_map_complete_1st_half_feed%01i_and_co6_map_complete_2nd_half_feed%01i.h5', 'xs_grid_halfs_co6.png')
xs_with_model('xs_mean_full_co6.png', k_co6, xs_mean_co6, xs_sigma_co6)

k_co7, xs_mean_co7, xs_sigma_co7 = xs_feed_feed_grid('spectra/xs_co7_map_complete_1st_half_feed%01i_and_co7_map_complete_2nd_half_feed%01i.h5', 'xs_grid_halfs_co7.png')
xs_with_model('xs_mean_full_co7.png', k_co7, xs_mean_co7, xs_sigma_co7)
'''


