import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
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
           
          # if i != 7 and j != 7:
              try:
                  filepath = path_to_xs %(i+1, j+1)
                  with h5py.File(filepath, mode="r") as my_file:
                      #print ("finds file", i, j)
                      xs[i, j] = np.array(my_file['xs'][:])
                      #print (xs[i,j])
                      rms_xs_std[i, j] = np.array(my_file['rms_xs_std'][:])
                      #print (rms_xs_std[i,j])
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
              
              #if i != j and not np.isnan(chi2[i,j]): 
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
   plt.xlabel('Feed' + split1)
   plt.ylabel('Feed' + split2)
   cbar = plt.colorbar()
   cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()
   #print ("xs_div:", xs_div)
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div)



def xs_with_model(figure_name, k, xs_mean, xs_sigma, titlename, scan_strategy):
  
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
   transfer_Nils = scipy.interpolate.interp1d(k_Nils, T_Nils) 
   P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)
   if scan_strategy == 'ces':
      plotcolor = 'indianred'
   if scan_strategy == 'liss':
      plotcolor = 'teal'

   lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8
   fig = plt.figure()
   #fig.set_figwidth(8)
   ax1 = fig.add_subplot(211)
  
   ax1.errorbar(k, k * xs_mean / (transfer(k)*transfer_Nils(k)), k * xs_sigma / (transfer(k)*transfer_Nils(k)), fmt='o', color=plotcolor)
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10 \times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)
   if scan_strategy == 'ces':
      ax1.set_ylim(-lim*3, lim*3)              # ax1.set_ylim(0, 0.1)
   if scan_strategy == 'liss':
      ax1.set_ylim(-lim*2, lim*2)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.04,1.)
   ax1.set_xscale('log')
   ax1.set_title(titlename)
   ax1.grid()
   #ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5,1.]
   ax1.set_xticks(labnums)
   ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   #ax1.legend(ncol=3)
   
   ax2 = fig.add_subplot(212)
   #ax2.plot(k, diff_mean / error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='black')
  
   ax2.errorbar(k, xs_mean / xs_sigma, xs_sigma/xs_sigma, fmt='o', color=plotcolor)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax2.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=14)
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   ax2.set_ylim(-5, 5)
   ax2.set_xlim(0.04,1.)
   ax2.set_xscale('log')
   ax2.grid()
   #ax2.legend(ncol=3)
   ax2.set_xticks(labnums)
   ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   
   plt.tight_layout()
   #plt.legend()
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()



def call_all(mapname, split, scan_strategy):
   xs_files = 'spectra/xs_' + mapname + '_1st_' + split + '_feed%01i_and_' + mapname +'_2nd_' + split +'_feed%01i.h5'
   k, xs_mean, xs_sigma = xs_feed_feed_grid(xs_files, 'xs_grid_' + mapname + '_' + split + '.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split')
   name_list = mapname.split('_')
   titlename = name_list[0] + ' field, coadded ' + name_list[3] + ', ' + name_list[4] + ' ' + name_list[5] + ', ' + scan_strategy
   figurename = 'xs_mean_' + mapname + '_' + split + '.pdf'
   xs_with_model(figurename, k, xs_mean, xs_sigma, titlename, scan_strategy)
   print (figurename)
   #return k, xs_mean, xs_sigma

'''
call_all('co7_map_coadded_ambt_upper_elev_ces', 'dayn', 'ces')
call_all('co7_map_coadded_ambt_upper_elev_liss', 'dayn', 'liss')
call_all('co7_map_coadded_ambt_lower_elev_ces', 'dayn', 'ces')
call_all('co7_map_coadded_ambt_lower_elev_liss', 'dayn','liss')
call_all('co7_map_coadded_elev_lower_ambt_ces', 'dayn', 'ces')
call_all('co7_map_coadded_elev_lower_ambt_liss', 'dayn', 'liss')
call_all('co7_map_coadded_elev_upper_ambt_ces', 'dayn', 'ces')
call_all('co7_map_coadded_elev_upper_ambt_liss', 'dayn', 'liss')
'''


def call_all_both_coadded(mapname, split, scan_strategy):
   xs_files = 'spectra/xs_' + mapname + '_1st_' + split + '_feed%01i_and_' + mapname +'_2nd_' + split +'_feed%01i.h5'
   k, xs_mean, xs_sigma = xs_feed_feed_grid(xs_files, 'xs_grid_' + mapname + '_' + split + '.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split')
   name_list = mapname.split('_')
   titlename = name_list[0] + ' field, coadded El and Amb, '  + scan_strategy
   figurename = 'xs_mean_' + mapname + '_' + split + '.pdf'
   xs_with_model(figurename, k, xs_mean, xs_sigma, titlename, scan_strategy)
   print (figurename)
   #return k, xs_mean, xs_sigma
'''
#here I correct the naming liss/ces
call_all_both_coadded('co7_map_ces', 'dayn', 'liss')
call_all_both_coadded('co7_map_liss', 'dayn', 'ces')
call_all_both_coadded('co6_map_ces', 'dayn', 'liss')
call_all_both_coadded('co6_map_liss', 'dayn', 'ces')
call_all_both_coadded('co2_map_ces', 'dayn', 'liss')
call_all_both_coadded('co2_map_liss', 'dayn', 'ces')

'''
def call_all_wday(mapname, split, scan_strategy):
   xs_files = 'spectra/xs_' + mapname + '_1st_' + split + '_feed%01i_and_' + mapname +'_2nd_' + split +'_feed%01i.h5'
   k, xs_mean, xs_sigma = xs_feed_feed_grid(xs_files, 'xs_grid_' + mapname + '_' + split + '.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split')
   name_list = mapname.split('_')
   titlename = name_list[0] + ' field, ' + name_list[2] + ', ' + scan_strategy
   figurename = 'xs_mean_' + mapname + '_' + split + '.pdf'
   xs_with_model(figurename, k, xs_mean, xs_sigma, titlename, scan_strategy)
   print (figurename)
   #return k, xs_mean, xs_sigma

call_all_wday('co6_elmap_day_liss','elev', 'liss')
call_all_wday('co6_elmap_day_ces','elev', 'ces')
call_all_wday('co6_elmap_night_liss','elev', 'liss')
call_all_wday('co6_elmap_night_ces','elev', 'ces')

call_all_wday('co7_elmap_day_liss','elev', 'liss')
call_all_wday('co7_elmap_day_ces','elev', 'ces')
call_all_wday('co7_elmap_night_liss','elev', 'liss')
call_all_wday('co7_elmap_night_ces','elev', 'ces')

#call_all_wday('co2_map_day_liss', 'dayn','liss') #here scanning strategy should be correct, fixed that program version
#call_all_wday('co2_map_day_ces','dayn', 'ces')


