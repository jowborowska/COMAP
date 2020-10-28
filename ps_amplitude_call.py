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

def xs_feed_feed_grid_lower_half(path_to_xs, figure_name, split1, split2):
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
   #fill all the parts from upper half with nan
   chi2[:] = np.nan
   xs[:] = np.nan
   rms_xs_std[:] = np.nan
   noise[:] = np.nan

   for i in range(n_feed):
       for j in range(i):
           if i != 7 and j != 7:
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
              if i != j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
                  print ("if test worked")
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:
                  xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                  
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



def xs_feed_feed_grid_upper_half(path_to_xs, figure_name, split1, split2):
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
   #fill all the parts from upper half with nan
   chi2[:] = np.nan
   xs[:] = np.nan
   rms_xs_std[:] = np.nan
   noise[:] = np.nan

   for i in range(n_feed):
       for j in range(i):
           if i != 7 and j != 7:
              try:
                  filepath = path_to_xs %(j+1, i+1)
                  with h5py.File(filepath, mode="r") as my_file:
                      xs[j, i] = np.array(my_file['xs'][:])
                      rms_xs_std[j, i] = np.array(my_file['rms_xs_std'][:])
                      k[:] = np.array(my_file['k'][:])
              except:
                  xs[j, i] = np.nan
                  rms_xs_std[j, i] = np.nan
            
              w = np.sum(1 / rms_xs_std[j,i])
              noise[j,i] = 1 / np.sqrt(w)
              chi3 = np.sum((xs[j,i] / rms_xs_std[j,i]) ** 3) #we need chi3 to take the sign into account - positive or negative correlation

              chi2[j, i] = np.sign(chi3) * abs((np.sum((xs[j,i] / rms_xs_std[j,i]) ** 2) - n_k) / np.sqrt(2 * n_k)) #chi2 gives magnitude - how far it is from the white noise
              print ("chi2: ", chi2[j, i]) #this chi2 is very very big, so it never comes through the if-test - check how to generate maps with smaller chi2 maybe :)
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal   
              if i != j and not np.isnan(chi2[i,j]): #cut on chi2 not necessary for the testing
              #if abs(chi2[j,i]) < 5. and not np.isnan(chi2[j,i]) and j != i:
                  xs_sum += xs[j,i] / rms_xs_std[j,i] ** 2
                  print ("if test worked")
                  xs_div += 1 / rms_xs_std[j,i] ** 2
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

def xs_with_model(figure_name, k, xs_mean_l, xs_mean_u, xs_sigma_l, xs_sigma_u, title_name):
  
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
   transfer_Nils = scipy.interpolate.interp1d(k_Nils, T_Nils) 
   P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)
   
   error = ((xs_sigma_l**2 + xs_sigma_u**2)**(0.5))*0.5
   diff_mean = (xs_mean_u - xs_mean_l)/2.
   sum_mean = (xs_mean_u + xs_mean_l)/2.
   print ("diff and sum: ", diff_mean, sum_mean)
   lim = np.mean(np.abs(sum_mean[4:-2] * k[4:-2])) * 8
   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.errorbar(k, k * diff_mean / (transfer(k)*transfer_Nils(k)), k * error / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{diff}(k)$', color='teal')
  # ax1.errorbar(k, k * xs_mean_l / (transfer(k)*transfer_Nils(k)), k * xs_sigma_l / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{lower}(k)$', color='red')
  # ax1.errorbar(k, k * xs_mean_u / (transfer(k)*transfer_Nils(k)), k * xs_sigma_u / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{upper}(k)$', color='green')
   '''
   error_scaled = k * error / (transfer(k)*transfer_Nils(k))   
   
   ax1.plot(k, k * diff_mean / (transfer(k)*transfer_Nils(k)),  label=r'$k\tilde{C}_{diff}(k)$', color='teal')
   ax1.fill_between(x=k, y1=k * diff_mean / (transfer(k)*transfer_Nils(k)) - error_scaled, y2=k * diff_mean / (transfer(k)*transfer_Nils(k)) + error_scaled, facecolor='paleturquoise', edgecolor='paleturquoise')
   

   ax1.plot(k, k * sum_mean / (transfer(k)*transfer_Nils(k)),  label=r'$k\tilde{C}_{sum}(k)$', color='purple')
   ax1.fill_between(x=k, y1=k * sum_mean / (transfer(k)*transfer_Nils(k)) - error_scaled, y2=k * sum_mean / (transfer(k)*transfer_Nils(k)) + error_scaled, facecolor='plum', edgecolor='plum')
   '''
   ax1.set_title(title_name, fontsize=13)
 #  ax1.errorbar(k, k * sum_mean / (transfer(k)*transfer_Nils(k)), k * error / (transfer(k)*transfer_Nils(k)), fmt='o', label=r'$k\tilde{C}_{sum}(k)$', color='purple')
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean_l, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10 \times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=13)
   ax1.set_ylim(-lim*4, lim*4)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.03,k[-1]+0.1)
   ax1.set_xscale('log')
   ax1.grid()
   #plt.legend(bbox_to_anchor=(0, 0.61))
   ax1.legend(ncol=4)
   ax2 = fig.add_subplot(212)
   #ax2.plot(k, diff_mean / error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='black')
   ax2.errorbar(k, diff_mean / error, error/error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='lightseagreen')
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   #ax2.errorbar(k, xs_mean_l / xs_sigma_l, xs_sigma_l/xs_sigma_l, fmt='o', label=r'$\tilde{C}_{lower}(k)$', color='red')
   #ax2.errorbar(k, xs_mean_u / xs_sigma_u, xs_sigma_u /xs_sigma_u, fmt='o', label=r'$\tilde{C}_{upper}(k)$', color='green')


   ax2.plot(k, 0 * xs_mean_l, 'k', alpha=0.4)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=13)
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=13)
   ax2.set_ylim(-5, 5)
   ax2.set_xlim(0.03,k[-1]+0.1)
   ax2.set_xscale('log')
   ax2.grid()
   ax2.legend(ncol=4)
   plt.tight_layout()
   plt.legend(ncol=4)
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


def xs_feed_feed_grid_new(path_to_xs, figure_name, split1, split2, bigger_smaller):
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
   chi2[:] = np.nan
   xs[:] = np.nan
   rms_xs_std[:] = np.nan
   noise[:] = np.nan
   for i in range(n_feed):
       for j in range(n_feed):
           if i != 7 and j != 7:
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
              if bigger_smaller == True:
                 if i > 18-j and i!=j: #cut on chi2 not necessary for the testing              
                    chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k)) #chi2 gives magnitude - how far it is from the white noise

              if bigger_smaller == False:
                 if i < 18-j and i!=j:
                    chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k)) #chi2 gives magnitude - how far it is from the white noise

              #print ("chi2: ", chi2[i, j]) #this chi2 is very very big, so it never comes through the if-test - check how to generate maps with smaller chi2 maybe :)
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal      
              if bigger_smaller == True:
                 if i > 18-j and not np.isnan(chi2[i,j]) and i !=j: #cut on chi2 not necessary for the testing
                    xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                    print ("if test worked")
                    xs_div += 1 / rms_xs_std[i,j] ** 2
                    n_sum += 1

              if bigger_smaller == False:
                 if i < 18-j and not np.isnan(chi2[i,j]) and i!=j:
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

'''
k_co7_night_dayn_l, xs_mean_co7_night_dayn_l, xs_sigma_co7_night_dayn_l = xs_feed_feed_grid_lower_half('spectra/xs_co7_map_complete_night_1st_dayn_feed%01i_and_co7_map_complete_night_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_lhalf.png', ' of 1st dayn split', ' of 2nd dayn split')

k_co7_night_dayn_u, xs_mean_co7_night_dayn_u, xs_sigma_co7_night_dayn_u = xs_feed_feed_grid_upper_half('spectra/xs_co7_map_complete_night_1st_dayn_feed%01i_and_co7_map_complete_night_2nd_dayn_feed%01i.h5', 'xs_grid_dayn_uhalf.png', ' of 1st dayn split', ' of 2nd dayn split')

xs_with_model('xs_mean_dayn_co7_night_halfs_uplow.png', k_co7_night_dayn_l, xs_mean_co7_night_dayn_l, xs_mean_co7_night_dayn_u, xs_sigma_co7_night_dayn_l,xs_sigma_co7_night_dayn_u, 'Daytime-Nighttime data split')



k_co7_night_half_l, xs_mean_co7_night_half_l, xs_sigma_co7_night_half_l = xs_feed_feed_grid_lower_half('spectra/xs_co7_map_complete_night_1st_half_feed%01i_and_co7_map_complete_night_2nd_half_feed%01i.h5', 'xs_grid_half_lhalf.png', ' of 1st half split', ' of 2nd half split')

k_co7_night_half_u, xs_mean_co7_night_half_u, xs_sigma_co7_night_half_u = xs_feed_feed_grid_upper_half('spectra/xs_co7_map_complete_night_1st_half_feed%01i_and_co7_map_complete_night_2nd_half_feed%01i.h5', 'xs_grid_half_uhalf.png', ' of 1st half split', ' of 2nd half split')

xs_with_model('xs_mean_half_co7_night_halfs_null.pdf', k_co7_night_half_l, xs_mean_co7_night_half_l, xs_mean_co7_night_half_u, xs_sigma_co7_night_half_l,xs_sigma_co7_night_half_u, 'Half mission split' )


k_co7_night_sidr_l, xs_mean_co7_night_sidr_l, xs_sigma_co7_night_sidr_l = xs_feed_feed_grid_lower_half('spectra/xs_co7_map_complete_night_1st_sidr_feed%01i_and_co7_map_complete_night_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_lhalf.png', ' of 1st sidr split', ' of 2nd sidr split')

k_co7_night_sidr_u, xs_mean_co7_night_sidr_u, xs_sigma_co7_night_sidr_u = xs_feed_feed_grid_upper_half('spectra/xs_co7_map_complete_night_1st_sidr_feed%01i_and_co7_map_complete_night_2nd_sidr_feed%01i.h5', 'xs_grid_sidr_uhalf.png', ' of 1st sidr split', ' of 2nd sidr split')

xs_with_model('xs_mean_sidr_co7_night_halfs_null.pdf', k_co7_night_sidr_l, xs_mean_co7_night_sidr_l, xs_mean_co7_night_sidr_u, xs_sigma_co7_night_sidr_l,xs_sigma_co7_night_sidr_u, 'Sidereal time split' )




k_sim1, xs_mean_sim1, xs_sigma_sim1 = xs_feed_feed_grid_new('spectra/xs_20oct_1test_2splits_1st_sim_feed%01i_and_20oct_1test_2splits_2nd_sim_feed%01i.h5', 'xs_grid_test.png', ' of 1st sim split', ' of 2nd sim split', True)
k_sim2, xs_mean_sim2, xs_sigma_sim2 = xs_feed_feed_grid_new('spectra/xs_20oct_1test_2splits_1st_sim_feed%01i_and_20oct_1test_2splits_2nd_sim_feed%01i.h5', 'xs_grid_test.png', ' of 1st sim split', ' of 2nd sim split', False)


k_sim_l, xs_mean_sim_l, xs_sigma_sim_l = xs_feed_feed_grid_lower_half('spectra/xs_20oct_1test_2splits_1st_sim_feed%01i_and_20oct_1test_2splits_2nd_sim_feed%01i.h5', 'xs_grid_test_l.png', ' of 1st sim split', ' of 2nd sim split')

k_sim_u, xs_mean_sim_u, xs_sigma_sim_u = xs_feed_feed_grid_upper_half('spectra/xs_20oct_1test_2splits_1st_sim_feed%01i_and_20oct_1test_2splits_2nd_sim_feed%01i.h5', 'xs_grid_test_u.png', ' of 1st sim split', ' of 2nd sim split')

xs_with_model('xs_mean_sim_null.png', k_sim1, xs_mean_sim1, xs_mean_sim2, xs_sigma_sim2, xs_sigma_sim2, 'Simulated split')
'''

def call_all(mapname, split):
   xs_files = 'spectra/xs_' + mapname + '_1st_' + split + '_feed%01i_and_' + mapname +'_2nd_' + split +'_feed%01i.h5'
   kl, xs_mean_l, xs_sigma_l = xs_feed_feed_grid_new(xs_files, 'xs_grid_' +mapname + '_lower.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split', True)
   ku, xs_mean_u, xs_sigma_u = xs_feed_feed_grid_new(xs_files, 'xs_grid_' +mapname + '_upper.png', ' of 1st ' + split + ' split', ' of 2nd ' + split + ' split', False)
   xs_with_model('xs_mean_' + mapname + '_null.png', kl, xs_mean_l, xs_mean_u, xs_sigma_l, xs_sigma_u, mapname + ', '+ split + 'split')
   print ("Created files:")
   print('xs_grid_' +mapname + '_lower.png')
   print('xs_grid_' +mapname + '_upper.png')
   print('xs_mean_' + mapname + '_null.png')

call_all('co2_map_complete_sunel_ces', 'dayn')
'''
call_all('20oct_1test_2splits', 'sim')
'''


