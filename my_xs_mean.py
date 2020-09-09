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

def xs_feed_feed_grid(path_to_xs, figure_name):
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

           chi2[i, j] = np.sign(chi3) * (np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k) #chi2 gives magnitude - how far it is from the white noise
           if abs(chi2[i,j]) < 5.0 and not np.isnan(chi2[i,j]) and i != j: #if excess power is smaller than 5 sigma and chi2 is not nan, and we are not on the diagonal
               xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
               xs_div += 1 / rms_xs_std[i,j] ** 2
               n_sum += 1


   plt.figure()
   vmax = 15
   plt.imshow(chi2, interpolation='none', vmin=-vmax, vmax=vmax, extent=(0.5, n_feed + 0.5, n_feed + 0.5, 0.5))
   new_tick_locations = np.array(range(n_feed)) + 1
   plt.xticks(new_tick_locations)
   plt.yticks(new_tick_locations)
   plt.xlabel('Feed')
   plt.ylabel('Feed')
   cbar = plt.colorbar()
   cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
   plt.savefig(figure_name, bbox_inches='tight')
   plt.show()
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div)

def xs_with_model(figure_name, k_th, ps_th, ps_th_nobeam, ps_copps, ps_copps_nobeam):
  
   transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
   lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8

   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.errorbar(k, k * xs_mean / transfer(k), k * xs_sigma / transfer(k), fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   ax1.plot(k_th, k_th * ps_th_nobeam * 20, 'r--', label=r'$20 \times kP_{Theory}(k)$')
   ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
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
   plt.show()

#theory spectrum
k_th = np.load('k.npy')
ps_th = np.load('ps.npy')
ps_th_nobeam = np.load('psn.npy') #instrumental beam, less sensitive to small scales line broadening, error bars go up at high k, something with the intrinsic resolution of the telescope (?)

#values from COPPS
ps_copps = 8.746e3 * ps_th / ps_th_nobeam #shot noise level
ps_copps_nobeam = 8.7e3

k, xs_mean, xs_sigma = xs_feed_feed_grid('spectra/xsco2_map_complete.h5_1st_half_feed%01i_and_co2_map_complete.h5_2nd_half_feed%01i.h5', 'xs_grid_halfs_co2.png')
xs_with_model('xs_mean_full.png', k_th, ps_th, ps_th_nobeam, ps_copps, ps_copps_nobeam)
