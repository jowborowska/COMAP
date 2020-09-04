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

def xs_feed_feed_grid(path_to_xs):
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
           chi3 = np.sum((xs[i,j] / rms_xs_std[i,j]) ** 3)

           chi2[i, j] = np.sign(chi3) * (np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k)
           if abs(chi2[i,j]) < 5.0 and not np.isnan(chi2[i,j]) and i != j: # and i != 7 and j != 7 and i!=1 and j!=1 and i!=10 and j!=10: # and (i == 4 or j == 4):
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
   plt.savefig('xs_grid_par_co2_full.png', bbox_inches='tight')
   plt.show()

xs_feed_feed_grid('spectra/xsco2_map_complete.h5_1st_half_feed%01i_and_co2_map_complete.h5_2nd_half_feed%01i.h5')
