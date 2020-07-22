import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import my_class

n = len(sys.argv) - 1 #number of maps
list_of_n_map_names = []

for i in range(n):
    list_of_n_map_names.append(sys.argv[i+1])
if n < 2:
    print('Missing filenames!')
    print('Usage: python my_script.py mapname_1 mapname_2 ... mapname_n') #python my_script.py co6_013836_200601_map.h5 co6_013855_200602_map.h5
    sys.exit(1)


my_xs = my_class.CrossSpectrum_nmaps(list_of_n_map_names)

calculated_xs = my_xs.get_information(list_of_n_map_names) 
print calculated_xs #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair

xs, k, nmodes = my_xs.calculate_xs()

rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

my_xs.make_h5()

#plot all cross-spectra that have been calculated
for i in range(len(calculated_xs)):
   my_xs.plot_xs(k, xs, rms_sig, rms_mean, i)


