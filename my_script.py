import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import my_class

n = len(sys.argv) - 3 #number of maps
list_of_n_map_names = []


if len(sys.argv) < 4 :
    print('Provide at least one file name (for xs between half splits) or two file names (for xs between whole maps)!')
    print('Then specify the feed (or write: coadded) and True/False for the half split!')
    print('Usage: python my_script.py mapname_1 mapname_2 ... mapname_n feed_number/coadded True/False')
    sys.exit(1)

for i in range(n):
    list_of_n_map_names.append(sys.argv[i+1])

if sys.argv[-2] == 'coadded':
   feed = None
else:
   feed = int(sys.argv[-2]) #if None, takes the coadded feeds
print 'feed', feed
half_split = bool(sys.argv[-1]) #if False, takes the map made out of entire data set

my_xs = my_class.CrossSpectrum_nmaps(list_of_n_map_names, half_split, feed)

calculated_xs = my_xs.get_information()
print calculated_xs #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair

xs, k, nmodes = my_xs.calculate_xs()

rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

my_xs.make_h5()

#plot all cross-spectra that have been calculated
for i in range(len(calculated_xs)):
   my_xs.plot_xs(k, xs, rms_sig, rms_mean, i, save=False)


