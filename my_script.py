import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import my_class
import multiprocessing

def run_all_methods(feed,feed1,feed2):
   my_xs = my_class.CrossSpectrum_nmaps(list_of_n_map_names,jk, feed, feed1, feed2)

   calculated_xs = my_xs.get_information()
   if feed1!=None and feed2!=None:
      print "Created xs between", calculated_xs[0][1], "and", calculated_xs[0][2] #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair
   else:
      print calculated_xs
   xs, k, nmodes = my_xs.calculate_xs()

   rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

   my_xs.make_h5()

   #plot all cross-spectra that have been calculated
   if feed1!=None and feed2!=None:
      my_xs.plot_xs(k, xs, rms_sig, rms_mean, 0, save=True)
   else:
      for i in range(len(calculated_xs)):
         my_xs.plot_xs(k, xs, rms_sig, rms_mean, i, save=True)

def all_feed_combo_xs(p):
    i = p // 19 + 1 #floor division, divides and returns the integer value of the quotient (it dumps the digits after the decimal)
    j = p % 19 + 1 #modulus, divides and returns the value of the remainder
    
    if i == 4 or i == 6 or i == 7: #avoid these feeds (were turned off for most of the mission)
        return p
    if j == 4 or j == 6 or j == 7: #avoid these feeds (were turned off for most of the mission)
        return p
    run_all_methods(None, feed1=i,feed2=j)
    return p

n = len(sys.argv) - 3 #number of maps
list_of_n_map_names = []

if len(sys.argv) < 4 :
    print('Provide at least one file name (for xs between half splits) or two file names (for xs between whole maps)!')
    print('Then specify the feed number or make xs for all feeds or for coadded feeds; then the name of the jk or False (for entire map)!') 
    print('Usage: python my_script.py mapname_1 mapname_2 ... mapname_n feed_number/coadded/all dayn/half/odde/sdlb/False') #odde and sdlb work only with 'coadded'
    sys.exit(1)
if len(sys.argv) == 4 and sys.argv[-1] == 'False' and sys.argv[-2] != 'all':
    print('Only one file name specified, with no split - unable to create xs! Try for all feed-combo or give more maps/splits.')
    sys.exit(1)

for i in range(n):
    list_of_n_map_names.append(sys.argv[i+1])

if sys.argv[-1] == 'False': #if False, takes the map made out of entire data set
   jk = False 
if sys.argv[-1] == 'dayn': #day/night split
   jk = 'dayn'
if sys.argv[-1] == 'half': #half mission splits
   jk = 'half'
if sys.argv[-1] == 'odde': #splits odd/even numbered obsIDs
   jk = 'odde'
if sys.argv[-1] == 'sdlb': #splits the four saddlebags
   jk = 'sdlb'

feed_name = sys.argv[-2]
if feed_name == 'coadded':
   feed = 30 #a random number meaning that we take the coadded feed-maps
if feed_name != 'coadded' and feed_name != 'all':
   feed = int(sys.argv[-2])
if feed_name == 'all': #'all' makes xs for all feed-combinations, either give it one map name or one map name with half split = True !
   feed_combos = list(range(19*19)) #number of combinations between feeds
   pool = multiprocessing.Pool(24) #here number of cores
   np.array(pool.map(all_feed_combo_xs, feed_combos))
if feed_name != 'all':
   run_all_methods(feed, None, None)
