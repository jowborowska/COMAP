#Developing a class, which takes in a list of n maps (n map names, since we call map_cosmo inside this class)

#Comments:-------------------------------------------------------------------------------------------------
# we probably want to call the map_cosmo class from within the new cross spectrum class, since we do not neccesarily want to store all the maps in memory at once - DONE
# maybe try to return the index of a given xs, so that we always know which maps were taken and can use it in plot - DONE
# then download a third map and check if everything works fine - DONE
# remember that these error bars come from random simulations, so they will be a bit different each time 
# add names of maps in plotting method and the "save" option - DONE
# check h5 creating method (the current version is not tested and changed very much from Havard's one) - DONE

import numpy as np
import h5py
import tools
import map_cosmo
import matplotlib.pyplot as plt
import PS_function #P(k) = k**-3
#import create_map_h5_new.PS_function as PS_f #<-- this didn't work, wanted the same command arguments as create_map_h5_new.py

class CrossSpectrum_nmaps():
    def __init__(self, list_of_n_map_names, jk=False, feed=None, feed1=None, feed2=None, n_of_splits=2):
        if feed == 30:
           feed = None
           self.feed_name = '_coadded'
        if feed != 30 and feed is not None:
           self.feed_name = '_feed' + str(feed)
        else:
           self.feed_name1 = '_feed' + str(feed1)
           self.feed_name2 = '_feed' + str(feed2)
        self.names_of_maps = list_of_n_map_names #the names schould indicate which map and feed we take
        self.names = []
        for name in self.names_of_maps:
           name = name.rpartition('/')[-1] #get rid of the path, leave only the name of the map
           name = name.rpartition('.')[0] #get rid of the ".h5" part
           if jk == False:
              if feed1 == None and feed2 == None:
                 self.names.append(name + self.feed_name)
              else:
                 self.names.append(name + self.feed_name1)
                 self.names.append(name + self.feed_name2)
           if jk != False and jk != 'sim':
              if feed1 == None and feed2 == None:
                 self.names.append(name + '_1st_' + jk + self.feed_name)
                 self.names.append(name + '_2nd_'+ jk + self.feed_name)
              else:
                 self.names.append(name + '_1st_' + jk + self.feed_name1)
                 self.names.append(name + '_2nd_' + jk + self.feed_name2)
           if jk != False and jk == 'sim':
              for g in range(n_of_splits):
                 map_split_number = g + 1
                 map_split_name ='split%01i_' %(map_split_number) + name
                 print (map_split_name)
                 self.names.append(map_split_name)
                 

        self.maps = []
        for map_name in list_of_n_map_names:
           if jk != False:
              if feed1==None and feed2==None:
                 for split_no in range(n_of_splits): #there are two splits from mapmaker so far, can be more from simulations
                    my_map_split = map_cosmo.MapCosmo(map_name, feed, jk, split_no)
                    self.maps.append(my_map_split)
                    
              else:
                 my_map_first = map_cosmo.MapCosmo(map_name, feed1, jk, 0)
                 my_map_second = map_cosmo.MapCosmo(map_name, feed2, jk, 1)
                 self.maps.append(my_map_first)
                 self.maps.append(my_map_second)
           else:
              if feed1==None and feed2==None:
                 my_map = map_cosmo.MapCosmo(map_name, feed) 
                 self.maps.append(my_map)
              else:
                 my_map1 = map_cosmo.MapCosmo(map_name, feed1) 
                 my_map2 = map_cosmo.MapCosmo(map_name, feed2)
                 self.maps.append(my_map1)
                 self.maps.append(my_map2)
        #self.weights_are_normalized = False
   
    #NORMALIZE WEIGHTS FOR A GIVEN PAIR OF MAPS
    def normalize_weights(self, i, j):
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w / norm
        self.maps[j].w = self.maps[j].w / norm
       
    #REVERSE NORMALIZE_WEIGHTS, TO NORMALIZE EACH PAIR OF MAPS INDEPENDENTLY
    def reverse_normalization(self, i, j):    
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w*norm
        self.maps[j].w = self.maps[j].w*norm

    #INFORM WHICH XS INDEX CORRESPONDS TO WHICH MAP-PAIR
    def get_information(self):
        indexes_xs = []
        index = -1 
        for i in range(len(self.maps)):
          for j in range(i,len(self.maps)): 
             if i != j: 
                index += 1
                indexes_xs.append([index,self.names[i],self.names[j]])
        return indexes_xs
             
    #COMPUTE ALL THE XS BETWEEN THE n FEED-AVERAGED MAPS
    def calculate_xs(self, no_of_k_bins=15): #here take the number of k-bins as an argument 
        n_k = no_of_k_bins
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
        
        #store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        for i in range(len(self.maps)):
           for j in range(i,len(self.maps)): #ensure that we don't compute the same pairs of maps twice <---------is this correct ?
              if i != j: #ensure that we compute xs, not auto spectrum
                 
                 self.normalize_weights(i,j) #normalize weights for given xs pair of maps
                 
                 
                 my_xs, my_k, my_nmodes = tools.compute_cross_spec3d(
                 (self.maps[i].map * np.sqrt(self.maps[i].w*self.maps[j].w), self.maps[j].map * np.sqrt(self.maps[i].w*self.maps[j].w)),
                 self.k_bin_edges, dx=self.maps[i].dx, dy=self.maps[i].dy, dz=self.maps[i].dz)

                 self.reverse_normalization(i,j) #go back to the previous state to normalize again with a different map-pair

                 self.xs.append(my_xs)
                 self.k.append(my_k)
                 self.nmodes.append(my_nmodes)
        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)
        return self.xs, self.k, self.nmodes
   
    #RUN NOISE SIMULATIONS (for all combinations of n maps, to match xs)
    def run_noise_sims(self, n_sims, seed=None):
        self.rms_xs_mean = []
        self.rms_xs_std = []
        for i in range(len(self.maps)):
           for j in range(i,len(self.maps)): #ensure that we don't take the same pairs of maps twice 
              if i != j: #only for xs, not auto spectra

                 self.normalize_weights(i,j)

                 if seed is not None:
                     if self.maps[i].feed is not None:
                         feeds = np.array([self.maps[i].feed, self.maps[j].feed])
                     else:
                         feeds = np.array([1, 1])
            
                 rms_xs = np.zeros((len(self.k_bin_edges) - 1, n_sims))
                 for g in range(n_sims):
                     randmap = [np.zeros(self.maps[i].rms.shape), np.zeros(self.maps[i].rms.shape)]
                     for l in range(2):
                         if seed is not None:
                             np.random.seed(seed * (g + 1) * (l + 1) * feeds[l])
                         randmap[l] = np.random.randn(*self.maps[l].rms.shape) * self.maps[l].rms

                     rms_xs[:, g] = tools.compute_cross_spec3d(
                         (randmap[0] * np.sqrt(self.maps[i].w*self.maps[j].w), randmap[1] * np.sqrt(self.maps[i].w*self.maps[j].w)),
                         self.k_bin_edges, dx=self.maps[i].dx, dy=self.maps[i].dy, dz=self.maps[i].dz)[0]
                 
                 self.reverse_normalization(i,j) #go back to the previous state to normalize again with a different map-pair

                 self.rms_xs_mean.append(np.mean(rms_xs, axis=1))
                 self.rms_xs_std.append(np.std(rms_xs, axis=1))
        return self.rms_xs_mean, self.rms_xs_std
    
    #MAKE SEPARATE H5 FILE FOR EACH XS
    def make_h5(self, outname=None):
        index = -1 
        for i in range(len(self.maps)):
           for j in range(i,len(self.maps)): #ensure that we don't take the same pairs of maps twice 
              if i != j: #only for xs, not auto spectra
                 index += 1 #find a correct xs, etc.

                 if outname is None:
                     tools.ensure_dir_exists('spectra')
                     outname = 'spectra/xs_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.h5'          

                 f1 = h5py.File(outname, 'w')
                 try:
                     f1.create_dataset('mappath1', data=self.maps[i].mappath)
                     f1.create_dataset('mappath2', data=self.maps[j].mappath)
                     f1.create_dataset('xs', data=self.xs[index])
                     f1.create_dataset('k', data=self.k[index])
                     f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
                     f1.create_dataset('nmodes', data=self.nmodes[index])
                 except:
                     print('No power spectrum calculated.')
                     return 
                 try:
                     f1.create_dataset('rms_xs_mean', data=self.rms_xs_mean[index])
                     f1.create_dataset('rms_xs_std', data=self.rms_xs_std[index])
                 except:
                     pass
                 
                 f1.close()

    #PLOT XS (previously in the script code)
    def plot_xs(self, k_array, xs_array, rms_sig_array, rms_mean_array, index, save=False):
       
       k = k_array[index]
       xs = xs_array[index]
       rms_sig = rms_sig_array[index]
       rms_mean = rms_mean_array[index]
       
       
       
       #lim = 200.
       fig = plt.figure()
       fig.suptitle('xs of ' + self.get_information()[index][1] + ' and ' + self.get_information()[index][2])
       ax1 = fig.add_subplot(211)
       ax1.errorbar(k, k*xs, k*rms_sig, fmt='o', label=r'$k\tilde{C}_{data}(k)$') #added k*
       ax1.plot(k, 0 * rms_mean, 'k', label=r'$\tilde{C}_{noise}(k)$', alpha=0.4)
       ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
       ax1.set_ylabel(r'$\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^3$]')
       
       lim = np.mean(np.abs(xs[4:])) * 4
       if not np.isnan(lim):
          ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)

       ax1.set_xscale('log')
       ax1.grid()
       plt.legend()

       ax2 = fig.add_subplot(212)
       ax2.errorbar(k, xs / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
       ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
       ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
       ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
       ax2.set_ylim(-12, 12)
       ax2.set_xscale('log')
       ax2.grid()
       plt.legend()
       if save==True:
          tools.ensure_dir_exists('figures')
          name_for_figure = 'figures/xs_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.png'
          plt.savefig(name_for_figure, bbox_inches='tight')
          print ('Figure saved as', name_for_figure)

       #plt.show()


