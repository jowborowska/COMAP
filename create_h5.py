# GOAL: create a sample h5 file with generated map to run tests on it
# things needed in map_cosmo: x, y, map, rms, map_beam, rms_beam
# for a real map: x (120,) float64; y (120,) float64; map (19, 4, 64, 120, 120) float32; rms (19, 4, 64, 120, 120) float32; map_beam (4, 64, 120, 120) float32; rms_beam (4, 64, 120, 120) float32

#Comments:
# I need to run noise simulations to create rms for the files 
# give the program an option of choosing which field to "simulate"
# I can only start with x, y, map_beam, rms_beam and afterwards add map and rms, which hold values for different feeds
# ASK ABOUT CALCULATING RMS

import numpy as np
import numpy.fft as fft
import h5py
import tools

#this one is from tools
def compute_power_spec2d(x, k_bin_edges, dx=1, dy=1):
    n_x, n_y = x.shape
    Pk_2D = np.abs(fft.fftn(x)) ** 2 * dx * dy / (n_x * n_y)

    kx = np.fft.fftfreq(n_x, dx)
    ky = np.fft.fftfreq(n_y, dy)

    kgrid = np.sqrt(sum(ki ** 2 for ki in np.meshgrid(kx, ky, indexing='ij')))

    Pk_nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=Pk_2D[kgrid > 0])[0]
    nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

    # Pk = Pk_nmodes / nmodes
    # k = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
    k = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
    Pk = np.zeros_like(k)
    Pk[np.where(nmodes > 0)] = Pk_nmodes[np.where(nmodes > 0)] / nmodes[np.where(nmodes > 0)]
    return Pk, k, nmodes

def compute_PS(my_map,x,y):
    n_x = len(x)
    n_y = len(y)
    Delta_x = x[1] - x[0]
    Delta_y = y[1] - y[0]

    #physical wave number in each direction
    k_x = 2*np.pi*np.fft.fftfreq(n_x,Delta_x)
    k_y = 2*np.pi*np.fft.fftfreq(n_y,Delta_y)

    f_k = np.fft.fftn(my_map)
    PS = np.abs(f_k)**2
    k_array = np.zeros(my_map.shape)

    #fill k_array with abs(k) values
    index = 0
    for i in range(n_x):
       for j in range(n_y):
          k_array[i][j]= np.linalg.norm(np.array([k_x[i],k_y[j]]))
          index += 1

    #flatten PS array and k array 
    PS = PS.flatten()
    k_array = k_array.flatten()

    #co-ad PS values corresponding to the same k-values, use 25 k-intervals/bins
    k_histogram, k_bins = np.histogram(k_array, bins=25)
    PS_binned = np.zeros(len(k_histogram))
    for i in range(len(PS_binned)):
       for j in range(len(PS)):  #check all values if they are in this bin
          if k_array[j] > k_bins[i] and k_array[j] < k_bins[i+1]: #if we are in the correct k-bin, add the corresponding PS values together
             PS_binned[i] += PS[j]

    #binned and averaged PS
    PS_binned_averaged = PS_binned/k_histogram
    return k_bins, k_histogram, PS_binned_averaged #histogram (=nmodes)- how many in each bin, bins - gives bin-intervals

#x and y are bin centers from mapmaker - these are different for different fields - read them from a real map
def read_from_a_real_map(mapname): #mapnames I have: 'co7_011989_good_map.h5', 'co6_013836_200601_map.h5'
   with h5py.File(mapname, mode="r") as my_file:
      x = np.array(my_file['x'][:]) #these x, y are are bin centers from mapmaker
      y = np.array(my_file['y'][:])
   return x, y

#!!!!!!!!!!!!!!!! either make this to work in 2D or use my way fo creating the maps
#This one can be modified - P(k) defines the distribution, a temperature map is one realization from this distribution
def PS_function(k_array):
   print "k", k_array.shape
   PS_array = np.zeros(len(k_array))
   for i in range(len(k_array)):
      if k_array[i] != 0.: 
          PS_array[i] = k_array[i]**(-3.)
      else:
         PS_array[i] = 0.
   return PS_array

#Generate more PS's from created map and estimate uncertainty - IS THIS RMS I AM LOOKING FOR ?
def estimate_uncertainty(some_map,x,y,power_spectrum_function, no_of_sim):
   k_bins, k_histogram, PS_1 = compute_PS(some_map,x,y)
   no_of_bins = len(k_histogram)
   PS_array = np.zeros((no_of_sim,no_of_bins))
   PS_mean = np.zeros(no_of_bins) 
   for i in range(no_of_sim):
      some_map2 = create_map_2d(power_spectrum_function,x,y)
      k_bins, k_histogram, PS_2 = compute_PS(some_map2, x, y)   
      PS_array[i]= PS_2
      PS_mean += PS_2
   std = np.zeros(no_of_bins)
   for j in range(no_of_bins):
      std[j] = np.std(PS_array[:,j])
   PS_mean = PS_mean/float(no_of_sim)
   return std, k_bins[1:], PS_mean

#this is the one from tools, slightly modified
def create_map_2d(power_spectrum_function, x, y):
    n_x = len(x)
    n_y = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    A  = ((x[-1] - x[0])*(y[-1] - y[0]))
    fftfield = np.zeros((n_x, n_y), dtype=complex)
    z = power_spectrum_function(np.abs(np.sqrt(fft.fftfreq(n_x, d=dx)[:, None]**2 + fft.fftfreq(n_y, d=dy)[None, :]**2)))
    field = np.random.randn(n_x, n_y, 2)
    fftfield[:] = n_x * n_y * (field[:, :, 0] + 1j * field[:, :, 1])*np.sqrt(z/A)
    return np.real(np.fft.ifft2(fftfield))

def create_h5_map(power_spectrum_function, mapname, outputname):
  x,y = read_from_a_real_map(mapname)
  my_map = create_map_2d(power_spectrum_function,x,y)
  std, k_bins, PS_mean = estimate_uncertainty(my_map,x,y,power_spectrum_function,50)
  return std.shape

print create_h5_map(PS_function, 'co7_011989_good_map.h5', 'test_map.h5')


'''
# Maybe base rms calculation on something more like this: (from PS class)
def run_noise_sims(n_sims): #only white noise here
       if not self.weights_are_normalized: self.normalize_weights()
        
       rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
       for i in range(n_sims):
           randmap = self.map.rms * np.random.randn(*self.map.rms.shape)

           rms_ps[:, i] = tools.compute_power_spec3d(
               randmap * self.map.w, self.k_bin_edges,
               dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
               )[0]
       self.rms_ps_mean = np.mean(rms_ps, axis=1)
       self.rms_ps_std = np.std(rms_ps, axis=1)
       return self.rms_ps_mean, self.rms_ps_std

'''
