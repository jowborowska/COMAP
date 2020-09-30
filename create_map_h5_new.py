# GOAL: create a sample h5 file with generated map to run tests on it
# things needed in map_cosmo: x, y, map, rms, map_beam, rms_beam
# for a real map: x (120,) float64; y (120,) float64; map (19, 4, 64, 120, 120) float32; rms (19, 4, 64, 120, 120) float32; map_beam (4, 64, 120, 120) float32; rms_beam (4, 64, 120, 120) float32

#Comments:
# maybe make an option of generaring x,y,freq without a real map available or for different fields ?
# ask about details of create_output_map(x,y,z)

import numpy as np
import numpy.fft as fft
import h5py
import tools
import sys
import PS_function

#x and y are bin centers from mapmaker - these are different for different fields - read them and frequency from a real map
def read_from_a_real_map(mapname): #examples of mapnames I have on my computer: 'co7_011989_good_map.h5', 'co6_013836_200601_map.h5'
   with h5py.File(mapname, mode="r") as my_file:
      frequency = np.array(my_file['freq'][:])
      frequency = np.reshape(frequency, 4*64)
      x = np.array(my_file['x'][:]) #these x, y are are bin centers from mapmaker
      y = np.array(my_file['y'][:])
   return frequency, x, y

#convert x and y from degrees to Mpc and frequency to redshift to Mpc
def x_y_freq_to_Mpc(x,y,freq):
   n_f = len(freq) #256
   h = 0.7
   deg2mpc = 76.22 / h  # at redshift 2.9
   dz2mpc = 699.62 / h # redshift 2.4 to 3.4
   z_mid = 2.9
   dnu = 32.2e-3  # GHz
   nu_rest = 115  # GHz
   dz = (1 + z_mid) ** 2 * dnu / nu_rest  # conversion 
   redshift = np.linspace(z_mid - n_f/2*dz, z_mid + n_f/2*dz, n_f + 1)
   meandec = np.mean(y)
   x = x * deg2mpc * np.cos(meandec * np.pi / 180) #account for the effect that 'circles of latitude' around poles are smaller than equator
   y = y * deg2mpc 
   z = tools.edge2cent(redshift * dz2mpc)
   return x,y,z

#almost the same as in tools, modified n_x, n_y, n_z to keep the correct shape
def create_map_3d(power_spectrum_function, x, y, z):
    n_x = len(x) 
    n_y = len(y) 
    n_z = len(z) 
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    V = ((x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0]))  # whole volume in Mpc^3
    fftfield = np.zeros((n_x, n_y, n_z), dtype=complex)
    z = power_spectrum_function(
        np.abs(2.*np.pi*np.sqrt(fft.fftfreq(n_x, d=dx)[:, None, None]**2
                       + fft.fftfreq(n_y, d=dy)[None, :, None]**2
                       + fft.fftfreq(n_z, d=dz)[None, None, :]**2))
    )
    #np.random.seed(1) #to get the same signal every time
    field = np.random.randn(n_x, n_y, n_z, 2)
    #seed = None
    fftfield[:] = n_x * n_y * n_z * (field[:, :, :, 0] + 1j * field[:, :, :, 1])*np.sqrt(z/V)
    return np.real(np.fft.ifftn(fftfield)), dx*dy*dz

#simulate noise and create the map (signal + noise)
def create_output_map(x,y,z, signal_map): 
   muK2K = 1e-6 #micro Kelvins to Kelvins
   #muK2K = 1. #keep everything in micro Kelvins
   #x_ind, y_ind, z_ind = np.indices(signal_map.shape)
   #r = np.hypot(x[x_ind] - 2, y[y_ind] - 2, z[z_ind] - 2)
   #rms_map = (r / np.max(r.flatten()) + 0.05) * np.std(signal_map.flatten()) ** 2.5 / 5.0
   #rms_map = rms_map*muK2K
   #np.random.seed(1)
   rms_map = 10.*muK2K + 10.*np.random.uniform(0.0, 1.*muK2K, (120, 120, 256)) #rms drawn from uniform dist of 10 muK, the standard deviation of the noise in each voxel
   #rms_map = np.zeros_like(rms_map)+5.*muK2K
   #np.random.seed() #keep the same rms all the time
   w = 1./rms_map ** 2
   noise_map = np.random.randn(*rms_map.shape) * rms_map
   output_map = signal_map*muK2K + noise_map
   #output_map = noise_map
   return output_map.transpose(2, 0, 1), rms_map.transpose(2, 0, 1), signal_map.transpose(2, 0, 1)*muK2K, w.transpose(2, 0, 1)
   #return output_map, rms_map, signal_map, w



def create_highest_split_map(x,y,z,signal_map,n_splits):
   no_of_feeds = 19
   map_beam_shape = (4, 64, 120, 120)
   map_split_shape = (n_splits,19, 4, 64, 120, 120)
   map_split = np.zeros(map_split_shape)
   rms_split = np.zeros(map_split_shape)
   weights_split = np.zeros(map_split_shape)
   for g in range(n_splits):
      for i in range(no_of_feeds):
         output_map_single_feed, rms_map_single_feed, signal_map_single_feed, weights_single_feed = create_output_map(x,y,z, signal_map)
         output_map_single_feed = np.reshape(output_map_single_feed,map_beam_shape)
         rms_map_single_feed = np.reshape(rms_map_single_feed,map_beam_shape)
         weights_single_feed = np.reshape(weights_single_feed,map_beam_shape)
         map_split[g,i] = output_map_single_feed 
         rms_split[g,i] = rms_map_single_feed
         weights_split[g,i] = weights_single_feed
   return map_split, rms_split, weights_split

#create main map and beam map - this is needed only once for all jk maps
def coadd_splits_to_whole_map(from_n_splits,from_map_split, from_rms_split, from_weights_split):
   no_of_feeds = 19
   map_shape = (19, 4, 64, 120, 120)
   map_beam_shape = (4, 64, 120, 120)
   data_map = np.zeros(map_shape)
   data_beam_map = np.zeros(map_beam_shape) 
   rms_map = np.zeros(map_shape)
   rms_beam_map = np.zeros(map_beam_shape) 
   w_sum1 = np.zeros(map_shape) 
   w_sum2 = np.zeros(map_beam_shape) 
   for i in range(from_n_splits):
      w_sum1 += from_weights_split[i]
      data_map += from_weights_split[i]*from_map_split[i]
   data_map = data_map/w_sum1
   rms_map = w_sum1**(-0.5)
   for j in range(no_of_feeds):
      w_sum2 += 1./rms_map[j]**2
      data_beam_map += (1./rms_map[j]**2)*data_map[j]
   data_beam_map = data_beam_map/w_sum2
   rms_beam_map = w_sum2**(-0.5)
   return data_map, rms_map, data_beam_map, rms_beam_map

#the point here is to read a map with highest number of splits and then co-add splits to create maps with smaller number of splits and the final mapof whole data
def coadd_splits_to_splits(from_n_splits, to_n_splits, from_map_split, from_rms_split):
   map_split_shape = (to_n_splits,19, 4, 64, 120, 120)
   to_map_split = np.zeros(map_split_shape)
   to_rms_split = np.zeros(map_split_shape)
   for j in range(to_n_splits):
      w_sum = np.zeros((19, 4, 64, 120, 120))
      for i in np.arange(2*j,2*j+2):
         w_sum += 1./from_rms_split[i]**2.
         to_map_split[j] += (1./from_rms_split[i]**2.)*from_map_split[i]
      to_map_split[j] = to_map_split[j]/w_sum
      to_rms_split[j] = w_sum**(-0.5)  
      
   return to_map_split, to_rms_split


#n_splits_max has to be 2**N, where N is an integer - number of maps
def create_h5_with_jk(x,y,z, x_deg, y_deg, freq, signal_map, n_splits_max, N, date):
   map_split_highest, rms_split_highest, weights_split_highest = create_highest_split_map(x,y,z,signal_map,n_splits_max)
   data_map, rms_map, data_beam_map, rms_beam_map = coadd_splits_to_whole_map(n_splits_max,map_split_highest, rms_split_highest, weights_split_highest)
   n_splits_collection = 2**np.arange(N+1)
   n_splits_collection = n_splits_collection[1:] #get rid of 1st element, which is 1
   n_splits_collection = np.flip(n_splits_collection)
   for i in range(len(n_splits_collection)):
      n_splits = n_splits_collection[i]
      output_name = date + '_%stest_%ssplits.h5' %(1, n_splits)    
      print ('Creating the map ' + output_name)
      if n_splits == n_splits_max:
         map_split = map_split_highest
         rms_split = rms_split_highest
      else:
         map_split, rms_split = coadd_splits_to_splits(n_splits_collection[i-1], n_splits, map_split, rms_split)
      f = h5py.File(output_name, 'w')
      f.create_dataset('rms', data=rms_map)
      f.create_dataset('map', data=data_map)
      f.create_dataset('rms_coadd', data=rms_beam_map) #previously called rms_beam
      f.create_dataset('map_coadd', data=data_beam_map) #previously called map_beam
      f.create_dataset('x', data=x_deg)
      f.create_dataset('y', data=y_deg)
      f.create_dataset('freq', data=freq)
      f.create_dataset('/jackknives/map_sim', data=map_split)
      f.create_dataset('/jackknives/rms_sim', data=rms_split)
      f.close()


      
#create an output file for a regular map, without jack knifes
def create_h5(x,y,z, x_deg, y_deg, freq, output_name, signal_map, real_rms=False):
   no_of_feeds = 19
   map_shape = (19, 4, 64, 120, 120)
   map_beam_shape = (4, 64, 120, 120)
   data_map = np.zeros(map_shape)
   data_beam_map = np.zeros(map_beam_shape) #map beam is the noise weighted mean of the feed maps (sum of weights*data_map of each feed divided by w_sum)
   rms_map = np.zeros(map_shape)
   rms_beam_map = np.zeros(map_beam_shape) #sum of weights*rms_map of each feed divided by w_sum
   w_sum = np.zeros(map_beam_shape) #sum of weights of each feed

   
   for i in range(no_of_feeds):
      if real_rms == False:
         output_map_single_feed, rms_map_single_feed, signal_map_single_feed, weights_single_feed = create_output_map(x,y,z, signal_map)
         output_map_single_feed = np.reshape(output_map_single_feed,map_beam_shape)
         rms_map_single_feed = np.reshape(rms_map_single_feed,map_beam_shape)
         weights_single_feed = np.reshape(weights_single_feed, map_beam_shape)
      if real_rms == True:
         muK2K = 1e-6 #micro Kelvins to Kelvins
         signal_map_single_feed = signal_map.transpose(2, 0, 1)*muK2K
         signal_map_single_feed = np.reshape(signal_map_single_feed,map_beam_shape)
         with h5py.File('co7_011989_good_map.h5', mode="r") as my_file:
           rms_map_single_feed = np.array(my_file['rms'][i])
         noise_map_single_feed = np.random.randn(*rms_map_single_feed.shape) * rms_map_single_feed
         mask = np.zeros_like(rms_map_single_feed)
         mask[(rms_map_single_feed != 0.0)] = 1.0
         where = (mask == 1.0) #mask is 1 when we have something
         weights_single_feed = np.zeros_like(rms_map_single_feed)
         weights_single_feed[where] = 1./rms_map_single_feed[where] ** 2
         output_map_single_feed = signal_map_single_feed + noise_map_single_feed
        
      data_map[i] = output_map_single_feed
      rms_map[i] = rms_map_single_feed
      w_sum += weights_single_feed
      data_beam_map += weights_single_feed*output_map_single_feed
   data_beam_map = data_beam_map/w_sum
   if real_rms == False:
      rms_beam_map = w_sum**(-0.5)
   if real_rms == True:
      with h5py.File('co7_011989_good_map.h5', mode="r") as my_file:
         rms_beam_map = np.array(my_file['rms_beam'][:])
   f = h5py.File(output_name, 'w')
   f.create_dataset('rms', data=rms_map)
   f.create_dataset('map', data=data_map)
   f.create_dataset('rms_coadd', data=rms_beam_map) #previously called rms_beam
   f.create_dataset('map_coadd', data=data_beam_map) #previously called map_beam
   f.create_dataset('x', data=x_deg)
   f.create_dataset('y', data=y_deg)
   f.create_dataset('freq', data=freq)

   f.close()

freq, x_deg, y_deg = read_from_a_real_map('co7_011989_good_map.h5') #the same ones go to the output h5 file
x,y,z = x_y_freq_to_Mpc(x_deg,y_deg,freq)
signal_map, Voxel_volume = create_map_3d(PS_function.PS_f, x, y, z) #do this only once to have the same singal for each map
#print x,y,z
#print signal_map.flatten().std()**2*Voxel_volume #3688010.7646109615

#sys.exit()

#signal_map = np.zeros_like(signal_map)


n = len(sys.argv)
if n < 2:
    print('Missing number of maps to generate or/and number of splits!')
    print('Example: python create_map_h5_new.py N 2**N/0 <- to create N maps with max number of 2**N/no splits')
    sys.exit(1)


date = '30sept'
N = int(sys.argv[1]) #number of maps
if int(sys.argv[2]) != 0: #create maps with jk splits
   n_splits_max = int(sys.argv[2]) 
   create_h5_with_jk(x,y,z, x_deg, y_deg, freq, signal_map, n_splits_max, N, date)

if int(sys.argv[2]) == 0: #create maps with no jk splits
   names = []
   for i in range(N):
      output_name = date + '_%stest.h5' %(i+1) 
      print ('Creating the map ' + output_name)
      create_h5(x,y,z,x_deg,y_deg,freq,output_name, signal_map, False)
      names.append(output_name)

   print ('produced maps: ', names) #print this to have ready argument for my_script
