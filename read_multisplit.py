import h5py
import numpy as np

mapname = 'co6_map_complete_sunel_multisplit.h5' #co2_map_complete_sunel_multisplit.h5
mappath = '/mn/stornext/d16/cmbco/comap/protodir/maps/' + mapname

input_map = h5py.File(mappath, 'r')
keys_list = list(input_map.keys())

#print (keys_list)
'''
['feeds', 'freq', 'jackknives', 'map', 'map_coadd', 'mean_az', 'mean_el', 'n_x', 'n_y', 'nhit', 'nhit_coadd', 'njk', 'nside', 'nsim', 'patch_center', 'rms', 'rms_coadd', 'time', 'x', 'y']
'''
jackknives = input_map['jackknives']

#print (jackknives.keys())
'''
['jk_def', 'jk_feedmap', 'map_sidr', 'map_split', 'nhit_sidr', 'nhit_split', 'rms_sidr', 'rms_split']
'''
map_split = np.array(jackknives['map_split'][:]) #shape (16, 19, 4, 64, 120, 120)
rms_split = np.array(jackknives['rms_split'][:]) #shape (16, 19, 4, 64, 120, 120)
shp = map_split.shape
map_split = map_split.reshape((2,2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5])) #cesc, sune, elev, ambt, feed, sideband, freq, x, y
rms_split = rms_split.reshape((2,2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5]))

def coadd_split(old_map_split, old_rms_split, elev_or_ambt):
   new_map_shape = (2,2,2,19,4,64,120,120)
   new_map_split = np.zeros(new_map_shape)
   new_rms_split = np.zeros(new_map_shape)
   w_sum = np.zeros(new_map_shape)
   if elev_or_ambt == 'elev':
      print ('Coadding elev-split.')
      for i in range(2):
            mask = np.zeros(new_map_shape)
            mask[(old_rms_split[:,:,i,:,:,:,:,:,:] != 0.0)] = 1.0
            where = (mask == 1.0) 
            weight = np.zeros(new_map_shape)
            weight[where] = 1./old_rms_split[:,:,i,:,:,:,:,:,:][where]**2.
            w_sum += weight
            new_map_split += weight*old_map_split[:,:,i,:,:,:,:,:,:]
   if elev_or_ambt == 'ambt':
      print ('Coadding ambt-split.')
      for i in range(2):
            mask = np.zeros(new_map_shape)
            mask[(old_rms_split[:,:,:,i,:,:,:,:,:] != 0.0)] = 1.0
            where = (mask == 1.0) 
            weight = np.zeros(new_map_shape)
            weight[where] = 1./old_rms_split[:,:,:,i,:,:,:,:,:][where]**2.
            w_sum += weight
            new_map_split += weight*old_map_split[:,:,:,i,:,:,:,:,:]
   
   mask2 =  np.zeros_like(new_map_shape)
   mask2[(w_sum != 0.0)] = 1.0
   where2 = (mask2 == 1.0)
   new_map_split[where2] = new_map_split[where2]/w_sum[where2]
   new_rms_split[where2] = w_sum[where2]**(-0.5)  
   return new_map_split, new_rms_split

#we want to coadd two elevation-splits and look at ambient, as well as add two ambient-splits and look at elevation   
map_split_coadded_elev, rms_split_coadded_elev = coadd_split(map_split, rms_split, 'elev') #cesc, sune, ambt, feed, sideband, freq, x, y
map_split_coadded_ambt, rms_split_coadded_ambt = coadd_split(map_split, rms_split, 'ambt') #cesc, sune, elev, feed, sideband, freq, x, y

#now, fro three first indices 0 means first half of the data, 1 means second half of the data, with respect to that feature
#for coadded elev we would have 4 final maps: lower cesc - upper ambt [0,:,1], upper cesc - lower ambt [1,:,0], lower cesc - lower ambt [0,:,0], upper cesc - upper ambient [1,:,1], and similarly for coadded ambt -> the whole program would give us 8 maps with two sune-splits






