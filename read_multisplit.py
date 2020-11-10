import h5py
import numpy as np

mapname = 'co6_map_complete_sunel_multisplit.h5' #co2_map_complete_sunel_multisplit.h5
field = 'co6'
mappath = '/mn/stornext/d16/cmbco/comap/protodir/maps/' + mapname

input_map = h5py.File(mappath, 'r')
keys_list = list(input_map.keys())

#print (keys_list)
'''
['feeds', 'freq', 'jackknives', 'map', 'map_coadd', 'mean_az', 'mean_el', 'n_x', 'n_y', 'nhit', 'nhit_coadd', 'njk', 'nside', 'nsim', 'patch_center', 'rms', 'rms_coadd', 'time', 'x', 'y']
'''
print ('Reading common parts.')
data_map = np.array(input_map['map'][:])
rms_map = np.array(input_map['rms'][:])
data_beam_map = np.array(input_map['map_coadd'][:])
rms_beam_map = np.array(input_map['rms_coadd'][:])
x = np.array(input_map['x'][:])
y = np.array(input_map['y'][:])
freq = np.array(input_map['freq'][:])

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
   
   mask2 =  np.zeros(new_map_shape)
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

mapnames_created = [] 
def create_output_map(cesc, elev, ambt, field, map_out, rms_out):
    #create the name
    part0 = field + '_map_'
    if elev == 'coadded':
       part1 = 'coadded_elev_'
       my_map = map_out[cesc,:,ambt,:,:,:,:,:]
       my_rms = rms_out[cesc,:,ambt,:,:,:,:,:]
    if ambt == 'coadded':
       part1 = 'coadded_ambt_'
       my_map = map_out[cesc,:,elev,:,:,:,:,:]
       my_rms = rms_out[cesc,:,elev,:,:,:,:,:]
    if cesc == 0:
       part2 = 'ces.h5' 
    if cesc == 1:
       part2 = 'liss.h5' 
    if ambt == 0:
       part3 = 'lower_ambt_'
    if ambt == 1:
       part3 = 'upper_ambt_'
    if elev == 0:
       part3 = 'lower_elev_'
    if elev == 1:
       part3 = 'upper_elev_'
    new_mapname = part0 + part1 + part3 + part2
    print ('Creating HDF5 file for the map ' + new_mapname + '.')
    mapnames_created.append(new_mapname)

    f = h5py.File(new_mapname, 'w')
    f.create_dataset('rms', data=rms_map)
    f.create_dataset('map', data=data_map)
    f.create_dataset('rms_coadd', data=rms_beam_map) 
    f.create_dataset('map_coadd', data=data_beam_map) 
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('freq', data=freq)
    f.create_dataset('/jackknives/map_dayn', data=my_map)
    f.create_dataset('/jackknives/rms_dayn', data=my_rms)
    f.close()

#for ces, upper elev, coadded ambt, co6 field
create_output_map(0,1,'coadded',field, map_split_coadded_ambt, rms_split_coadded_ambt)

#for liss, upper elev, coadded ambt, co6 field
create_output_map(1,1,'coadded',field, map_split_coadded_ambt, rms_split_coadded_ambt)

#for ces, lower elev, coadded ambt, co6 field
create_output_map(0,0,'coadded',field, map_split_coadded_ambt, rms_split_coadded_ambt)

#for liss, lower elev, coadded ambt, co6 field
create_output_map(1,0,'coadded',field, map_split_coadded_ambt, rms_split_coadded_ambt)

#for ces, coadded elev, lower ambt, co6 field
create_output_map(0,'coadded', 0,field, map_split_coadded_elev, rms_split_coadded_elev)

#for liss, coadded elev, lower ambt, co6 field
create_output_map(1,'coadded', 0,field, map_split_coadded_elev, rms_split_coadded_elev)

#for ces, coadded elev, upper ambt, co6 field
create_output_map(0,'coadded', 1, field, map_split_coadded_elev, rms_split_coadded_elev)

#for liss, coadded elev, upper ambt, co6 field
create_output_map(1,'coadded', 1, field, map_split_coadded_elev, rms_split_coadded_elev)

print ('All the maps created: ', mapnames_created)

