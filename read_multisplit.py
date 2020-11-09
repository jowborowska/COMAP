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
map_split = jackknives['map_split']
#map_split = np.array(map_split)
print (map_split)
