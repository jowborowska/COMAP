#Check the structure of map h5 file - both for a real map and then my generated map_file
#things needed in map_cosmo: x, y, map, rms, map_beam, rms_beam

import h5py

#f = h5py.File('co7_011989_good_map.h5', 'r')
f = h5py.File('my_map.h5', 'r')
keys_list = list(f.keys()) #[u'feeds', u'freq', u'map', u'map_beam', u'mean_az', u'mean_el', u'n_x', u'n_y', u'nhit', u'nhit_beam', u'njk', u'nside', u'nsim', u'patch_center', u'rms', u'rms_beam', u'time', u'x', u'y']

for element in keys_list:
   f_element = f[element]
   print element, f_element.shape, f_element.dtype
'''
#for the real map:
feeds (0,) int32
freq (4, 64) float64
map (19, 4, 64, 120, 120) float32
map_beam (4, 64, 120, 120) float32
mean_az () float64
mean_el () float64
n_x () int32
n_y () int32
nhit (19, 4, 64, 120, 120) int32
nhit_beam (4, 64, 120, 120) int32
njk () int32
nside () int32
nsim () int32
patch_center (2,) float64
rms (19, 4, 64, 120, 120) float32
rms_beam (4, 64, 120, 120) float32
time (2,) float64
x (120,) float64
y (120,) float64
'''
my_map = f['map']
print my_map




