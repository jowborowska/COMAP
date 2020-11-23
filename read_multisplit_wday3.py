import h5py
import numpy as np

mapname = 'co7_map_complete_fullday.h5'
field = 'co7'
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
map_split = np.array(jackknives['map_split'][:]) #shape (8, 19, 4, 64, 120, 120)
rms_split = np.array(jackknives['rms_split'][:]) #shape (8, 19, 4, 64, 120, 120)
shp = map_split.shape
map_split = map_split.reshape((2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5])) #cesc, snup, elev, feed, sideband, freq, x, y
rms_split = rms_split.reshape((2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5]))


mapnames_created = [] 
def create_output_map(cesc, snup, field, map_out, rms_out):
    #create the name
    part0 = field + '_elmap_' #cause I write El split as dayn
    my_map = map_out[cesc,snup,:,:,:,:,:,:]
    my_rms = rms_out[cesc,snup,:,:,:,:,:,:]
    if cesc == 0:
       part2 = 'liss.h5' #this is liss, I fixed it for this version
    if cesc == 1:
       part2 = 'ces.h5' 
    if snup == 0:
       part1 = 'night_'
    if snup == 1:
       part1 = 'day_'
    
    new_mapname = part0 + part1 + part2
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




create_output_map(0,1,field, map_split, rms_split)
create_output_map(1,1,field, map_split, rms_split)

create_output_map(0,0,field, map_split, rms_split)
create_output_map(1,0,field, map_split, rms_split)



print ('All the maps created: ', mapnames_created)

'''
All the maps created:  ['co6_elmap_day_liss.h5', 'co6_elmap_day_ces.h5', 'co6_elmap_night_liss.h5', 'co6_elmap_night_ces.h5']
'''


