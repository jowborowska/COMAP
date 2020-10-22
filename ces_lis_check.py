import h5py
import numpy as np

co7_ces = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_map_complete_night_ces.h5', 'r')
co7_liss = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_map_complete_night_liss.h5', 'r')
difference = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_night_ces_liss_subtracted.h5', 'r')

difference_map = difference['/jackknives/map_dayn'][0]

ces = co7_ces['/jackknives/map_dayn'][0]

liss = co7_liss['/jackknives/map_dayn'][0]

print ('ces-lis: ')
print (ces == liss)
print ('difference: ')
print (sum(difference_map))
