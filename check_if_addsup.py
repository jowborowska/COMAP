import h5py
import numpy as np

early_split = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_map_complete_night_earlysid.h5', 'r')
late_split = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_map_complete_night_latesid.h5', 'r')
complete = h5py.File('/mn/stornext/d16/cmbco/comap/protodir/maps/co7_map_complete_night.h5', 'r')

complete_map = complete['map']
complete_rms = complete['rms']

split1 = early_split['/jackknives/map_sidr'][0]
split2 = early_split['/jackknives/map_sidr'][1]
split3 = late_split['/jackknives/map_sidr'][0]
split4 = late_split['/jackknives/map_sidr'][1]

split1_rms = early_split['/jackknives/rms_sidr'][0]
split2_rms = early_split['/jackknives/rms_sidr'][1]
split3_rms = late_split['/jackknives/rms_sidr'][0]
split4_rms = late_split['/jackknives/rms_sidr'][1]

splits = np.array([split1,split2,split3,split4])
rms = np.array([split1_rms,split2_rms,split3_rms,split4_rms])

map_coadded = np.zeros((19, 4, 64, 120, 120))
w_sum = np.zeros((19, 4, 64, 120, 120))

for i in range(4):
   w = 1./rms[i]**2.
   w_sum +=w
   map_coadded += w*splits[i]
map_coadded = map_coadded/w_sum
rms_coadded = w_sum**(-0.5)

print ('MAP: ')
print (map_coadded == complete_map)
print ('RMS: ')
print (rms_coadded == complete_rms)
