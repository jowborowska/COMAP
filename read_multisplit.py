import h5py

mapname = 'co6_map_complete_sunel_multisplit.h5' #co2_map_complete_sunel_multisplit.h5
mappath = '/mn/stornext/d16/cmbco/comap/protodir/maps/' + mapname

input_map = h5py.File(mappath, 'r')
keys_list = list(input_map.keys())

print (keys_list)

