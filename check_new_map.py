import h5py

f = h5py.File('co2_map_complete.h5', 'r')

keys_list = list(f.keys())

print keys_list
'''
[u'feeds', u'freq', u'jackknives', u'map', u'map_coadd', u'mean_az', u'mean_el', u'n_x', u'n_y', u'nhit', u'nhit_coadd', u'njk', u'nside', u'nsim', u'patch_center', u'rms', u'rms_coadd', u'time', u'x', u'y']
'''

jackknives = f['jackknives']
print jackknives.keys()
'''
[u'jk_def', u'jk_feedmap', u'map_dayn', u'map_half', u'map_odde', u'map_sdlb', u'nhit_dayn', u'nhit_half', u'nhit_odde', u'nhit_sdlb', u'rms_dayn', u'rms_half', u'rms_odde', u'rms_sdlb']
'''

half_split = jackknives['map_half']
print half_split.shape #(2, 19, 4, 64, 120, 120)
