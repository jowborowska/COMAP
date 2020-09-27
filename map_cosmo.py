
import numpy as np
import h5py
import tools
import sys

#CONVERTS TO RECTANGULAR GRID IN COMOVING COORDINATES, get units right, store info about the map, pointings to 3D grid, build around one voxel in the middle
class MapCosmo(): 
    def __init__(self, mappath, feed = None, jk = None, split_no = None):
        self.feed = feed
        self.interpret_mapname(mappath)
        
        with h5py.File(mappath, mode="r") as my_file:
            self.x = np.array(my_file['x'][:]) #these x, y are are bin centers from mapmaker
            self.y = np.array(my_file['y'][:])
            if feed is not None:
                if split_no is not None:
                   self.map = np.array(my_file['/jackknives/map_' + jk][split_no,feed-1])
                   self.rms = np.array(my_file['/jackknives/rms_' + jk][split_no,feed-1])
                if split_no==None:
                   self.map = np.array(my_file['map'][feed-1])
                   self.rms = np.array(my_file['rms'][feed-1])
                
            else: #create or read map_coadd - all the feeds 'added' together
                
                if jk == 'dayn' or jk == 'half' or jk == 'sim':
                   print ('Creating coadded feed map for the split.')
                   self.map, self.rms = self.coadd_feed_maps(my_file,split_no,jk)
                if jk == 'odde' or jk == 'sdlb':
                   self.map = np.array(my_file['/jackknives/map_' + jk][split_no]) 
                   self.rms = np.array(my_file['/jackknives/map_' + jk][split_no])   
                if jk == None: 
                   self.map = np.array(my_file['map_coadd'][:]) 
                   self.rms = np.array(my_file['rms_coadd'][:])
        
        h = 0.7
        deg2mpc = 76.22 / h  # at redshift 2.9
        dz2mpc = 699.62 / h # redshift 2.4 to 3.4
        K2muK = 1e6
        z_mid = 2.9
        dnu = 32.2e-3  # GHz
        nu_rest = 115  # GHz
        dz = (1 + z_mid) ** 2 * dnu / nu_rest  # conversion 
        n_f = 256  # 64 * 4
        redshift = np.linspace(z_mid - n_f/2*dz, z_mid + n_f/2*dz, n_f + 1)


        self.map = self.map.transpose(3, 2, 0, 1) * K2muK
        self.rms = self.rms.transpose(3, 2, 0, 1) * K2muK

        sh = self.map.shape
        self.map = self.map.reshape((sh[0], sh[1], sh[2] * sh[3])) #go from 4 sidebands to 1 frequency axis
        self.rms = self.rms.reshape((sh[0], sh[1], sh[2] * sh[3]))
        self.mask = np.zeros_like(self.rms)
        self.mask[(self.rms != 0.0)] = 1.0
        where = (self.mask == 1.0) #mask is 1 when we have something
        self.w = np.zeros_like(self.rms)
        self.w[where] = 1 / self.rms[where] ** 2 #weight noise (->pseudo PS), weights outside mask are 0

        meandec = np.mean(self.y)
        self.x = self.x * deg2mpc * np.cos(meandec * np.pi / 180) #account for the effect that 'circles of latitude' around poles are smaller than equator
        self.y = self.y * deg2mpc 
        self.z = tools.edge2cent(redshift * dz2mpc)
        
        self.nz = len(self.z)
        self.nx = len(self.x)
        self.ny = len(self.y)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        self.voxel_volume = self.dx * self.dy * self.dz  # voxel volume in Mpc^3

    def coadd_feed_maps(self,map_file, which_split,jk): 
        map_single_feed = np.array(map_file['/jackknives/map_' + jk][which_split,0])
        my_map = np.zeros_like(map_single_feed)
        my_rms = np.zeros_like(map_single_feed)
        weight_sum = np.zeros_like(map_single_feed)
        for i in range(19):
            map_single_feed = np.array(map_file['/jackknives/map_'+jk][which_split,i])
            rms_single_feed = np.array(map_file['/jackknives/rms_'+jk][which_split,i])
            mask = np.zeros_like(rms_single_feed)
            mask[(rms_single_feed != 0.0)] = 1.0
            where = (mask == 1.0) 
            weight_single_feed = np.zeros_like(rms_single_feed)
            weight_single_feed[where] = 1 / rms_single_feed[where] ** 2 
            my_map += map_single_feed*weight_single_feed
            weight_sum += weight_single_feed
        mask2 =  np.zeros_like(weight_sum)
        mask2[(weight_sum != 0.0)] = 1.0
        where2 = (mask2 == 1.0)
        my_map[where2] = my_map[where2]/weight_sum[where2]
        my_rms[where2] = weight_sum[where2]**(-0.5)
        return my_map, my_rms

    def interpret_mapname(self, mappath):
        self.mappath = mappath
        mapname = mappath.rpartition('/')[-1]
        mapname = ''.join(mapname.rpartition('.')[:-2])

        parts = mapname.rpartition('_')
        try:
            self.field = parts[0]
            self.map_string = ''.join(parts[2:])
            if not self.field == '':
                self.save_string = '_' + self.field + '_' + self.map_string
            else:
                self.save_string = '_' + self.map_string
        except:
            print('Unable to find field or map_string')
            self.field = ''
            self.map_string = ''
            self.save_string = ''
        
        if self.feed is not None:
            self.save_string = self.save_string + '_%02i' % self.feed
