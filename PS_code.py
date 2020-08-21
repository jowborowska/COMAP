import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D

#3D map of brightness temperature
bright_temp = np.load('map_example.npy') #[Kelvin]
bright_temp = bright_temp*1e6 #[micro K]

#voxel edge arrays in the x, y and z directions [comoving Mpc]
x_direction = np.load('x.npy')
y_direction = np.load('y.npy')
z_direction = np.load('z.npy')

#number of voxels in each direction, 20 20 32
n_x = x_direction.size - 1
n_y = y_direction.size - 1
n_z = z_direction.size - 1

#voxel size in each direction [comoving Mpc], 5.0 5.0 6.25
Delta_x = x_direction[1] - x_direction[0]
Delta_y = y_direction[1] - y_direction[0]
Delta_z = z_direction[1] - z_direction[0]


def calculate_sperically_averaged_PS(bright_temp, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z):
   #Volume of this single voxel [comoving Mpc^3], 156.25
   V_vox = Delta_x*Delta_y*Delta_z

   #physical wave number in each direction
   k_x = 2*np.pi*np.fft.fftfreq(n_x,Delta_x)
   k_y = 2*np.pi*np.fft.fftfreq(n_y,Delta_y)
   k_z = 2*np.pi*np.fft.fftfreq(n_z,Delta_z)

   f_k = np.fft.fftn(bright_temp)
   PS = np.abs(f_k)**2
   k_array = np.zeros(bright_temp.shape)
   
   #fill k_array with abs(k) values
   index = 0
   for i in range(n_x):
      for j in range(n_y):
        for k in range(n_z):
            k_array[i][j][k]= np.linalg.norm(np.array([k_x[i],k_y[j], k_z[k]]))
            index += 1

   #flatten PS array and k array 
   PS = PS.flatten()
   k_array = k_array.flatten()

   #co-ad PS values corresponding to the same k-values, use 25 k-intervals/bins
   k_histogram, k_bins = np.histogram(k_array, bins=25)
   PS_binned = np.zeros(len(k_histogram))
   for i in range(len(PS_binned)):
      for j in range(len(PS)):  #check all values if they are in this bin
         if k_array[j] > k_bins[i] and k_array[j] < k_bins[i+1]: #if we are in the correct k-bin, add the corresponding PS values together
            PS_binned[i] += PS[j]

   #print k_histogram #25
   #print k_bins #26

   #compute spherically averaged PS
   PS_binned_averaged = PS_binned/k_histogram
   PS_sa = PS_binned_averaged*V_vox/(n_x*n_y*n_z)
   return k_bins, k_histogram, PS_sa

k_bins, k_histogram, PS_sa = calculate_sperically_averaged_PS(bright_temp, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z)

#fit a curve - find parameters for power law
def fit_power_law(PS, k_histogram, k_bins):

   def target_func(x, a, b):
          return  (x**b)*a 

   InitialGuess = np.array([4.73, -3.29])
   variance = PS**2/k_histogram
   popt, pcov = opt.curve_fit(target_func, k_bins[1:], PS, p0=InitialGuess, sigma=variance)
   return popt, pcov, target_func

popt, pcov, target_func = fit_power_law(PS_sa, k_histogram, k_bins)

def create_map(PS_sa, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z, parameters, target_func):
   V_vox = Delta_x*Delta_y*Delta_z
   new_map = np.zeros((n_x,n_y,n_z))
   #PS = PS_sa*n_x*n_y*n_z/V_vox #f_k squared and averaged
   PS = PS_sa
   #TRY TO UN-BIN f_k's
   k_x = 2*np.pi*np.fft.fftfreq(n_x,Delta_x)
   k_y = 2*np.pi*np.fft.fftfreq(n_y,Delta_y)
   k_z = 2*np.pi*np.fft.fftfreq(n_z,Delta_z)
   k_array = np.zeros(new_map.shape)
   #fill k_array with abs(k) values
   index = 0
   for i in range(n_x):
      for j in range(n_y):
        for k in range(n_z):
            k_array[i][j][k]= np.linalg.norm(np.array([k_x[i],k_y[j], k_z[k]]))
            index += 1
   k_array_flat = k_array.flatten()
   k_histogram, k_bins = np.histogram(k_array_flat, bins=25) #histogram - how many in each bin, bins - gives bin-intervals
   PS_model = np.zeros_like(new_map)
   for i in range(n_x):
      for j in range(n_y):
         for k in range(n_z):
            if k_array[i][j][k] != 0:
               PS_model[i][j][k] = target_func(k_array[i][j][k], *parameters)
   f = (np.random.randn(n_x, n_y, n_z) + 1j * np.random.randn(n_x, n_y, n_z)) / np.sqrt(2.) * np.sqrt(PS_model)
   f = f*np.sqrt(n_x*n_y*n_z/V_vox)
   x_k = np.fft.ifftn(f)
   x_k = np.real(x_k)
   return x_k*np.sqrt(2.)


a_map = create_map(PS_sa, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z,popt, target_func)
#print a_map

k_bins2, k_histogram2, PS_sa2 = calculate_sperically_averaged_PS(a_map, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z)


#Now generate more PS's from created map
def esimate_uncertainty(some_map, original_PS_sa, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z, popt, target_func, no_of_sim):
   k_bins, k_histogram, PS_sa = calculate_sperically_averaged_PS(some_map, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z) 
   no_of_bins = len(k_histogram)
   PS_array = np.zeros((no_of_sim,no_of_bins))
   PS_mean = np.zeros(no_of_bins) 
   for i in range(no_of_sim):
      some_map2= create_map(PS_sa, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z, popt, target_func)
      k_bins, k_histogram, PS_sa2 = calculate_sperically_averaged_PS(some_map2, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z)   
      PS_array[i]= PS_sa2
      PS_mean += PS_sa2
   std = np.zeros(no_of_bins)
   for j in range(no_of_bins):
      std[j] = np.std(PS_array[:,j])
   PS_mean = PS_mean/float(no_of_sim)
   return std, k_bins[1:], PS_mean
std, k_bins_std, PS_mean = esimate_uncertainty(a_map, PS_sa, n_x, n_y, n_z, Delta_x, Delta_y, Delta_z, popt, target_func, 100)



plt.figure()
plt.plot(k_bins_std, std**2, label='var =$\sigma_{PS}^2$, from sims ')
#plt.plot(k_bins_std, PS_mean*Delta_x*Delta_y*Delta_z/(n_x*n_y*n_z), label='mean PS from sims')
plt.plot(k_bins_std, 2.*target_func(k_bins[1:], *popt)**2./k_histogram, label='$2P(k)^2/n_{modes}$, analytic')
plt.xscale('log')
plt.yscale('log')
plt.title('variance in each bin estimated from 100 simulated maps', fontsize=14)
plt.xlabel('k [Mpc$^{-1}$]', fontsize=14)
plt.legend()
plt.show()

x_original = bright_temp[:,0]
y_original = bright_temp[:,1]
z_original = bright_temp[:,2]

x = a_map[:,0]
y = a_map[:,1]
z = a_map[:,2]

#for the original brightness temperature map
fig_original = plt.figure()
ax_original = fig_original.add_subplot(111, projection='3d')
ax_original.scatter(x_original,y_original,z_original)
plt.show()

#for my created map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()

plt.figure()
plt.scatter(k_bins[1:],PS_sa, s=20, color='salmon', label='P(k)')
plt.plot(k_bins[1:], target_func(k_bins[1:], *popt), color='firebrick', label='$ak^b$, a = 4.73, b = -3.29, scipy optimize with variance' )
plt.errorbar(k_bins[1:],target_func(k_bins[1:], *popt), yerr=std, fmt='none', label='$\pm \sigma$ from 100 sims')
#plt.plot(k_bins[1:], target_func(k_bins[1:],4.5, -3.2), color='violet', label='$ak^b$, a = 4.5, b = -3.2, my fit')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('P(k) [$\mu$K$^2$Mpc$^{3}$]', fontsize=14)
plt.xlabel('k [Mpc$^{-1}$]', fontsize=14)
plt.legend()
plt.show()


plt.figure()
plt.scatter(k_bins2[1:],PS_sa2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('P(k) [$\mu$K$^2$Mpc$^{3}$], from created map', fontsize=14)
plt.xlabel('k [Mpc$^{-1}$]', fontsize=14)
plt.legend()
plt.show()

