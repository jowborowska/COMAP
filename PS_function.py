import numpy as np 

#P(k) = k**-3, This one can be modified - P(k) defines the distribution, a temperature map is one realization from this distribution
def PS_f(k_array):
   shape = k_array.shape
   k_array = k_array.flatten()
   PS_array = np.zeros(len(k_array))
   for i in range(len(k_array)):
      if k_array[i] != 0.: 
          PS_array[i] = 2.*k_array[i]**(-3.)
      else:
         PS_array[i] = 0.
   PS_array = np.reshape(PS_array,shape)
   return PS_array
