import torch
import numpy as np
import matplotlib.pyplot as plt

psum = torch.load('psum45.pt')
# print(psum.shape)
psum = psum.flatten().numpy()

print(psum.shape, 'max:', max(psum), 'min:', min(psum))

bins = np.arange(-64, 65, 2)

#plt.hist(psum, bins)
#plt.show()


#x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
#num_bins = 5
n, bins, patches = plt.hist(psum, 64, facecolor='blue', alpha=0.5)
plt.show()
