# This code compute dihedral of the trajectory and compute density distribution.

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib
plt.switch_backend('agg')
from matplotlib import cm

from torch.utils.data import Dataset, DataLoader
import math


import os
import shutil
if not os.path.exists('Dists'):
    os.makedirs('Dists')
# else:
# 	shutil.rmtree('Dists') # remove everything in the folder, I don't have to use this.
# 	os.makedirs('Dists')

# import sys
# sys.stdout = open('output.txt','a')
# print("Compute_torsion:\n")

# ny = 100

# for i in range(ny):
# 	filename = 'Simtraj/Simtraj_'+str(i+1)+'.txt'
# 	data_temp = np.loadtxt(filename) # there are 1,000,000 frames
# 	data_temp = np.float32(data_temp)	# change to float32, and transfer to tensor
# 	if(i == 0):
# 		data = data_temp
# 	else:
# 		data = np.concatenate((data,data_temp),axis = 0)
# 	print('input Simtraj:', i)


def Compute_dist(U): # U is a n by 15 matrix, n exmaples, each example is a 15 dimensional vector
	n = len(U)

	P = U.reshape(n,10,3)	#10 atoms, shape (n,10,3)

	V12 = P[:,1:10,:] - P[:,0:9,:] # 9 1-2 distance, shape (n,9,3)
	V13 = P[:,2:10,:] - P[:,0:8,:] # 8	(n,8,3)
	V14 = P[:,3:10,:] - P[:,0:7,:] # 7 	(n,7,3)
	V15 = P[:,4:10,:] - P[:,0:6,:] # 6
	V16 = P[:,5:10,:] - P[:,0:5,:] # 5 
	V17 = P[:,6:10,:] - P[:,0:4,:] # 4
	V18 = P[:,7:10,:] - P[:,0:3,:] # 3
	V19 = P[:,8:10,:] - P[:,0:2,:] # 2
	V110 = P[:,9,:] - P[:,0,:]     # 1
	V110 = V110.reshape(n,1,3)
	# print(V12.shape,V13.shape,V14.shape,V15.shape,V16.shape,V17.shape,V18.shape,V19.shape,(V110.reshape(n,1,3)).shape)
	V = torch.cat((V12,V13,V14,V15,V16,V17,V18,V19,V110),dim=1) # shape ( n,45,3)

	C = torch.cross(V12[:,0:8,:],V12[:,1:9,:], dim=2) # 8 cross prdcut, plan norm shape (n,8,3)

	# N = torch.cross(C[:,1:8],V12[:,1:8,:], dim=2) # 7 plan vector, shape (n,7,3)

	# COS = torch.sum(C[:,0:7,:]*C[:,1:8,:], dim=2)/torch.norm(C[:,0:7,:],dim=2)/torch.norm(C[:,1:8,:],dim=2) # 7 cos dihedrals, shape (n,7)

	# SIN = torch.sum(C[:,0:7,:]*N[:,0:7,:], dim=2)/torch.norm(C[:,0:7,:],dim=2)/torch.norm(N[:,0:7,:],dim=2) # 7 sin dihedrals, shape (n,7)

	# Angles = torch.acos(torch.sum(V12[:,0:8,:]*V12[:,1:9,:],dim=2)/torch.norm(V12[:,0:8,:],dim=2)/torch.norm(V12[:,1:9,:],dim=2))  # 8 angles, shape (n,8)

	Dist = torch.norm(V,dim=2)  # 45 pairwise distance, shape (n,45)

	# out = torch.cat((Dist,Angles,COS,SIN),dim=1) # out put shape (n,45+8+7+7) = (n,67)
	out = Dist

	return out

for I in range(5): # 5 folds
	for J in range(10): # 2 sets initial data
		for K in range(10): 
			# for K in range(2):	# first and second 500k steps

			# data = np.loadtxt('../2_Simulation/Simtraj'+str(I+1)+'.txt') 

			filename = '../2_Simulation_new/TRAJs/Simtraj'+'_'+str(I+1)+'_'+str(J+1)+'_'+str(K+1)+'.npy'

			if not os.path.exists(filename):
				continue

			data = np.load('../2_Simulation_new/TRAJs/Simtraj'+'_'+str(I+1)+'_'+str(J+1)+'_'+str(K+1)+'.npy') 
			# data = np.load('/scratch/jw108/Chignolin_data/coordforce.npy') # there are 1,000,000 frames
			# data = data[0:1000000,0:30]
			data = np.float32(data)	

			N = len(data)
			data = torch.from_numpy(data)
			data.requires_grad = True

			Dist = np.zeros((N,45))


			n = 100
			N1 = int(N/n)
			for i in range(n):
				inputs = data[i*N1:(i+1)*N1,:]

				Dist[i*N1:(i+1)*N1,:] = Compute_dist(inputs).detach()

				if(i%1 == 0):
					print('i=',i)


			order = np.array([0,   9,  17,  24,  30,  35,  39,  42,  44,   1,  10,  18,  25,  31,  36, 40,  43,   2,  11,  19,  26,  32,  37,  41,   3,  12,  20,  27,  33,  38, 4,  13,  21,  28,  34,   5,  14,  22,  29,   6,  15,  23,   7,  16,   8])

			Dist = Dist[:,order]

			np.save('Dists/Dist'+'_'+str(I+1)+'_'+str(J+1)+'_'+str(K+1)+'.npy',Dist)





exit()



