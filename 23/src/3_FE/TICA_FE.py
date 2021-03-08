# This code compute dihedral of the trajectory and compute density distribution.

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np 

import pyemma
from glob import glob

import matplotlib.pyplot as plt 
import matplotlib
plt.switch_backend('agg')
from matplotlib import cm

from torch.utils.data import Dataset, DataLoader
import math

import os
import shutil
# import sys
# sys.stdout = open('output.txt','a')
# print("Compute_torsion:\n")

def plot_FE(Z):

	H, xgrid, ygrid = np.histogram2d(Z[:,0],Z[:,1],bins = 100)

	H = H.transpose()

	fig = plt.figure(figsize=(13,6))
	ax = fig.gca()
	surf = plt.imshow(-np.log(H), cmap = cm.jet, interpolation='nearest', origin='low', extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]],aspect=0.8)
	ax.set_xlabel(r'$\phi$',fontsize=20)
	ax.set_ylabel(r'$\psi$',fontsize=20)
	plt.xlim((-30,5))
	plt.ylim((-25,15))
	clb = fig.colorbar(surf, shrink=1.0, aspect=10)
	plt.clim(-8.4,0)
	# clb.ax.set_title(r'$\beta$F',fontsize=20)
	plt.savefig('FreeEnergy1_avg.pdf',dpi=100)




portion = ['a','b']
ZZ_concat = []

for i in range(5):
	if(i!=2):
		continue
	for j in range(10):
		for k in range(10):

			# if(i!=1):
			# 	continue

			filename = 'Dists/Dist'+'_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'.npy'

			if not os.path.exists(filename):
				continue

			Dist = np.load('Dists/Dist'+'_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'.npy')

			tica_object = pyemma.load('tica_object.pyemma')

			ZZ = tica_object.transform(Dist)

			if(i==2 and j==0 and k==0):
				ZZ_concat = ZZ
			else:
				ZZ_concat = np.concatenate((ZZ_concat,ZZ))

			# np.save('ZZ'+str(i+1)+str(j+1)+portion[k]+'.npy',ZZ)

			# pyemma.plots.plot_free_energy(ZZall[:,0], ZZall[:,1])

			# plt.figure(figsize=(7, 3))
			# pyemma.plots.plot_free_energy(ZZ[:,0], ZZ[:,1],cmap='jet')
			# # pyemma.plots.plot_free_energy(Torsion[:,0],Torsion[:,1], nbins=64, cmap='jet')
			# # fig.tight_layout()
			# plt.xlim((-27,0))
			# plt.ylim((-19,15))
			# # plt.ylim((-21,31))

			# plt.savefig('FreeEnergy1'+str(i+1)+str(j+1)+portion[k]+'.png',dpi=100)



			# plt.figure(figsize=(7, 3))
			# pyemma.plots.plot_free_energy(ZZ[:,0], ZZ[:,2],cmap='jet')
			# # pyemma.plots.plot_free_energy(Torsion[:,0],Torsion[:,1], nbins=64, cmap='jet')
			# # fig.tight_layout()
			# plt.xlim((-27,0))
			# # plt.ylim((-19,15))
			# plt.ylim((-21,31))

			# plt.savefig('FreeEnergy2'+str(i+1)+str(j+1)+portion[k]+'.png',dpi=100)



			# fig = plt.figure(figsize=(7, 3))
			# ax = fig.gca()
			# plt.plot(np.arange(0,10000),ZZ[0:10000,0], c = 'blue', label = 'TICA1_t',linewidth=0.5, zorder=1) 
			# ax.set_xlabel('time steps',fontsize=20)
			# ax.set_ylabel('TICA1',fontsize=20)
			# plt.savefig('TICA1_t'+str(i+1)+str(j+1)+portion[k]+'.png',dpi=100)


print(ZZ_concat.shape)
# ZZ_avg = ZZ_concat[np.arange(0,1500000,2),:]
sample = np.random.choice(ZZ_concat.shape[0], 1000000, replace=True)
ZZ_avg = ZZ_concat[sample,:]
print(ZZ_avg.shape)
np.save('ZZ_avg.npy',ZZ_avg)
plot_FE(ZZ_avg)


ZZ_AA = np.load('ZZ_AA.npy')

def compute_1DFE(Z):
	ZZ_AA = np.load('ZZ_AA.npy')
	Min = np.min(ZZ_AA[:,0])
	Max = np.max(ZZ_AA[:,0])
	step = 0.02

	grid = np.arange(Min,Max,step)

	count, bins = np.histogram(Z[:,0], grid)
	N = len(ZZ_AA[:,0])

	prob_density = count/N/step
	FE = -np.log(prob_density)
	FE = FE - np.min(FE)

	return FE, bins


def plot_1DFE(Z1,Z2):
	FE1, bins1 = compute_1DFE(Z1)
	FE2, bins2 = compute_1DFE(Z2)

	np.save('FETIC1.npy',np.concatenate((bins1[:-1].reshape(-1,1),FE1.reshape(-1,1)),axis=1))

	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()

	plt.plot(bins1[:-1],FE1, c = 'blue', label = '3BD_dihedral')
	plt.plot(bins2[:-1],FE2, c = 'red', label = 'AA')
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc = 'upper center',fontsize = 20)

	plt.savefig('1D_FE.png',dpi=100)


plot_1DFE(ZZ_avg,ZZ_AA)
# print(ZZ_avg.shape, ZZ_AA.shape)

# Min = np.min(ZZ_AA[:,0])
# Max = np.max(ZZ_AA[:,0])

# step = 0.02

# grid = np.arange(Min,Max,step)

# count, bins = np.histogram(ZZ_avg[:,0], grid)
# count1, bins1 = np.histogram(ZZ_AA[:,0], grid)

# N = len(ZZ_AA[:,0])

# prob_density = count/N/step
# prob_density1 = count1/N/step

# fig = plt.figure(figsize=(8, 6))
# ax = fig.gca()

# FE = -np.log(prob_density)
# FE1 = -np.log(prob_density1)

# plt.plot(bins[:-1],FE, c = 'blue', label = 'CG')
# plt.plot(bins[:-1],FE1, c = 'red', label = 'AA')

# plt.legend(loc = 'upper right')





exit()


Dist1 = np.load('Dist1.npy')
Dist2 = np.load('Dist2.npy')
Dist3 = np.load('Dist3.npy')
Dist4 = np.load('Dist4.npy')
Dist5 = np.load('Dist5.npy')
Dist = np.concatenate((Dist1,Dist2,Dist3,Dist4,Dist5),axis = 0)
select = np.arange(0,5000000,5)
Dist = Dist[select,:]

tica_object = pyemma.load('tica_object.pyemma')

ZZ = tica_object.transform(Dist)

np.save('ZZ_avg.npy',ZZ)

# pyemma.plots.plot_free_energy(ZZall[:,0], ZZall[:,1])

plt.figure(figsize=(7, 3))
pyemma.plots.plot_free_energy(ZZ[:,0], ZZ[:,1],cmap='jet')
# pyemma.plots.plot_free_energy(Torsion[:,0],Torsion[:,1], nbins=64, cmap='jet')
# fig.tight_layout()
plt.xlim((-27,0))
plt.ylim((-19,15))
# plt.ylim((-21,31))

plt.savefig('FreeEnergy1_avg.png',dpi=100)



plt.figure(figsize=(7, 3))
pyemma.plots.plot_free_energy(ZZ[:,0], ZZ[:,2],cmap='jet')
# pyemma.plots.plot_free_energy(Torsion[:,0],Torsion[:,1], nbins=64, cmap='jet')
# fig.tight_layout()
plt.xlim((-27,0))
# plt.ylim((-19,15))
plt.ylim((-21,31))

plt.savefig('FreeEnergy2_avg.png',dpi=100)
