# This code is to use learnt NN to runsimulation over 1D x-axis, once trajectory outside of the bounday, restart
# the previous point, this could give the correct distribution within the bounday.

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import cm

import os
import shutil
import time

# import sys
# sys.stdout = open('output1.txt','a')
# print("Simulation:\n")

import argparse
# initiate the parser
parser = argparse.ArgumentParser()
# add long and short argument
parser.add_argument("--modelID", "-m", help="select cross validation inde")
parser.add_argument("--modelID2", "-m2", help="select cross validation inde")

args = parser.parse_args()

modelID = int(args.modelID)
modelID2 = int(args.modelID2)

beta = 0.01
beta = 1.6775


Power = 6
repelconst = 5.5**Power

# 2BD
# Depth = 5
# Width = 30
# Power = 10
# repelconst = 4.0**Power
# Lambda = 3.0

# 3BD
# Depth3 = 4
# Width3 = 40
# Lambda3 = 4.0


# bond and angle means and constants
BondConst = np.loadtxt('../BondConst.txt') 
BondConst = np.float32(BondConst)
BondConst = BondConst.transpose()
# BondConst = BondConst[:,[0,4,7,9,10,11,12]]
BondConst = torch.from_numpy(BondConst)

Zscore_dist = np.loadtxt('../Zscore_dist.txt') 
Zscore_dist = np.float32(Zscore_dist)
Zscore_dist = Zscore_dist.transpose()
Zscore_dist = torch.from_numpy(Zscore_dist)


Zscore_cos = np.loadtxt('../Zscore_cos.txt') 
Zscore_cos = np.float32(Zscore_cos)
Zscore_cos = Zscore_cos.transpose()
Zscore_cos = torch.from_numpy(Zscore_cos)

Zscore_sin = np.loadtxt('../Zscore_sin.txt') 
Zscore_sin = np.float32(Zscore_sin)
Zscore_sin = Zscore_sin.transpose()
Zscore_sin = torch.from_numpy(Zscore_sin)



## Generate permutation list, from 10 to 4, there will be 210 
INDEX1_3BD = np.zeros((120,3))
it = 0
for i in np.arange(0,10):
	for j in np.arange(i+1,10):
		for k in np.arange(j+1,10):
			# for l in np.arange(k+1,10):
			INDEX1_3BD[it] = [i,j,k]
			it=it+1
			

INDEX1_4BD = np.zeros((210,4))
it = 0
for i in np.arange(0,10):
	for j in np.arange(i+1,10):
		for k in np.arange(j+1,10):
			for l in np.arange(k+1,10):
				INDEX1_4BD[it] = [i,j,k,l]
				it=it+1


# map from pairwise atom index tutor to the index in the distance vetor			
INDEX2 = {}
it = 0
for i in np.arange(1,10):
	for j in np.arange(i,10):
		INDEX2[(j-i,j)] = it
		it = it + 1


# For each 3 body, find the corresponding 3 distance index
INDEX3_3BD = np.zeros((120,3))
for i in np.arange(120):
	it = 0
	for j in np.arange(0,3):
		for k in np.arange(j+1,3):
			INDEX3_3BD[i,it] = INDEX2[(int(INDEX1_3BD[i,j]),int(INDEX1_3BD[i,k]))]
			it = it+1


INDEX3_4BD = np.zeros((210,6))
for i in np.arange(210):
	it = 0
	for j in np.arange(0,4):
		for k in np.arange(j+1,4):
			INDEX3_4BD[i,it] = INDEX2[(int(INDEX1_4BD[i,j]),int(INDEX1_4BD[i,k]))]
			it = it+1

# for each 4 body, find the corresponding 3 vector index
INDEX4 = np.zeros((210,3))
for i in np.arange(210):
	for j in np.arange(0,3):
		INDEX4[i,j] = INDEX2[(int(INDEX1_4BD[i,j]),int(INDEX1_4BD[i,j+1]))]


# Find the index of the 7 adjacent 4-body, say the original dihedrals
INDEX5 = np.zeros(7)
it = 0
itt = 0
for i in np.arange(0,10):
	for j in np.arange(i+1,10):
		for k in np.arange(j+1,10):
			for l in np.arange(k+1,10):
				if(j==i+1 and k==j+1 and l==k+1):
					INDEX5[itt] = it
					itt = itt+1
				it=it+1

# define a feature layer
class Feature_old(nn.Module): 

	def __init__(self):
		super(Feature_old, self).__init__()

	def forward(self, U): # U is a n by 15 matrix, n exmaples, each example is a 15 dimensional vector
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

		N = torch.cross(C[:,1:8],V12[:,1:8,:], dim=2) # 7 plan vector, shape (n,7,3)

		COS = torch.sum(C[:,0:7,:]*C[:,1:8,:], dim=2)/torch.norm(C[:,0:7,:],dim=2)/torch.norm(C[:,1:8,:],dim=2) # 7 cos dihedrals, shape (n,7)

		SIN = torch.sum(C[:,0:7,:]*N[:,0:7,:], dim=2)/torch.norm(C[:,0:7,:],dim=2)/torch.norm(N[:,0:7,:],dim=2) # 7 sin dihedrals, shape (n,7)

		Angles = torch.acos(torch.sum(V12[:,0:8,:]*V12[:,1:9,:],dim=2)/torch.norm(V12[:,0:8,:],dim=2)/torch.norm(V12[:,1:9,:],dim=2))  # 8 angles, shape (n,8)

		Dist = torch.norm(V,dim=2)  # 45 pairwise distance, shape (n,45)

		out = torch.cat((Dist,Angles,COS,SIN),dim=1) # out put shape (n,45+8+7+7) = (n,67)

		return Dist, COS, SIN, Angles


class Feature_new(nn.Module): 

	def __init__(self):
		super(Feature_new, self).__init__()

	def forward(self, U): # U is a n by 15 matrix, n exmaples, each example is a 15 dimensional vector
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

		C1 = torch.cross(V[:,INDEX4[:,0],:],V[:,INDEX4[:,1],:], dim=2) # 8 cross prdcut, plan norm shape (n,8,3)
		C2 = torch.cross(V[:,INDEX4[:,1],:],V[:,INDEX4[:,2],:], dim=2) # 8 cross prdcut, plan norm shape (n,8,3)

		N = torch.cross(C2[:,:,:],V[:,INDEX4[:,1],:], dim=2) # 7 plan vector, shape (n,7,3)

		dC1 = torch.norm(C1,dim=2)
		dC2 = torch.norm(C2,dim=2)

		COS = torch.sum(C1*C2, dim=2)/dC1/dC2 # 7 cos dihedrals, shape (n,7)

		SIN = torch.sum(N*C1, dim=2)/dC1/torch.norm(N[:,:,:],dim=2) # 7 sin dihedrals, shape (n,7)

		Angles = torch.acos(torch.sum(V12[:,0:8,:]*V12[:,1:9,:],dim=2)/torch.norm(V12[:,0:8,:],dim=2)/torch.norm(V12[:,1:9,:],dim=2))  # 8 angles, shape (n,8)

		Dist = torch.norm(V,dim=2)  # 45 pairwise distance, shape (n,45)

		return Dist, COS, SIN, Angles




# compute bond angle potential
class bondPotential(nn.Module): 

	def __init__(self,BondConst): # Const is a 2 by k matrix, first row is the mean, second row is the harmonic const
		super(bondPotential, self).__init__()

	def forward(self, U): # U is a n by k matrix, n exmaples, each example has its k features
		n = len(U)		
		Potential = torch.sum(BondConst[0,:]*(U-BondConst[1,:])**2, 1).reshape(n,1)/2
		return Potential

class repelPotential(nn.Module): 

	def __init__(self): # Const is a 2 by k matrix, first row is the mean, second row is the harmonic const
		super(repelPotential, self).__init__()

	def forward(self, U): # U is a n by k matrix, n exmaples, each example has its k features
		n = len(U)		
		# Potential = torch.sum(BondConst[0,:]*(U-BondConst[1,:])**2, 1).reshape(n,1)/2
		Potential = torch.sum(repelconst/U**Power,1).reshape(n,1)
		return Potential



class NLunit_Multibody(nn.Module):

	def __init__(self,ni, n,nl):
		super(NLunit_Multibody, self).__init__()

		names = ['0']*(nl+1)
		for i in np.arange(1,nl+1):
			names[i] = str(i)
			
		self.net = nn.Sequential()
		self.net.add_module('0',nn.Linear(ni,n))
		self.net.add_module('0act',torch.nn.Tanh())
		for i in np.arange(1,nl):
			self.net.add_module(names[i],nn.Linear(n,n))
			self.net.add_module(names[i]+'act',torch.nn.Tanh())

		self.net.add_module(names[nl],nn.Linear(n,1))

	def forward(self, x):

		out = self.net(x)

		return out


class Model2BD(torch.nn.Module):

	def __init__(self,Zscore_dist, Zscore_cos, Zscore_sin,BondConst,n,nl):
		super(Model2BD, self).__init__()
		# linear layers

		self.Splines = nn.ModuleList()
		for i in range(45):
			self.Splines.append(NLunit_Multibody(1,n,nl))
		for i in range(7):
			self.Splines.append(NLunit_Multibody(2,n,nl))

		# self.feature = Feature_new()
		self.bondPotential = bondPotential(BondConst)
		self.repelPotential = repelPotential()


	def forward(self, F_dist, F_cos, F_sin, F_angle): 
		# feature_dist, feature_cos, feature_sin = F
		selection_feature = np.arange(0,45)
		FNorm_dist = (F_dist[:,selection_feature]-Zscore_dist[0,selection_feature])/Zscore_dist[1,selection_feature]
		FNorm_cos = (F_cos-Zscore_cos[0,:])/Zscore_cos[1,:]
		FNorm_sin = (F_sin-Zscore_sin[0,:])/Zscore_sin[1,:]


		for i in range(45):
			# F = torch.cat((FNorm_dist[:,INDEX3[i,:]],FNorm_cos[:,[i]],FNorm_sin[:,[i]]),dim=1)
			F = FNorm_dist[:,[i]]
			# print(F.shape)
			if(i==0):
				out = self.Splines[0](F)
			else:
				out = out + self.Splines[i](F)	

		for i in range(7):
			out = out + self.Splines[i+45](torch.cat((FNorm_cos[:,[int(INDEX5[i])]],FNorm_sin[:,[int(INDEX5[i])]]),dim=1))	


		selection_bond = np.arange(0,9)
		selection_repel = np.arange(17,45)

		U1 = out + self.bondPotential(torch.cat((F_dist[:,selection_bond],F_angle),dim=1)) + self.repelPotential(F_dist[:,selection_repel])
		#print(feature.shape)
		U = sum(U1) # sum of potential for all example
		# force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return U # return force and potential for all training example



# model2BD = torch.load('Mymodel1_2BD.pt')
# model3BD = torch.load('Mymodel1_3BD.pt')
#model = Model()  # for new training
model2BD = torch.load('../../1_CV/Mymodel2BD'+str(modelID)+'.pt') # for transfer learning
# Distance_edge = np.loadtxt('Distance_edge.txt') 

## simulation part

# set boundaries

feature_Chignolin = Feature_new()

def Simulate(x0, dt, T, freq, SID):

	for i in range(10-SID+1):
		print('ID = ',SID+i)
		X = np.zeros((int(T/freq/10), ny, 30))
		if(i==0):
			x_old = x0
			X[0,:,:] = x0
		else:
			X[0,:,:] = x_old

		for t in range(1,int(T/10)):

			if(t%1000 == 0 ):
				print(t)

			x_old_tensor = torch.tensor(x_old).reshape(ny,30)
			x_old_tensor.requires_grad = True

			F_dist,F_cos,F_sin,F_angle = feature_Chignolin(x_old_tensor)

			# U_2BD = model2BD(Features)
			# U_3BD = model3BD(Features)
			U_2BD = model2BD(F_dist,F_cos,F_sin,F_angle)

			# y_pred_2BD = torch.autograd.grad(-U_2BD,x_old_tensor,create_graph=True, retain_graph=True)
			# y_pred_3BD = torch.autograd.grad(-U_3BD,x_old_tensor,create_graph=True, retain_graph=True)
			y_pred_2BD = torch.autograd.grad(-U_2BD,x_old_tensor,create_graph=True, retain_graph=True)

			# force, U = model(x_old_tensor)
			#print('x_old = ',x_old)

			# force_2BD = np.array(y_pred_2BD[0].detach())
			# force_3BD = np.array(y_pred_3BD[0].detach())
			force_2BD = np.array(y_pred_2BD[0].detach())
			force = force_2BD
			#print(force)
			#x_new = x_old + force*dt + np.sqrt(2*dt/beta)*np.random.randn(1,15)
			x_new = x_old + force*dt + np.sqrt(2*dt/beta)*np.random.randn(ny,30)

			if(np.sum(np.isnan(x_new[0,0]))):
				print(x_new)
				exit()
				
			#print(x_new)
			# while(0): # continue sample if the point is outside the boundary.
			# 	distance_new = CompteDistance(x_new)
			# 	#print(x_new[0,:], Distance_edge[0,:])
			# 	flag = sum(distance_new[0,:]<Distance_edge[0,:]) + sum(distance_new[0,:]>Distance_edge[1,:])
			# 	if(flag>0):
			# 		x_new = trapping(x_new, distance_new, dt)
			# 	else:
			# 		break
			
			if(t%freq == 0):
				X[int(t/freq),:,:] = x_new

			x_old = x_new
			x_old = np.float32(x_old)

		filenameS = '../TRAJs/Simtraj'+'_'+str(modelID)+'_'+str(modelID2)+'_'+str(SID+i)+'.npy'
		np.save(filenameS, X.reshape(-1,30))

		print('Saving: '+filenameS)
		# time.sleep(30)

		while(not os.path.exists(filenameS)):
			1
			# time.sleep(1)
			# np.save(filenameS, X.reshape(-1,30))

		print('Finished saving: '+filenameS, X.shape)
		
	return 0


ny = 10

T = 1000000

freq = 100
dt = 5e-4

### search for current traj
StartID = 0
for i in range(10):
	filename = '../TRAJs/Simtraj'+'_'+str(modelID)+'_'+str(modelID2)+'_'+str(i+1)+'.npy'

	if os.path.exists(filename):
		continue
	else:
		StartID = i+1
		break

print('StartID= ', StartID)

if(StartID==10):
	exit()

if(StartID == 1):
	data = np.load('/home/jw108/Remount_Nots/Chignolin_data/coordforce.npy') # there are 1,000,000 frames
	# data = np.load('/home/jw108/Remount_Got_Lnd/Chignolin_data/coordforce.npy') # there are 1,000,000 frames

	data = np.float32(data)
	# np.random.seed(5)
	sample = np.random.choice(1800000,ny, replace = False)
	x0 = data[sample,0:30]
	del data
else:
	data = np.load('../TRAJs/Simtraj'+'_'+str(modelID)+'_'+str(modelID2)+'_'+str(StartID-1)+'.npy')
	data = np.float32(data)
	data = data.reshape(int(T/freq/10),ny,30)
	# np.random.seed(5)
	# sample = np.random.choice(1000000,ny, replace = False)
	x0 = data[-1,:,:]
	del data




# data = np.loadtxt('/home/jw108/Remount/Dialanine_NN/coordforce_sub.txt') # there are 1,000,000 frames
# data = np.float32(data)
# np.random.seed(5)
# sample = np.random.choice(4579,ny, replace = False)
# x0 = data[sample,0:15]


Simtraj = Simulate(x0, dt, T, freq, StartID)

# for i in range(ny):
# 	filename = 'Simtraj/Simtraj_' + str(i+1)+'.txt'
# 	np.savetxt(filename, Simtraj3[:,i,:], fmt='%.6f')








