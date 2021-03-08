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

# import sys
# sys.stdout = open('output.txt','a')

# Torsion = np.loadtxt('/home/jw108/Remount_Nots/Force_Match/Dialanine_data/Torsion.txt')
# data = np.loadtxt('/home/jw108/Remount_Nots/Force_Match/Dialanine_data/coordforce.txt') # there are 1,000,000 frames
# data = np.float32(data)	# change to float32, and transfer to tensor
#np.random.shuffle(data) #  First shuffle the data, and use first 99% to be training set, last 1% = 10,000 to be test set


Torsion = np.load('/home/jw108/Remount_Nots/Chignolin_data/ZZ.npy')
# ny = 100 

# for i in range(ny):
# 	filename = '../Simtraj/Simtraj_'+str(i+1)+'.txt'
# 	data_temp = np.loadtxt(filename) # there are 1,000,000 frames
# 	data_temp = np.float32(data_temp)	# change to float32, and transfer to tensor
# 	if(i == 0):
# 		data = data_temp
# 	else:
# 		data = np.concatenate((data,data_temp),axis = 0)
# 	print('input Simtraj:', i)

data = np.load('/home/jw108/Remount_Nots/Chignolin_data/coordforce.npy')
data = np.float32(data)

N = len(data)
#data = data[np.arange(0,N,5),:]
data = torch.from_numpy(data)
data.requires_grad = True

beta = 1.6775

Power = 6
repelconst = 5.5**Power

##### Define CGnet

BondConst = np.loadtxt('BondConst.txt') 
BondConst = np.float32(BondConst)
BondConst = BondConst.transpose()
# BondConst = BondConst[:,[0,4,7,9,10,11,12]]
BondConst = torch.from_numpy(BondConst)

Zscore_dist = np.loadtxt('Zscore_dist.txt') 
Zscore_dist = np.float32(Zscore_dist)
Zscore_dist = Zscore_dist.transpose()
Zscore_dist = torch.from_numpy(Zscore_dist)


Zscore_cos = np.loadtxt('Zscore_cos.txt') 
Zscore_cos = np.float32(Zscore_cos)
Zscore_cos = Zscore_cos.transpose()
Zscore_cos = torch.from_numpy(Zscore_cos)

Zscore_sin = np.loadtxt('Zscore_sin.txt') 
Zscore_sin = np.float32(Zscore_sin)
Zscore_sin = Zscore_sin.transpose()
Zscore_sin = torch.from_numpy(Zscore_sin)

# define a feature layer
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
		n1 = len(U)

		P = U.reshape(n1,10,3)	#10 atoms, shape (n,10,3)

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
		n1 = len(U)

		P = U.reshape(n1,10,3)	#10 atoms, shape (n,10,3)

		V12 = P[:,1:10,:] - P[:,0:9,:] # 9 1-2 distance, shape (n,9,3)
		V13 = P[:,2:10,:] - P[:,0:8,:] # 8	(n,8,3)
		V14 = P[:,3:10,:] - P[:,0:7,:] # 7 	(n,7,3)
		V15 = P[:,4:10,:] - P[:,0:6,:] # 6
		V16 = P[:,5:10,:] - P[:,0:5,:] # 5 
		V17 = P[:,6:10,:] - P[:,0:4,:] # 4
		V18 = P[:,7:10,:] - P[:,0:3,:] # 3
		V19 = P[:,8:10,:] - P[:,0:2,:] # 2
		V110 = P[:,9,:] - P[:,0,:]     # 1
		V110 = V110.reshape(n1,1,3)
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
		return U1 # return force and potential for all training example



class Model3BD(torch.nn.Module):

	def __init__(self,Zscore_dist, Zscore_cos, Zscore_sin,BondConst,n,nl):
		super(Model3BD, self).__init__()
		# linear layers

		self.Splines = nn.ModuleList()
		for i in range(120):
			self.Splines.append(NLunit_Multibody(3,n,nl))
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


		for i in range(120):
			# F = torch.cat((FNorm_dist[:,INDEX3[i,:]],FNorm_cos[:,[i]],FNorm_sin[:,[i]]),dim=1)
			F = FNorm_dist[:,INDEX3_3BD[i,:]]
			# print(F.shape)
			if(i==0):
				out = self.Splines[0](F)
			else:
				out = out + self.Splines[i](F)	

		for i in range(7):
			out = out + self.Splines[i+120](torch.cat((FNorm_cos[:,[int(INDEX5[i])]],FNorm_sin[:,[int(INDEX5[i])]]),dim=1))	


		# selection_bond = np.arange(0,9)
		# selection_repel = np.arange(17,45)

		# U1 = out + self.bondPotential(torch.cat((F_dist[:,selection_bond],F_angle),dim=1)) + self.repelPotential(F_dist[:,selection_repel])
		#print(feature.shape)
		U = sum(out) # sum of potential for all example
		# force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return out # return force and potential for all training example


class Model4BD(torch.nn.Module):

	def __init__(self,Zscore_dist, Zscore_cos, Zscore_sin,BondConst,n,nl):
		super(Model4BD, self).__init__()
		# linear layers

		self.Splines = nn.ModuleList()
		for i in range(210):
			self.Splines.append(NLunit_Multibody(6,n,nl))
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


		for i in range(210):
			# F = torch.cat((FNorm_dist[:,INDEX3[i,:]],FNorm_cos[:,[i]],FNorm_sin[:,[i]]),dim=1)
			F = FNorm_dist[:,INDEX3_4BD[i,:]]

			# print(F.shape)
			if(i==0):
				out = self.Splines[0](F)
			else:
				out = out + self.Splines[i](F)			

		for i in range(7):
			out = out + self.Splines[i+210](torch.cat((FNorm_cos[:,[int(INDEX5[i])]],FNorm_sin[:,[int(INDEX5[i])]]),dim=1))	


		# selection_bond = np.arange(0,9)
		# selection_repel = np.arange(17,45)

		# U1 = out + self.bondPotential(torch.cat((F_dist[:,selection_bond],F_angle),dim=1)) + self.repelPotential(F_dist[:,selection_repel])
		#print(feature.shape)
		U = sum(out) # sum of potential for all example
		# force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return out # return force and potential for all training example



class Model4BD_ch(torch.nn.Module):

	def __init__(self,Zscore_dist, Zscore_cos, Zscore_sin,BondConst,n,nl):
		super(Model4BD_ch, self).__init__()
		# linear layers

		self.Splines = nn.ModuleList()
		for i in range(210):
			self.Splines.append(NLunit_Multibody(8,n,nl))
		for i in range(7):
			self.Splines.append(NLunit_Multibody(2,n,nl))	

		self.feature = Feature_new()
		self.bondPotential = bondPotential(BondConst)
		self.repelPotential = repelPotential()


	def forward(self, F_dist, F_cos, F_sin, F_angle): 
		# feature_dist, feature_cos, feature_sin = F
		selection_feature = np.arange(0,45)
		FNorm_dist = (F_dist[:,selection_feature]-Zscore_dist[0,selection_feature])/Zscore_dist[1,selection_feature]
		FNorm_cos = (F_cos-Zscore_cos[0,:])/Zscore_cos[1,:]
		FNorm_sin = (F_sin-Zscore_sin[0,:])/Zscore_sin[1,:]


		for i in range(210):
			F = torch.cat((FNorm_dist[:,INDEX3_4BD[i,:]],FNorm_cos[:,[i]],FNorm_sin[:,[i]]),dim=1)
			# print(F.shape)
			if(i==0):
				out = self.Splines[0](F)
			else:
				out = out + self.Splines[i](F)			

		for i in range(7):
			out = out + self.Splines[i+210](torch.cat((FNorm_cos[:,[int(INDEX5[i])]],FNorm_sin[:,[int(INDEX5[i])]]),dim=1))	


		# selection_bond = np.arange(0,9)
		# selection_repel = np.arange(17,45)

		# U1 = out + self.bondPotential(torch.cat((F_dist[:,selection_bond],F_angle),dim=1)) + self.repelPotential(F_dist[:,selection_repel])
		#print(feature.shape)
		U = sum(out) # sum of potential for all example
		# force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return out # return force and potential for all training example

model2BD = torch.load('../../../2BD/B_3_60_5_BB/1_CV/Mymodel2BD3.pt')
model3BD = torch.load('../../../23BD/B_3_60_5_BB/1_CV/Mymodel3BD4.pt')

# model3BD = torch.load('../1_CV/Mymodel3BD5.pt')
# model4BD = torch.load('../../../234BD/C_5_200_10_B/1_CV/Mymodel4BD2.pt')
# model4BDchiral = torch.load('../1_CV/Mymodel4BD1.pt')
# modelNBD = torch.load('MymodelNBD5.pt') # for transfer learning
model4BD1 = torch.load('../1_CV/Mymodel4BD1.pt')
model4BD2 = torch.load('../1_CV/Mymodel4BD2.pt')
model4BD3 = torch.load('../1_CV/Mymodel4BD3.pt')
model4BD4 = torch.load('../1_CV/Mymodel4BD4.pt')
model4BD5 = torch.load('../1_CV/Mymodel4BD5.pt')


U4_break = np.zeros((N,5)) # 0,1,2 are total, pri and net
U234_break = np.zeros((N,5)) # 0,1,2 are total, pri and net


feature_Chignolin = Feature_new()

n = 100
N1 = int(N/n)
for i in range(n):
	inputs = data[i*N1:(i+1)*N1,0:30]

	print(inputs.shape)

	F_dist,F_cos,F_sin,F_angle = feature_Chignolin(inputs)

	U_2BD = model2BD(F_dist,F_cos,F_sin,F_angle).detach()
	U_3BD = model3BD(F_dist,F_cos,F_sin,F_angle).detach()
	# U_4BD = model4BD(Features).detach()
	# U_4BDchiral = model4BDchiral(Features).detach()
	# U23 = U_2BD + U_3BD
	# Total1 = U_2BD + U_3BD + U_4BD
	# Total2 = U_2BD + U_3BD + U_4BDchiral
	U_4BD1 = model4BD1(F_dist,F_cos,F_sin,F_angle).detach()
	U_4BD2 = model4BD2(F_dist,F_cos,F_sin,F_angle).detach()
	U_4BD3 = model4BD3(F_dist,F_cos,F_sin,F_angle).detach()
	U_4BD4 = model4BD4(F_dist,F_cos,F_sin,F_angle).detach()
	U_4BD5 = model4BD5(F_dist,F_cos,F_sin,F_angle).detach()


	print(U_4BD1.shape)

	# U_break[i*N1:(i+1)*N1,:] = np.concatenate((U_2BD,U_3BD,U_4BD,U_4BDchiral,U23,Total1,Total2),axis=1)
	U4_break[i*N1:(i+1)*N1,:] = np.concatenate((U_4BD1,U_4BD2,U_4BD3,U_4BD4,U_4BD5),axis=1)
	U234_break[i*N1:(i+1)*N1,:] = np.concatenate((U_4BD1,U_4BD2,U_4BD3,U_4BD4,U_4BD5),axis=1)+U_2BD+U_3BD

	# U234_break[i*N1:(i+1)*N1,:] = np.concatenate((U_4BD1,U_4BD2,U_4BD3,U_4BD4,U_4BD5),axis=1)+U_2BD+U_3BD
	# U_break[i*N1:(i+1)*N1,:] = U_NBD

	if(i%1 == 0):
		print('i=',i)

print(U4_break.shape)
np.savetxt('U4_break.txt', U4_break, fmt='%.3f')
np.savetxt('U234_break.txt', U234_break, fmt='%.3f')
# np.savetxt('U234_break.txt', U234_break)


# force, U,UN,UP, feature = model(data[0:2,:])
# print(force)
# print(U)
# print(UN)
# print(UP)
# print(feature)

U4_break = U4_break - np.min(U4_break,axis=0)
U234_break = U234_break - np.min(U234_break,axis=0)


def plot_U(U,title,level):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.gca()
	colors = np.arange(len(Torsion))
	surf = plt.scatter(Torsion[:,0],Torsion[:,1], c = U[0:1000000], marker='.',s=1,cmap='jet')
	ax.set_xlabel(r'$\phi$',fontsize=20)
	ax.set_ylabel(r'$\psi$',fontsize=20)
	plt.xlim((-30,5))
	plt.ylim((-20,20))
	# plt.title('Torsion')
	fig.colorbar(surf, shrink=1.0, aspect=9)
	minP = np.min(U)
	maxP = np.max(U)
	plt.clim(minP,minP+level)
	# plt.clim(-3,9)
	plt.savefig(title+'.png',dpi=100)




# plot_U(U4_break[:,0],'U4BD_1',4)
# plot_U(U4_break[:,1],'U4BD_2',4)
# plot_U(U4_break[:,2],'U4BD_3',4)
# plot_U(U4_break[:,3],'U4BD_4',4)
# plot_U(U4_break[:,4],'U4BD_5',4)

# plot_U(U234_break[:,0],'U234BD_1',30)
# plot_U(U234_break[:,1],'U234BD_2',30)
# plot_U(U234_break[:,2],'U234BD_3',30)
# plot_U(U234_break[:,3],'U234BD_4',30)
# plot_U(U234_break[:,4],'U234BD_5',30)



exit()


