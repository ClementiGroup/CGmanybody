## Use NN to learn 15D force from 15D coordinates.


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

import argparse
# initiate the parser
parser = argparse.ArgumentParser()
# add long and short argument
parser.add_argument("--modelID", "-m", help="select cross validation inde")

args = parser.parse_args()

modelID = int(args.modelID)


seed = 7

BS = 512
E = 5 # Number of epoch
FS = 360000 # fold size N/5 1.8m/5 = 360000

Depth = 3
Width = 60
Power = 6
repelconst = 5.5**Power
Lambda = 3.0


torch.manual_seed(seed)
np.random.seed(seed)


data = np.load('/home/jw108/Remount_Nots/Chignolin_data/coordforce.npy')
# data = np.load('/home/jw108/Remount_Got_Lnd/Chignolin_data/coordforce.npy') # there are 1,000,000 frames
data = np.float32(data)	# change to float32, and transfer to tensor

np.random.shuffle(data) #  First shuffle the data, and use first 99% to be training set, last 1% = 10,000 to be test set

# bond and angle means and constants
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

# Creat data loader
class moleculedataset(Dataset):

	def __init__(self, selection):
		#traj = np.loadtxt('coordforce.txt',dtype=np.float32)
		# self.len = traj.shape[0]
		self.len = len(selection)
		self.x_data = torch.from_numpy(data[selection,0:30])
		self.y_data = torch.from_numpy(data[selection,30:60])
		# self.cluster = torch.from_numpy(data[selection,30])

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


# Creating 5 cross validation sets and test sets
N_CG = 10

if(modelID == 1):
	selection_train = np.arange(FS,FS*5)
	selection_valid = np.arange(0,FS*1)
	dataset = moleculedataset(selection_train)
	train_loader = DataLoader(dataset = dataset, batch_size = BS, shuffle = True, num_workers = 1)
	valid_x = torch.from_numpy(data[selection_valid,0:N_CG*3])
	valid_x.requires_grad = True
	valid_y = torch.from_numpy(data[selection_valid,N_CG*3:N_CG*6])
elif(modelID == 2):
	selection_train = np.concatenate((np.arange(0,FS*1),np.arange(FS*2,FS*5)))
	selection_valid = np.arange(FS*1,FS*2)
	dataset = moleculedataset(selection_train)
	train_loader = DataLoader(dataset = dataset, batch_size = BS, shuffle = True, num_workers = 1)
	valid_x = torch.from_numpy(data[selection_valid,0:N_CG*3])
	valid_x.requires_grad = True
	valid_y = torch.from_numpy(data[selection_valid,N_CG*3:N_CG*6])
elif(modelID == 3):
	selection_train = np.concatenate((np.arange(0,FS*2),np.arange(FS*3,FS*5)))
	selection_valid = np.arange(FS*2,FS*3)
	dataset = moleculedataset(selection_train)
	train_loader = DataLoader(dataset = dataset, batch_size = BS, shuffle = True, num_workers = 1)
	valid_x = torch.from_numpy(data[selection_valid,0:N_CG*3])
	valid_x.requires_grad = True
	valid_y = torch.from_numpy(data[selection_valid,N_CG*3:N_CG*6])
elif(modelID == 4):
	selection_train = np.concatenate((np.arange(0,FS*3),np.arange(FS*4,FS*5)))
	selection_valid = np.arange(FS*3,FS*4)
	dataset = moleculedataset(selection_train)
	train_loader = DataLoader(dataset = dataset, batch_size = BS, shuffle = True, num_workers = 1)
	valid_x = torch.from_numpy(data[selection_valid,0:N_CG*3])
	valid_x.requires_grad = True
	valid_y = torch.from_numpy(data[selection_valid,N_CG*3:N_CG*6])
elif(modelID == 5):
	selection_train = np.arange(0,FS*4)
	selection_valid = np.arange(FS*4,FS*5)
	dataset = moleculedataset(selection_train)
	train_loader = DataLoader(dataset = dataset, batch_size = BS, shuffle = True, num_workers = 1)
	valid_x = torch.from_numpy(data[selection_valid,0:N_CG*3])
	valid_x.requires_grad = True
	valid_y = torch.from_numpy(data[selection_valid,N_CG*3:N_CG*6])



# Testset_x = torch.from_numpy(data[length:10000,0:30])
# Testset_x.requires_grad = True
# Testset_y = torch.from_numpy(data[length:10000,30:60])




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


INDEX1_5BD = np.zeros((252,5))
it = 0
for i in np.arange(0,10):
	for j in np.arange(i+1,10):
		for k in np.arange(j+1,10):
			for l in np.arange(k+1,10):
				for m in np.arange(l+1,10):
					INDEX1_5BD[it] = [i,j,k,l,m]
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


INDEX3_5BD = np.zeros((252,10))
for i in np.arange(252):
	it = 0
	for j in np.arange(0,5):
		for k in np.arange(j+1,5):
			INDEX3_5BD[i,it] = INDEX2[(int(INDEX1_5BD[i,j]),int(INDEX1_5BD[i,k]))]
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
		return U # return force and potential for all training example


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
		return U # return force and potential for all training example



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
		return U # return force and potential for all training example


class Model5BD(torch.nn.Module):

	def __init__(self,Zscore_dist, Zscore_cos, Zscore_sin,BondConst,n,nl):
		super(Model5BD, self).__init__()
		# linear layers

		self.Splines = nn.ModuleList()
		for i in range(252):
			self.Splines.append(NLunit_Multibody(10,n,nl))
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


		for i in range(252):
			# F = torch.cat((FNorm_dist[:,INDEX3[i,:]],FNorm_cos[:,[i]],FNorm_sin[:,[i]]),dim=1)
			F = FNorm_dist[:,INDEX3_5BD[i,:]]

			# print(F.shape)
			if(i==0):
				out = self.Splines[0](F)
			else:
				out = out + self.Splines[i](F)			

		for i in range(7):
			out = out + self.Splines[i+252](torch.cat((FNorm_cos[:,[int(INDEX5[i])]],FNorm_sin[:,[int(INDEX5[i])]]),dim=1))	


		# selection_bond = np.arange(0,9)
		# selection_repel = np.arange(17,45)

		# U1 = out + self.bondPotential(torch.cat((F_dist[:,selection_bond],F_angle),dim=1)) + self.repelPotential(F_dist[:,selection_repel])
		#print(feature.shape)
		U = sum(out) # sum of potential for all example
		# force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return U # return force and potential for all training example


f_valid = open('valid_Error'+str(modelID)+'.txt','w')

model2BD = torch.load('/scratch/jw108/Data/Mymodel2BD3.pt')
model3BD = torch.load('/scratch/jw108/Data/Mymodel3BD4.pt')
model4BD = torch.load('/scratch/jw108/Data/Mymodel4BD5.pt')



# import sys
# sys.stdout = open('output.txt','wt')

feature_Chignolin = Feature_new()

# Fvalid_dist,Fvalid_cos,Fvalid_sin,Fvalid_angle = feature_Chignolin(valid_x)



def Validation(model,valid_x,valid_y):
	n = 3600
	# y_pred_4BD = torch.rand(360000,30)
	criterion = nn.MSELoss()
	for j in range(100):
		# print('valid j =', j)
		s = j*n
		e = s+n
		valid_x_temp = valid_x[s:e,:]

		Fvalid_dist_temp,Fvalid_cos_temp,Fvalid_sin_temp,Fvalid_angle_temp = feature_Chignolin(valid_x_temp)

		U_2BD = model2BD(Fvalid_dist_temp,Fvalid_cos_temp,Fvalid_sin_temp,Fvalid_angle_temp)
		U_3BD = model3BD(Fvalid_dist_temp,Fvalid_cos_temp,Fvalid_sin_temp,Fvalid_angle_temp)
		U_4BD = model4BD(Fvalid_dist_temp,Fvalid_cos_temp,Fvalid_sin_temp,Fvalid_angle_temp)
		U_5BD = model(Fvalid_dist_temp,Fvalid_cos_temp,Fvalid_sin_temp,Fvalid_angle_temp)

		
		y_pred_2BD_temp = torch.autograd.grad(-U_2BD,valid_x_temp,create_graph=True, retain_graph=True)
		y_pred_3BD_temp = torch.autograd.grad(-U_3BD,valid_x_temp,create_graph=True, retain_graph=True)
		y_pred_4BD_temp = torch.autograd.grad(-U_4BD,valid_x_temp,create_graph=True, retain_graph=True)
		y_pred_5BD_temp = torch.autograd.grad(-U_5BD,valid_x_temp,create_graph=True, retain_graph=True)

		y_pred_temp = y_pred_2BD_temp[0] + y_pred_3BD_temp[0] + y_pred_4BD_temp[0] + y_pred_5BD_temp[0]

		testError_temp = criterion(y_pred_temp, valid_y[s:e,:])

		if(j==0):

			testError = testError_temp.data.item()

		else:

			testError = testError + testError_temp.data.item()


	# testError = criterion(predict, valid_y)
	# print(testError/100)

	return testError/100


def Training(model,train_loader,valid_x, valid_y, IDX):

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.003, weight_decay=0e-15) # there are other optimizer: Adam, LBFGS, RMSprop, Rprop...
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4], gamma=0.3)

	Error = np.zeros((1, 3)) # for storing training error as function of # of minibatches
	step = 0

	print('\nValiation set', IDX)	
	
	for epoch in range(E):
		scheduler.step() # LR decay

		for i, data in enumerate(train_loader, 0): # in each epoch, loop over all 1000 example

			inputs, labels = data

			inputs.requires_grad = True

			F_dist,F_cos,F_sin,F_angle = feature_Chignolin(inputs) # predictions

			# print(F_dist[0,:])
			# print(F_cos[0,:])
			# print(F_sin[0,:])
			# print(F_angle[0,:])
			# exit()

			U_2BD = model2BD(F_dist,F_cos,F_sin,F_angle)
			U_3BD = model3BD(F_dist,F_cos,F_sin,F_angle)
			U_4BD = model4BD(F_dist,F_cos,F_sin,F_angle)
			U_5BD = model(F_dist,F_cos,F_sin,F_angle)


			y_pred_2BD = torch.autograd.grad(-U_2BD,inputs,create_graph=True, retain_graph=True)
			y_pred_3BD = torch.autograd.grad(-U_3BD,inputs,create_graph=True, retain_graph=True)
			y_pred_4BD = torch.autograd.grad(-U_4BD,inputs,create_graph=True, retain_graph=True)
			y_pred_5BD = torch.autograd.grad(-U_5BD,inputs,create_graph=True, retain_graph=True)

			predict = y_pred_2BD[0] + y_pred_3BD[0] + y_pred_4BD[0] + y_pred_5BD[0]


			# print(predict)
			# exit()

			loss = criterion(predict, labels)

			# print(predict)
			# print(labels)

			# print(loss)
			# exit()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if True:
				for p in model.parameters():
					# print(np.array(p.shape).size)
					if(np.array(p.shape).size==2):
						# print(p.data)
						weight = p.data
						u,s,v = torch.svd(weight)
						lip_reg = torch.max(((s[0])/Lambda),torch.tensor([1.0]))
						# print(lip_reg)
						# print(p.data)
						p.data = p.data/lip_reg
						# print(p.data)
						# break

			# test error
			if(False and i%2000 == 0 and i!=0 and (epoch+1)%5==0 ):
				# U_4BD = model(Fvalid_dist,Fvalid_cos,Fvalid_sin,Fvalid_angle)

				# y_pred_4BD = torch.autograd.grad(-U_4BD,valid_x,create_graph=True, retain_graph=True)

				# predict = y_pred_4BD[0]

				# testError = criterion(predict, valid_y)

				# print(predict)
				# print(valid_y.shape)
				# print(testError)
			
				#testMAE = torch.sum(abs(y_pred[0] - Testset_y)).detach()/15/50000
				testError = Validation(model,valid_x,valid_y)

				Error[step,:] = [step, loss.data.item(), testError]
				Error = np.append(Error,[[0,0,0]],axis = 0)
				step = step + 1

				print(epoch, i, loss.data, testError)	
				#print('input = ', inputs, '\nfeature =', feature, '\noutput =', y_pred[0], '\ntarget =', labels, '\nPotential =', U_temp)
				#print(epoch, i, loss.data)	
				#print(feature_train[0,:])

			# test error
			if(i%1000 == 0 ):

				print(epoch, i, loss.data)	

		torch.save(model, 'Mymodel5BD'+IDX+'_Epoch'+str(epoch)+'.pt')

	# U_4BD = model(Fvalid_dist,Fvalid_cos,Fvalid_sin,Fvalid_angle)

	# y_pred_4BD = torch.autograd.grad(-U_4BD,valid_x,create_graph=True, retain_graph=True)

	# predict = y_pred_4BD[0]

	# validError = criterion(predict, valid_y).data.item()

	validError = Validation(model,valid_x,valid_y)
	#np.savetxt('Valid_Error'+IDX+'.txt', validError.data.item())
	print(validError, file = f_valid)

	Error = np.delete(Error, len(Error)-1, axis = 0)
	np.savetxt('Error'+IDX+'.txt', Error,fmt='%.6f')	# save error to file
	torch.save(model, 'Mymodel5BD'+IDX+'.pt')	# save learnt NN to file

	return validError

torch.manual_seed(seed)
model_temp = Model5BD(Zscore_dist, Zscore_cos, Zscore_sin,BondConst,Width,Depth)
valid_error = Training(model_temp, train_loader, valid_x, valid_y,str(modelID))

# torch.manual_seed(seed)
# model2 = Model(Zscore,BondConst,Width,Depth)
# valid_error2 = Training(model2, train_loader2, valid_x2, valid_y2,'2')

# torch.manual_seed(seed)
# model3 = Model(Zscore,BondConst,Width,Depth)
# valid_error3 = Training(model3, train_loader3, valid_x3, valid_y3,'3')

# torch.manual_seed(seed)
# model4 = Model(Zscore,BondConst,Width,Depth)
# valid_error4 = Training(model4, train_loader4, valid_x4, valid_y4,'4')

# torch.manual_seed(seed)
# model5 = Model(Zscore,BondConst,Width,Depth)
# valid_error5 = Training(model5, train_loader5, valid_x5, valid_y5,'5')

# valid_errors = np.array([valid_error1,valid_error2,valid_error3,valid_error4,valid_error5])
# mean_valid_error = np.mean(valid_errors)
# std_valid_error = np.std(valid_errors)

# print(mean_valid_error,' ', std_valid_error, file = f_valid)

f_valid.close()

exit()





