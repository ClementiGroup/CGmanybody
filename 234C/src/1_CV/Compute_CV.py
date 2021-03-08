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


f_valid = open('CV_error.txt','w')
data = np.zeros(5)

for i in range(5):
	data[i] = np.loadtxt('valid_Error'+str(i+1)+'.txt')
	print(data[i], file = f_valid)

# valid_errors = np.array([valid_error1,valid_error2,valid_error3,valid_error4,valid_error5])
mean_valid_error = np.mean(data)
std_valid_error = np.std(data)/np.sqrt(5)

print('Mean_Valid_error = ', mean_valid_error, '+/-', std_valid_error, file = f_valid)



exit()

seed = 5

BS = 512

Depth = 5
Width = 200
Lambda = 4.0

torch.manual_seed(seed)
np.random.seed(seed)

data = np.load('/scratch/jw108/Chignolin_data/coordforce.npy') # there are 1,000,000 frames
data = np.float32(data)	# change to float32, and transfer to tensor

np.random.shuffle(data) #  First shuffle the data, and use first 99% to be training set, last 1% = 10,000 to be test set

# bond and angle means and constants
BondConst = np.loadtxt('BondConst.txt') 
BondConst = np.float32(BondConst)
BondConst = BondConst.transpose()
# BondConst = BondConst[:,[0,4,7,9,10,11,12]]
BondConst = torch.from_numpy(BondConst)

Zscore = np.loadtxt('Zscore.txt') 
Zscore = np.float32(Zscore)
Zscore = Zscore.transpose()
Zscore = torch.from_numpy(Zscore)

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
FS = 360000 # fold size N/5 1.8m/5 = 360000

selection_train1 = np.arange(FS,FS*5)
selection_valid1 = np.arange(0,FS*1)
dataset1 = moleculedataset(selection_train1)
train_loader1 = DataLoader(dataset = dataset1, batch_size = BS, shuffle = True, num_workers = 1)
valid_x1 = torch.from_numpy(data[selection_valid1,0:30])
valid_x1.requires_grad = True
valid_y1 = torch.from_numpy(data[selection_valid1,30:60])

selection_train2 = np.concatenate((np.arange(0,FS*1),np.arange(FS*2,FS*5)))
selection_valid2 = np.arange(FS*1,FS*2)
dataset2 = moleculedataset(selection_train2)
train_loader2 = DataLoader(dataset = dataset2, batch_size = BS, shuffle = True, num_workers = 1)
valid_x2 = torch.from_numpy(data[selection_valid2,0:30])
valid_x2.requires_grad = True
valid_y2 = torch.from_numpy(data[selection_valid2,30:60])

selection_train3 = np.concatenate((np.arange(0,FS*2),np.arange(FS*3,FS*5)))
selection_valid3 = np.arange(FS*2,FS*3)
dataset3 = moleculedataset(selection_train3)
train_loader3 = DataLoader(dataset = dataset3, batch_size = BS, shuffle = True, num_workers = 1)
valid_x3 = torch.from_numpy(data[selection_valid3,0:30])
valid_x3.requires_grad = True
valid_y3 = torch.from_numpy(data[selection_valid3,30:60])

selection_train4 = np.concatenate((np.arange(0,FS*3),np.arange(FS*4,FS*5)))
selection_valid4 = np.arange(FS*3,FS*4)
dataset4 = moleculedataset(selection_train4)
train_loader4 = DataLoader(dataset = dataset4, batch_size = BS, shuffle = True, num_workers = 1)
valid_x4 = torch.from_numpy(data[selection_valid4,0:30])
valid_x4.requires_grad = True
valid_y4 = torch.from_numpy(data[selection_valid4,30:60])

selection_train5 = np.arange(0,FS*4)
selection_valid5 = np.arange(FS*4,FS*5)
dataset5 = moleculedataset(selection_train5)
train_loader5 = DataLoader(dataset = dataset5, batch_size = BS, shuffle = True, num_workers = 1)
valid_x5 = torch.from_numpy(data[selection_valid5,0:30])
valid_x5.requires_grad = True
valid_y5 = torch.from_numpy(data[selection_valid5,30:60])


# Testset_x = torch.from_numpy(data[length:10000,0:30])
# Testset_x.requires_grad = True
# Testset_y = torch.from_numpy(data[length:10000,30:60])


# define a feature layer
class Feature(nn.Module): 

	def __init__(self):
		super(Feature, self).__init__()

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

		return out


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
		Potential = torch.sum(71/U**2,1).reshape(n,1)
		return Potential

class Model(torch.nn.Module):

	def __init__(self,Zscore,BondConst,n,nl):
		super(Model, self).__init__()
		# linear layers

		names = ['0']*(nl+1)
		for i in np.arange(1,nl+1):
			names[i] = str(i)


		self.net = nn.Sequential()
		self.net.add_module('0',nn.Linear(59,n))
		for i in np.arange(1,nl):
			self.net.add_module(names[i],nn.Linear(n,n))
			self.net.add_module(names[i]+'act',torch.nn.Tanh())

		self.net.add_module(names[nl],nn.Linear(n,1))

		self.feature = Feature()
		self.bondPotential = bondPotential(BondConst)
		self.repelPotential = repelPotential()


	def forward(self, x): 
		feature = self.feature(x)
		selection_feature = np.concatenate((np.arange(0,45),np.arange(53,67)))
		featureNorm = (feature[:,selection_feature]-Zscore[0,selection_feature])/Zscore[1,selection_feature]

		out = self.net(featureNorm)

		selection_bond = np.concatenate((np.arange(0,9),np.arange(45,53)))
		selection_repel = np.arange(17,45)

		U1 = out + self.bondPotential(feature[:,selection_bond]) + self.repelPotential(feature[:,selection_repel])
		#print(feature.shape)
		U = sum(U1) # sum of potential for all example
		force = torch.autograd.grad(-U,x,create_graph=True, retain_graph=True) # this deals with multiple training examples
		# return out5
		return force, U1 # return force and potential for all training example

E = 20 # Number of epoch

f_valid = open('valid_Error5.txt','w')

# import sys
# sys.stdout = open('output.txt','wt')

def Training(model,train_loader,valid_x, valid_y, IDX):

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=0e-15) # there are other optimizer: Adam, LBFGS, RMSprop, Rprop...
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4,15], gamma=0.3)

	Error = np.zeros((1, 3)) # for storing training error as function of # of minibatches
	step = 0

	print('\nValiation set', IDX)	
	
	for epoch in range(E):
		scheduler.step() # LR decay

		for i, data in enumerate(train_loader, 0): # in each epoch, loop over all 1000 example

			inputs, labels = data

			inputs.requires_grad = True

			y_pred_train, U_temp = model(inputs) # predictions

			loss = criterion(y_pred_train[0], labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

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
			if(i%2000 == 0):
				y_pred, U_temp = model(valid_x)
				testError = criterion(y_pred[0], valid_y)
				#testMAE = torch.sum(abs(y_pred[0] - Testset_y)).detach()/15/50000

				Error[step,:] = [step, loss.data.item(), testError.data.item()]
				Error = np.append(Error,[[0,0,0]],axis = 0)
				step = step + 1

				print(epoch, i, loss.data, testError.data)	
				#print('input = ', inputs, '\nfeature =', feature, '\noutput =', y_pred[0], '\ntarget =', labels, '\nPotential =', U_temp)
				#print(epoch, i, loss.data)	
				#print(feature_train[0,:])


	y_pred, U_temp  = model(valid_x)
	validError = criterion(y_pred[0], valid_y).data.item()
	#np.savetxt('Valid_Error'+IDX+'.txt', validError.data.item())
	print(validError, file = f_valid)

	Error = np.delete(Error, len(Error)-1, axis = 0)
	np.savetxt('Error'+IDX+'.txt', Error,fmt='%.6f')	# save error to file
	torch.save(model, 'Mymodel'+IDX+'.pt')	# save learnt NN to file

	return validError

# torch.manual_seed(seed)
# model1 = Model(Zscore,BondConst,Width,Depth)
# valid_error1 = Training(model1, train_loader1, valid_x1, valid_y1,'1')

# torch.manual_seed(seed)
# model2 = Model(Zscore,BondConst,Width,Depth)
# valid_error2 = Training(model2, train_loader2, valid_x2, valid_y2,'2')

# torch.manual_seed(seed)
# model3 = Model(Zscore,BondConst,Width,Depth)
# valid_error3 = Training(model3, train_loader3, valid_x3, valid_y3,'3')

# torch.manual_seed(seed)
# model4 = Model(Zscore,BondConst,Width,Depth)
# valid_error4 = Training(model4, train_loader4, valid_x4, valid_y4,'4')

torch.manual_seed(seed)
model5 = Model(Zscore,BondConst,Width,Depth)
valid_error5 = Training(model5, train_loader5, valid_x5, valid_y5,'5')

# valid_errors = np.array([valid_error1,valid_error2,valid_error3,valid_error4,valid_error5])
# mean_valid_error = np.mean(valid_errors)
# std_valid_error = np.std(valid_errors)

# print(mean_valid_error,' ', std_valid_error, file = f_valid)

f_valid.close()

exit()





