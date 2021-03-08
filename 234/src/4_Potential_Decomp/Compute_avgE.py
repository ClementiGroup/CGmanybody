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
U4_break = np.loadtxt('U4_break.txt')
U234_break = np.loadtxt('U234_break.txt')



N = len(Torsion)

# loop over all N point, create Space index board, and index dictionary

print('Creating Index Board...')

Min_x = -30
Max_x = 5

Min_y = -20
Max_y = 20

Torsion[:,0] = Torsion[:,0] - Min_x
Torsion[:,1] = Torsion[:,1] - Min_y

gridN = 100

Step_x = (Max_x - Min_x)/gridN
Step_y = (Max_y - Min_y)/gridN

IndexBoard = np.zeros((gridN,gridN))

Mymap = {}

BinIdx = 1

U4 = np.zeros((gridN,gridN,5))
U234 = np.zeros((gridN,gridN,5))


# Generate grid and bins
for i in range(N):
	x = int(Torsion[i,0]/Step_x)
	y = int(Torsion[i,1]/Step_y)

	if(x==gridN):
		x=gridN-1
	if(y==gridN):
		y=gridN-1

	if(IndexBoard[x,y]==0):
		IndexBoard[x,y] = BinIdx
		Mymap[BinIdx] = np.array([i])		
		BinIdx = BinIdx+1
	else:
		ID = IndexBoard[x,y]
		Mymap[ID] = np.append(Mymap[ID],i)

	U4[x,y,:] = U4[x,y,:] + U4_break[i,:] # compute sum of U for each bin
	U234[x,y,:] = U234[x,y,:] + U234_break[i,:]

	if(i%10000 == 0):
		print('i =', i)

# Torsion = Torsion - Max
O = len(Mymap)
print(len(Mymap))



# compute average U for each bin

for i in range(O):
	x = int(Torsion[Mymap[i+1][0],0]/Step_x)
	y = int(Torsion[Mymap[i+1][0],1]/Step_y)

	U4[x,y,:] = U4[x,y,:]/len(Mymap[i+1])
	U234[x,y,:] = U234[x,y,:]/len(Mymap[i+1])



def plot_avgU(H,title,level):
	# H = U[:,:,0]
	H[IndexBoard==0]=np.nan
	H = H.transpose()

	H = H-np.min(H[np.isfinite(H)])
	Mincolor = np.min(H[np.isfinite(H)])
	Maxcolor = np.max(H[np.isfinite(H)])

	print(Mincolor)

	fig = plt.figure(figsize=(8,6))
	ax = fig.gca()
	surf = plt.imshow(H, cmap = cm.jet, interpolation='nearest', origin='low',extent=[-30, 5, -20, 20],aspect='auto')
	ax.set_xlabel(r'$\phi$',fontsize=20)
	ax.set_ylabel(r'$\psi$',fontsize=20)
	# plt.xlim((-3.14,3.14))
	# plt.ylim((-3.14,3.14))
	fig.colorbar(surf, shrink=1.0, aspect=9)
	plt.clim(Mincolor,Mincolor+level)
	# plt.clim(-3,9)
	plt.savefig(title,dpi=100)	




plot_avgU(U4[:,:,0],'Avg_U2_1a.pdf',10)
plot_avgU(U4[:,:,1],'Avg_U2_2a.pdf',10)
plot_avgU(U4[:,:,2],'Avg_U2_3a.pdf',10)
plot_avgU(U4[:,:,3],'Avg_U2_4a.pdf',10)
plot_avgU(U4[:,:,4],'Avg_U2_5a.pdf',10)


plot_avgU(U4[:,:,0],'Avg_U4_1b.pdf',30)
plot_avgU(U4[:,:,1],'Avg_U4_2b.pdf',30)
plot_avgU(U4[:,:,2],'Avg_U4_3b.pdf',30)
plot_avgU(U4[:,:,3],'Avg_U4_4b.pdf',30)
plot_avgU(U4[:,:,4],'Avg_U4_5b.pdf',30)

plot_avgU(U234[:,:,0],'Avg_U234_1.pdf',30)
plot_avgU(U234[:,:,1],'Avg_U234_2.pdf',30)
plot_avgU(U234[:,:,2],'Avg_U234_3.pdf',30)
plot_avgU(U234[:,:,3],'Avg_U234_4.pdf',30)
plot_avgU(U234[:,:,4],'Avg_U234_5.pdf',30)


exit()

