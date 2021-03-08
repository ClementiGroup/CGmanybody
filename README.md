# Many-body Coarse Graining Force Field

This document contains codes to construct many-body coarse grained molecular model of chignolin, including 2-body, 2,3-body, 2,3,4-body, 2,3,4C-body, and 2,3,4C,5-body, where 'C' represent chiral. 

Since the implementation of differemnt order of many-body required different codes, different many-body are in different sub folders.

In each many-body models, there are 4 individual procedures:

	-1_CV: conduct cross-validation, and traning of the model.

	-2_Simulation: conduct coarse graining simulation using the trained model.

	-3_FE: compute free energy of the simulation trajectory on 2-dimensional TICA space.
	
	-4_Potential_Decomp: calculate the CG potential predicted by many-body models. 

