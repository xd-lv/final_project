import torch


class Config:
	data = 'data/'+'WIKI/'
	epochs = 200
	dim = 150
	act_func = torch.nn.functional.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	num = 50  # number of negative samples for each positive one
	k = 1200 # number of nearst neighbours to sample from