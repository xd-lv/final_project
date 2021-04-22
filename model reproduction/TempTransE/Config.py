#import torch


class Config:
	data = 'data/'+'YAGO/'
	seq_len = 5
	epochs = 600
	dim = 300
	#act_func = torch.nn.functional.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	num = 125  # number of negative samples for each positive one
	k = 150 # number of nearst neighbours to sample from