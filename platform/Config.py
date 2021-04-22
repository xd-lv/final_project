import torch


class Config:
	data = 'data/'+'YAGO/'#训练哪个数据集可将YAGO改

	#下面可存储与模型或训练有关的超参数
	epochs = 100
	dim = 150
	act_func = torch.nn.functional.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	num = 10  # number of negative samples for each positive one
	k = 1200 # number of nearst neighbours to sample from
