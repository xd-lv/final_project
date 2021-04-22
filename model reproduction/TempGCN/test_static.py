import torch
import numpy as np
from include.Config import Config
from include.RDGCN import *
from include.Test import *
from include.Load import *
from include.utils import *

import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(3)


'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

'''
seed = 12306
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

if __name__ == '__main__':
    e, r_num, time_unit =get_total_number(Config.data,'stat.txt')

    TKG, last_time = load_TKG(Config.data, 'quadruples.txt')
    train_TKG, valid_TKG, test_TKG = set_partition(TKG, 0.7, 0.1, 0.2)

    seq_len = Config.seq_len

    model = RDGCN_layer(Config.dim).cuda()
    embedding_list = get_input_embeddings(Config.data, 'vectorList.json')

    input_embedding = get_input_layer(embedding_list)

    entity_pre_result=[]
    relation_pre_result=[]
    for time in range(30,33):
    #for time in range(last_time):
        print("time: ",time)
        KG = train_TKG[time]
        '''
        if time > 0 and time % seq_len == 0:
            get_hits_mrr(output, dual_X, KG)
            assert False
        '''
        output, dual_X, J = training(model, input_embedding, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.num, Config.k, e, r_num, KG, 0.006, Config.epochs)
        entity_pre_result.append(evaluate_entity(output, dual_X, test_TKG[time]))
        relation_pre_result.append(evaluate_relation(output, dual_X, test_TKG[time], r_num))
    print(entity_pre_result)
    print(relation_pre_result)
    print(np.mean(entity_pre_result, axis=0))
    print(np.mean(relation_pre_result, axis=0))
