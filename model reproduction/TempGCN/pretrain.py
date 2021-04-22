import torch
import numpy as np
from include.Config import Config
from include.RDGCN1 import *
from include.Test2 import *
from include.Load import *
from include.utils import *

import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(0)


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

    train_TKG, last_time = load_TKG(Config.data, 'train_quadruples.txt')
    test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
    static_bound=30
    train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', static_bound, 0)
    test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', static_bound, 0)
    valid_SKG = load_static_graph(Config.data, 'valid_quadruples.txt', static_bound, 0)
    all_SKG = set(train_SKG) | set(test_SKG) | set(valid_SKG)


    model = RDGCN_layer(Config.dim, Config.dim*2).cuda()
    #embedding_list = get_input_embeddings(Config.data, 'vectorList_TransE_dim150.json')
    embedding_list = torch.Tensor(e, Config.dim)
    torch.nn.init.xavier_uniform(embedding_list)
    input_embedding = model.get_input_layer(embedding_list)

    entity_pre_result=[]
    relation_pre_result=[]

    print("triples: ", len(train_SKG))
    output, dual_X, J=training(model, input_embedding, Config.act_func, Config.alpha, Config.beta, Config.gamma,
                                 Config.num, Config.k, e, r_num, train_SKG, all_SKG, 0.001, 2000)

    with open(Config.data + 'vectorList_RDGCN_xinit_bound30.json', 'w') as f:
        json.dump(input_embedding.cpu().detach().numpy().tolist(), f)
    #torch.save(model, Config.data + 'static_model.chkpnt')

    entity_pre_result.append(evaluate_entity(output, dual_X, test_SKG, list(all_SKG)))
    relation_pre_result.append(evaluate_relation(output, dual_X, test_SKG, list(all_SKG), r_num))

    print(entity_pre_result)
    print(relation_pre_result)
