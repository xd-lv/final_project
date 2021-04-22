import torch
import numpy as np
from include.Config import Config
from include.RDGCN3 import *
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
    valid_TKG, last_time = load_TKG(Config.data, 'valid_quadruples.txt')
    test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
    train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', last_time, 0)
    test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', last_time, 0)
    valid_SKG = load_static_graph(Config.data, 'valid_quadruples.txt', last_time, 0)
    all_SKG=set(train_SKG)|set(test_SKG)|set(valid_SKG)
    #train_TKG, valid_TKG, test_TKG = set_partition(TKG, 0.7, 0.1)

    KG_for_dual = get_kg_has_all_rel(train_TKG, r_num, last_time)

    model = RDGCN_layer(Config.dim, Config.dim*2).cuda()
    #model = torch.load(Config.data+'models/best_only_static.chkpnt', map_location=torch.device('cuda'))
    with open(file='C', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
    #embedding_list = get_input_embeddings(Config.data, 'vectorList_TransE_dim150.json')
    #embedding_list = get_input_embeddings(Config.data, 'vectorList_WIKI_RDGCN.json')
    #input_embedding = model.get_input_layer(embedding_list)

    input_embedding = model.ent_embeddings

    entity_pre_result=[]
    relation_pre_result=[]
    test_num=0
    for time in range(0, last_time):
    #for time in range(last_time):
        print("time: ",time)
        KG = train_TKG[time]
        print("triples: ", len(KG))
        training(model, input_embedding, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.num, Config.k, e, r_num, KG, all_SKG, 0.001, Config.epochs, valid_TKG[time])
        model = torch.load(Config.data+'models/best_only.chkpnt', map_location=torch.device('cuda'))
        output, dual_X = model.output, model.dual_X
        all_kg = list(set(train_TKG[time])|set(test_TKG[time])|set(valid_TKG[time]))
        entity_pre_result.append([elm*len(test_TKG[time]) for elm in evaluate_entity(output, dual_X, test_TKG[time], all_kg)])
        relation_pre_result.append([elm*len(test_TKG[time]) for elm in evaluate_relation(output, dual_X, test_TKG[time], all_kg, r_num)])
        test_num+=len(test_TKG[time])
        #input_embedding=output
        #model = RDGCN_layer(Config.dim).cuda()
    #print(entity_pre_result)
    #print(relation_pre_result)
    print(np.sum(entity_pre_result, axis=0)/test_num)
    print(np.sum(relation_pre_result, axis=0)/test_num)
