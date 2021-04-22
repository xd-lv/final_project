import torch
import numpy as np
from utils import *
from Config import Config
from run import *
import logging
import datetime

# torch.cuda.set_device(1)

class main_pseudocode():
    def __init__(self, id):
        self.datapath = self.getpath(id)
        self.savepath = 'result/'
        self.id = id
        print(self.datapath)

    def getpath(self,id):
        files = os.listdir('data/tech_'+str(id)+'/')
        return 'data/tech_'+str(id)+'/'+files[1]

    def model_train(self):
        if torch.cuda.is_available():
            cuda = 1
        else:
            cuda = 0
        Config.data = self.datapath
        e, r_num, time_unit = get_total_number(Config.data, 'stat.txt')

        train_TKG, last_time = load_TKG(Config.data, 'train_quadruples.txt')

        test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
        print(last_time)

        train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', last_time, 0)
        test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', last_time, 0)


        head_pre_result = []
        tail_pre_result = []

        test_num = 0
        print(Config.data)
        entity2id = load_enorrel2id(Config.data, 'entity2id.txt')
        relation2id = load_enorrel2id(Config.data, 'relation2id.txt')
        Run = run(cuda,self.savepath,self.id,last_time-1)
        Run.train(train_SKG, -1, entity2id,relation2id)
        s_evaluation_head, s_evaluation_tail = Run.evaluate(test_SKG, train_SKG + test_SKG, relation2id, entity2id, -1)

        print("head result:")
        print(s_evaluation_head)
        print("tail result:")
        print(s_evaluation_tail)


        for time in range(0, last_time):
            if time in test_TKG:
                Run.train(train_TKG[time], time, entity2id, relation2id)

                evaluation_head, evaluation_tail = Run.evaluate(test_TKG[time], train_TKG[time] + test_TKG[time],
                                                                relation2id, entity2id, time)

                result = [evaluation_head['HITS@1'], evaluation_head['HITS@3'], evaluation_head['HITS@10'],
                          evaluation_head['MRR']]
                print(result)
                for count in range(0, len(test_TKG[time])):
                    head_pre_result.append(result)

                result = [evaluation_tail['HITS@1'], evaluation_tail['HITS@3'], evaluation_tail['HITS@10'],
                          evaluation_tail['MRR']]
                print(result)
                for count in range(0, len(test_TKG[time])):
                    tail_pre_result.append(result)

                test_num += len(test_TKG[time])
                
                print("head result:")
                print(np.sum(head_pre_result, axis=0) / test_num)  

                print("tail result:")
                print(np.sum(tail_pre_result, axis=0) / test_num)










