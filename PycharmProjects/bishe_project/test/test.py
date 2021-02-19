import argparse
import json
import logging
import os
import random
import time as Time
import numpy as np
import torch
from utils import *
from Config import Config
from run import *

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

class Test():
    def __init__(self,isCUDA):
        self.isCUDA = isCUDA

    def load_model(self,time):
        model_path = "./result/model_"+str(time)+".pth"
        checkpoint = torch.load(model_path)

        model = KGEModel(
            nentity=self.nentity,
            nrelation=self.nrelation,
            hidden_dim=1000,
            gamma=24.0,
        )
        model.load_state_dict(checkpoint['net'])
        current_learning_rate = 0.0001
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model,optimizer


    def evaluate(self,valid_triples,all_true_triples,relation2id,entity2id,time):
        
        self.nentity = len(entity2id)
        self.nrelation = len(relation2id)
        self.kge_model, self.optimizer = self.load_model(time)
        if self.isCUDA==1:
            self.kge_model = self.kge_model.cuda()

        metrics = self.kge_model.test_step(self.kge_model, valid_triples, all_true_triples,len(entity2id),len(relation2id),self.isCUDA)
        evaluation_head = metrics[0]
        evaluation_tail = metrics[1]
        return evaluation_head,evaluation_tail


if __name__ == '__main__':
    e, r_num, time_unit = get_total_number(Config.data, 'stat.txt')  # 得到知识图谱的实体数目，关系数目，时间单元（默认是1）

    train_TKG, last_time = load_TKG(Config.data, 'train_quadruples.txt')  # 那个字典接收知识图谱，然后记录最后的时间戳  #动态知识图谱
    valid_TKG, last_time = load_TKG(Config.data, 'valid_quadruples.txt')
    test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
    # 时序知识图谱词典，TKG[time]为在time时的（头实体，关系，尾实体）的list; last_time为最后时间

    # 完全不考虑时间，就是三元组的静态知识图谱
    train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', last_time, 0)
    test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', last_time, 0)
    valid_SKG = load_static_graph(Config.data, 'valid_quadruples.txt', last_time, 0)
    # 将TKG转化为静态知识图谱SKG，格式为（头实体，关系，尾实体）

    head_pre_result = []  # 头存储实体预测结果
    tail_pre_result = []  # 尾存储实体预测结果
    relation_pre_result = []  # 存储关系预测结果
    test_num = 0  # 测试的事实的数目

    entity2id = load_enorrel2id(Config.data, 'entity2id.txt')
    relation2id = load_enorrel2id(Config.data, 'relation2id.txt')

    # result 测试结果：hits@1，hits@3, hits@10, mrr       'HITS@1' 'HITS@3' 'HITS@10' 'MRR'
    # 先在SKG上评价，作为baseline



    Test = Test(1)
    print("start testing")
    for time in range(0, last_time):
        if time in test_TKG and time in valid_TKG:
            print(time)
            evaluation_head, evaluation_tail = Test.evaluate(test_TKG[time],
                                                             train_TKG[time] + valid_TKG[time] + test_TKG[time],
                                                             relation2id, entity2id, time)

            result = [evaluation_head['HITS@1'], evaluation_head['HITS@3'], evaluation_head['HITS@10'],
                      evaluation_head['MRR']]
            for count in range(0, len(test_TKG[time])):
                head_pre_result.append(result)

            result = [evaluation_tail['HITS@1'], evaluation_tail['HITS@3'], evaluation_tail['HITS@10'],
                      evaluation_tail['MRR']]
            for count in range(0, len(test_TKG[time])):
                tail_pre_result.append(result)

            # result = [evaluation_relation['HITS@1'], evaluation_relation['HITS@3'], evaluation_relation['HITS@10'],
            #          evaluation_relation['MRR']]
            # relation_pre_result.append(elm * len(test_TKG) for elm in result)

            # head_pre_result.append(elm*len (test_TKG) for elm in evaluation_head)#实体预测结果，evaluate_entity()应返回[hits@1，hits@3, hits@10, mrr]
            # tail_pre_result.append(elm * len(test_TKG) for elm in evaluation_tail)
            # relation_pre_result.append(elm*len (test_TKG) for elm in evaluation_relation)
            test_num += len(test_TKG[time])

    print(np.sum(head_pre_result, axis=0) / test_num)  # 每个时间戳评价的加权平均值
    print(np.sum(tail_pre_result, axis=0) / test_num)
    #print(np.sum(relation_pre_result, axis=0) / test_num)
    # 整个TKG的评价结果

	







