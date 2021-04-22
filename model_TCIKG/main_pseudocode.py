import torch
import numpy as np
from utils import *
from Config import Config
from run import *
import logging
import datetime

# torch.cuda.set_device(1)

if __name__ == '__main__':
    Config.data = 'data/'+'tech/'
    e, r_num, time_unit = get_total_number(Config.data, 'stat.txt')  # 得到知识图谱的实体数目，关系数目，时间单元（默认是1）

    train_TKG, last_time = load_TKG(Config.data, 'train_quadruples.txt')  # 那个字典接收知识图谱，然后记录最后的时间戳  #动态知识图谱
    #valid_TKG, last_time = load_TKG(Config.data, 'valid_quadruples.txt')
    test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
    print(last_time)
    # 时序知识图谱词典，TKG[time]为在time时的（头实体，关系，尾实体）的list; last_time为最后时间
    #print(Config.TIME)
    # 完全不考虑时间，就是三元组的静态知识图谱
    train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', last_time, 0)
    test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', last_time, 0)
    #valid_SKG = load_static_graph(Config.data, 'valid_quadruples.txt', last_time, 0)
    # 将TKG转化为静态知识图谱SKG，格式为（头实体，关系，尾实体）

    head_pre_result = []  # 头存储实体预测结果
    tail_pre_result = []  # 尾存储实体预测结果
    relation_pre_result = []  # 存储关系预测结果
    test_num = 0  # 测试的事实的数目
    print(Config.data)
    entity2id = load_enorrel2id(Config.data, 'entity2id.txt')
    relation2id = load_enorrel2id(Config.data, 'relation2id.txt')
    Run = run(1)
    Run.train(train_SKG, -1, entity2id, relation2id)  # 将模型在静态知识图谱上预训练,得到特征和初始参数。注，有些简单模型像TransE只有特征，没有参数；也有一些模型只调整参数，不改变输入特征
    s_evaluation_head, s_evaluation_tail = Run.evaluate(test_SKG,train_SKG + test_SKG,relation2id, entity2id, -1)

    print("head result:")
    print(s_evaluation_head)
    print("tail result:")
    print(s_evaluation_tail)
    # result 测试结果：hits@1，hits@3, hits@10, mrr       'HITS@1' 'HITS@3' 'HITS@10' 'MRR'
    # 先在SKG上评价，作为baseline

    for time in range(0, last_time):
        if time in test_TKG :
            temp_KG=[]
            #for count in range(0,time+1):
            #    temp_KG = temp_KG + train_TKG[count]
            #print(len(temp_KG))
            Run.train(train_TKG[time], time, entity2id, relation2id)  # 将模型拟合于第time个时间戳，即根据第time个时间戳调整模型参数和特征X

            evaluation_head, evaluation_tail = Run.evaluate(test_TKG[time],train_TKG[time] + test_TKG[time],relation2id, entity2id, time)

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

    #log_file = 'log/sys_%s.log' % datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
    #log_level = logging.debug
    #log_format = '%(asctime)s[%(levelname)s]: %(message)s'
    #logging.basicConfig(filename=log_file, level=logging.WARNING, format=log_format)
    #logger = logging.getLogger()
    #logger.info('head result:')
    #logger.info(np.sum(head_pre_result, axis=0) / test_num)
    #logger.info('tail result:')
    #logger.info(np.sum(tail_pre_result, axis=0) / test_num)
    #logger.warning('This is a warning message!')
    #logger.error('This is a error message!')
    #logger.critical('This is a critical message!')
    
    print("head result:")
    print(np.sum(head_pre_result, axis=0) / test_num)  # 每个时间戳评价的加权平均值

    print("tail result:")
    print(np.sum(tail_pre_result, axis=0) / test_num)
    #print(np.sum(relation_pre_result, axis=0) / test_num)
    # 整个TKG的评价结果


