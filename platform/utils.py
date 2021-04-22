import numpy as np
import os
import torch

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])

def load_TKG(inPath, fileName):
    entity_num, rel_num, time_unit = get_total_number(inPath, 'stat.txt')
    with open(os.path.join(inPath, fileName), 'r') as fr:
        TKG={}
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3]) // time_unit
            if time not in TKG.keys():
                TKG[time]=[(head, rel, tail)]
            else:
                TKG[time].append((head, rel, tail))
        return TKG, time+1

def distance_func(head_emb, tail_emb, rel_emb):
    if head_emb.dim()==2:
        diff = torch.abs(torch.cat((head_emb, tail_emb), 1) - rel_emb)
        dis1 = diff.sum(axis=1)
        dimension = head_emb.shape[-1]
        dis2 =torch.abs(diff[:,:dimension]-diff[:,dimension:]).sum(axis=1)
        return dis1+0.2*dis2
        #return dis1
    else:
        diff = torch.abs(torch.cat((head_emb, tail_emb)) - rel_emb)
        dis1 = diff.sum()
        dimension = head_emb.shape[-1]
        dis2 = torch.abs(diff[:dimension] - diff[dimension:]).sum()
        return dis1 + 0.2 * dis2
        #return dis1

def get_kg_has_all_rel(TKG, r_num, last_time):
    max_rel=0
    max_ind=0
    rel_nums=[]
    for i in range(last_time):
        r_set=set()
        for tri in TKG[i]:
            r_set.add(tri[1])
        if len(r_set)==r_num:
            print("get dual_A from TKG: ",i)
            return TKG[i]
        rel_nums.append(len(r_set))
    return TKG[0]

def set_partition(TKG, train_prop, valid_prop):
    train_TKG={}
    valid_TKG={}
    test_TKG={}
    for time in TKG.keys():
        np.random.shuffle(TKG[time])
        num = len(TKG[time])
        train_TKG[time] = TKG[time][0:int(num*train_prop)]
        valid_TKG[time] = TKG[time][int(num*train_prop):int(num*(train_prop+valid_prop))]
        test_TKG[time] = TKG[time][int(num*(train_prop+valid_prop)):]
    return train_TKG, valid_TKG, test_TKG

def get_static_graph(TKG, start=None, seq_len=None):
    if start==None:
        start=0
    if seq_len==None:
        seq_len=max(TKG.keys())
    SKG=set()
    for time in TKG.keys():
        if time >= start and time < start + seq_len:
            for tri in TKG[time]:
                SKG.add(tri)
    return SKG

def load_static_graph(inPath, fileName, seq_len=5, start=0):
    entity_num, rel_num, time_unit = get_total_number(inPath, 'stat.txt')
    with open(os.path.join(inPath, fileName), 'r') as fr:
        SKG=set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3]) / time_unit
            if time == start + seq_len:
                break
            if time >= start:
                SKG.add((head, rel, tail))
    return list(SKG)

def load_static_file(inPath, fileName):
    entity_num=0
    rel_num=0
    with open(os.path.join(inPath, fileName), 'r') as fr:
        SKG=[]
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            if head > entity_num:
                entity_num=head
            if tail > entity_num:
                entity_num=tail
            if rel > rel_num:
                rel_num=rel
            SKG.append((head, rel, tail))
    return SKG, entity_num+1, rel_num+1
'''
def load_static_graph(inPath, fileName, seq_len):
    entity_num, rel_num, time_unit = get_total_number(inPath, 'stat.txt')
    with open(os.path.join(inPath, fileName), 'r') as fr:
        SKG=set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3]) / time_unit
            if time == seq_len:
                break
            SKG.add((head, rel, tail))
    return list(SKG)
'''
def get_triplets_tim(inPath, fileName, time_index):
    print("loading triplets at timing index ",time_index)
    tri_tim = []
    t = []
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            time = int(line_split[3])
            if time not in t:
                if len(t) == time_index+1:
                    return tri_tim, t
                elif len(t) == time_index:
                    t.append(time)
                    tri_tim.append(line_split[:3])
                else:
                    t.append(time)
            elif len(t) == time_index+1:
                tri_tim.append(line_split[:3])
        return tri_tim, t

def load_enorrel2id(inPath,fileName):
    entity2id = []
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            entity = line_split[0]
            entityNo = line_split[1]
            entity2id.append((entity,entityNo))

    return entity2id