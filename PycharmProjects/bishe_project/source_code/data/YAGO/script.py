import pandas as pd
import scipy.sparse as sp
import uuid, sys, os, time, argparse
from collections import defaultdict as ddict

def create_year2id(triple_time):
    year2id = dict()
    freq = ddict(int)
    count = 0
    year_list = []

    for k,v in triple_time.items():
        try:
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
        except:
            pdb.set_trace()

        if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
        if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))

    # for k,v in entity_time.items():
    # 	start = v[0].split('-')[0]
    # 	end = v[1].split('-')[0]
        
    # 	if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
    # 	if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))
    # 	# if int(start) > int(end):
    # 	# 	pdb.set_trace()
    
    year_list.sort()
    for year in year_list:
        freq[year] = freq[year] + 1

    year_class =[]
    count = 0
    for key in sorted(freq.keys()):
        count += freq[key]
        if count > 300:
            year_class.append(key)
            count = 0
    prev_year = 0
    i=0
    for i,yr in enumerate(year_class):
        year2id[(prev_year,yr)] = i
        prev_year = yr+1
    year2id[(prev_year, max(year_list))] = i + 1
    year_list =year_list

    # for k,v in entity_time.items():
    # 	if v[0] == '####-##-##' or v[1] == '####-##-##':
    # 		continue
    # 	if len(v[0].split('-')[0])!=4 or len(v[1].split('-')[0])!=4:
    # 		continue
    # 	start = v[0].split('-')[0]
    # 	end = v[1].split('-')[0]
    # for start in start_list:
    # 	if start not in start_year2id:
    # 		start_year2id[start] = count_start
    # 		count_start+=1

    # for end in end_list:
    # 	if end not in end_year2id:
    # 		end_year2id[end] = count_end
    # 		count_end+=1
    
    return year2id#每个键（prev_year, yr）代表一个时间戳平面，值i代表第i个时间戳

def create_id_labels(year2id,triple_time,dtype):
    YEARMAX = 3000
    YEARMIN =  -50
    
    inp_idx, start_idx, end_idx =[], [], []
    
    for k,v in triple_time.items():
        start = v[0].split('-')[0]
        end = v[1].split('-')[0]
        if start == '####':
            start = YEARMIN
        elif start.find('#') != -1 or len(start)!=4:
            continue

        if end == '####':
            end = YEARMAX
        elif end.find('#')!= -1 or len(end)!=4:
            continue
        
        start = int(start)
        end = int(end)
        
        if start > end:
            end = YEARMAX
        inp_idx.append(k)
        if start == YEARMIN:
            start_idx.append(0)
        else:
            for key,lbl in sorted(year2id.items(), key=lambda x:x[1]):
                if start >= key[0] and start <= key[1]:
                    start_idx.append(lbl)
        
        if end == YEARMAX:
            end_idx.append(len(year2id.keys())-1)
        else:
            for key,lbl in sorted(year2id.items(), key=lambda x:x[1]):
                if end >= key[0] and end <= key[1]:
                    end_idx.append(lbl)

    return inp_idx, start_idx, end_idx#为那些start或end不含#且年份大于100的三元组指定year的label；inp_idx保存这些三元组下标

def write_triplets():
    train_set = []
    train_time = dict()
    inp_idx, start_idx, end_idx ,labels = ddict(list), ddict(list), ddict(list), ddict(list)
    max_ent, max_rel, count = 0, 0, 0
    with open('train.txt') as filein:
        for line in filein:
            train_set.append([int(x.strip()) for x in line.split()[0:3]])
            train_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
            count+=1
    
    valid_set = []
    valid_time = dict()
    valid_inp_idx, valid_start_idx, valid_end_idx ,valid_labels = ddict(list), ddict(list), ddict(list), ddict(list)
    count = 0
    with open('valid.txt') as filein:
        for line in filein:
            valid_set.append([int(x.strip()) for x in line.split()[0:3]])
            valid_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
            count+=1
    
    test_set = []
    test_time = dict()
    test_inp_idx, test_start_idx, test_end_idx ,test_labels = ddict(list), ddict(list), ddict(list), ddict(list)
    count = 0
    with open('test.txt') as filein:
        for line in filein:
            test_set.append([int(x.strip()) for x in line.split()[0:3]])
            test_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
            count+=1
    

    with open('entity2id.txt','r', encoding="utf-8") as filein2:
        for line in filein2:
            # entity_time[int(line.split('\t')[1])]=[x.split()[0] for x in line.split()[2:4]]
            max_ent = max_ent+1
    
    year2id = create_year2id(train_time)
    num_class = len(year2id.keys())

    inp_idx['triple'], start_idx['triple'], end_idx['triple'] = create_id_labels(year2id, train_time,'triple')
    keep_idx = set(inp_idx['triple'])
    for i in range (len(train_set)-1,-1,-1):
        if i not in keep_idx:
            del train_set[i]
    
    valid_inp_idx['triple'], valid_start_idx['triple'], valid_end_idx['triple'] = create_id_labels(year2id, valid_time,'triple')
    keep_idx = set(valid_inp_idx['triple'])
    for i in range (len(valid_set)-1,-1,-1):
        if i not in keep_idx:
            del valid_set[i]

    test_inp_idx['triple'], test_start_idx['triple'], test_end_idx['triple'] = create_id_labels(year2id, test_time,'triple')
    keep_idx = set(test_inp_idx['triple'])
    for i in range (len(test_set)-1,-1,-1):
        if i not in keep_idx:
            del test_set[i]
    
    posh, rela, post = zip(*train_set)

    posh = list(posh) 
    post = list(post)
    rela = list(rela)

    head  =  [] 
    tail  =  []
    rel   =  []
    times = []

    for i in range(len(posh)):
        if start_idx['triple'][i] < end_idx['triple'][i]:
            for j in range(start_idx['triple'][i] + 1,end_idx['triple'][i] + 1):
                head.append(posh[i])
                rel.append(rela[i])
                tail.append(post[i])
                times.append(j-1)
    qua=zip(head, rel, tail, times)
    qua=sorted(qua, key=lambda x:x[3])
    with open('train_quadruples.txt', 'w') as fw:
        for line in qua:
            line_str = [str(s) for s in line]
            line_tab = '\t'.join(line_str)+'\n'
            fw.write(line_tab)
        fw.close()

    posh, rela, post = zip(*valid_set)

    posh = list(posh) 
    post = list(post)
    rela = list(rela)

    head  =  [] 
    tail  =  []
    rel   =  []
    times = []

    for i in range(len(posh)):
        if valid_start_idx['triple'][i] < valid_end_idx['triple'][i]:
            for j in range(valid_start_idx['triple'][i] + 1,valid_end_idx['triple'][i] + 1):
                head.append(posh[i])
                rel.append(rela[i])
                tail.append(post[i])
                times.append(j-1)
    qua=zip(head, rel, tail, times)
    qua=sorted(qua, key=lambda x:x[3])
    with open('valid_quadruples.txt', 'w') as fw:
        for line in qua:
            line_str = [str(s) for s in line]
            line_tab = '\t'.join(line_str)+'\n'
            fw.write(line_tab)
        fw.close()
    
    posh, rela, post = zip(*test_set)

    posh = list(posh) 
    post = list(post)
    rela = list(rela)

    head  =  [] 
    tail  =  []
    rel   =  []
    times = []

    for i in range(len(posh)):
        if test_start_idx['triple'][i] < test_end_idx['triple'][i]:
            for j in range(test_start_idx['triple'][i] + 1,test_end_idx['triple'][i] + 1):
                head.append(posh[i])
                rel.append(rela[i])
                tail.append(post[i])
                times.append(j-1)
    qua=zip(head, rel, tail, times)
    qua=sorted(qua, key=lambda x:x[3])
    with open('test_quadruples.txt', 'w') as fw:
        for line in qua:
            line_str = [str(s) for s in line]
            line_tab = '\t'.join(line_str)+'\n'
            fw.write(line_tab)
        fw.close()

if __name__== "__main__":
    write_triplets()