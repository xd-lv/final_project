
end_path = 'data_6/'
save_path = 'clear_data/'

relation=[]
entity = []
all_quad = []
time = []

num_time = 0
num_all_quad = 0
num_relation = 0
num_entity = 0

import os.path



#读取产业链点名
industry_node = []
industry_name = ['大数据','化工新材料','集成电路','人工智能','物联网','石墨烯','超导材料','生物制药']
import xlrd
def get_industry():
    data = []
    wb = xlrd.open_workbook('raw_data/10条产业链层级表汇总.xlsx')# 打开Excel文件
    for table in industry_name:
        sheet = wb.sheet_by_name(table)
        for a in range(2):
            cells = sheet.col_values(a)  # 每列数据赋值给cells
            #if cells[0] == '四级分类':
            #    break
            for i in range(1,len(cells)):
                if len(cells[i])!=0:
                    data.append(cells[i])  # 把每次循环读取的数据插入到list
    return data

entity = get_industry()
'''
fr = open(save_path+'/company_industry.txt')
entity = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[2] not in entity:
        entity.append(listFromLine[2])
print(len(entity))
'''

print(len(entity))

files = os.listdir(save_path)
for file in files:
    print(file)
    fr = open(save_path + file)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        all_quad.append(listFromLine)
        if listFromLine[0] not in entity:
            entity.append(listFromLine[0])
        if listFromLine[2] not in entity:
            entity.append(listFromLine[2])
        if listFromLine[1] not in relation:
            relation.append(listFromLine[1])
        if listFromLine[3] not in time:
            time.append(listFromLine[3])

print(len(all_quad))
print(len(entity))
print(len(relation))
print(len(time))
time.sort()
def list2set(list):
    temp = {}
    count = 0
    for element in list:
        temp[element] = str(count)
        count = count + 1
    return temp

entity_set = list2set(entity)
relation_set = list2set(relation)
time_set = list2set(time)
new_quas = []
for qua in all_quad:
    temp = []
    temp.append(entity_set[qua[0]])
    temp.append(relation_set[qua[1]])
    temp.append(entity_set[qua[2]])
    temp.append(time_set[qua[3]])
    new_quas.append(temp)


file = open(end_path+'train_quas.txt','w');
for qua in new_quas:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

file = open(end_path+'all_quas.txt','w');
for qua in all_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

file = open(end_path+'entity2id.txt','w');
index = 0
for line in entity:
    file.write(str(line)+"\t"+str(index))
    file.write("\n")
    index = index+1
file.close()

file = open(end_path+'relation2id.txt','w');
index = 0
for line in relation:
    file.write(str(line)+"\t"+str(index))
    file.write("\n")
    index = index+1
file.close()

file = open(end_path+'time2id.txt','w');
index = 0
for line in time:
    file.write(str(line)+"\t"+str(index))
    file.write("\n")
    index = index+1
file.close()

#test
from random import random

fr = open(end_path+'/train_quas.txt')
com_ind = []
other = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[1]=='12':
        com_ind.append(listFromLine)
    else:
        other.append(listFromLine)
print(len(com_ind))

testcompany = []
traincomapny =  []
temptest = []
for i in range(len(com_ind)):
    if i%2==0:
        temptest.append(com_ind[i])
    else:
        traincomapny.append(com_ind[i])
for i in range(len(temptest)):
    if i%2==0:
        testcompany.append(temptest[i])
    else:
        traincomapny.append(temptest[i])

file = open(end_path+'/train_quadruples.txt','w');
for qua in other:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
for element in traincomapny:
    for word in element:
        file.write(str(word) + "\t")
    file.write("\n")
file.close()

file = open(end_path+'/test_quadruples.txt','w');
for element in testcompany:
    for word in element:
        file.write(str(word) + "\t")
    file.write("\n")
file.close()


'''
#fr = open('data_1/train_quas.txt')
#num = len(fr.readlines())*0.8
#fr = open('data_1/train_quas.txt')
i=0
train = []
test = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    temp = []
    temp.append(listFromLine[0])
    temp.append(listFromLine[1])
    temp.append(listFromLine[2])
    temp.append(listFromLine[3])
    if i<num:
        train.append(temp)
    else:
        test.append(temp)
    i = i + 1

file = open(end_path+'train_quadruples.txt','w');
for qua in train:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

file = open(end_path+'test_quadruples.txt','w');
for qua in test:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()
'''

#sort
def sort(quas):
    sort_quas = []
    for i in range(48):
        for element in quas:
            if element[3]==str(i):
                sort_quas.append(element)
    return sort_quas

fr = open(end_path+'/test_quadruples.txt')
train = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    train.append(listFromLine)
new_train = sort(train)

file = open(end_path+'/test_quadruples.txt','w');
for qua in new_train:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()
fr = open(end_path+'/train_quadruples.txt')
train = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    train.append(listFromLine)
new_train = sort(train)

file = open(end_path+'/train_quadruples.txt','w');
for qua in new_train:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()
