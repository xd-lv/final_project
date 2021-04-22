'''
end_path = 'data_1/'
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

file = open(end_path+'all_quad.txt','w');
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
'''

fr = open('data_6/entity2id.txt')
for line in fr.readlines():
    line_split = line.split()
    print(line_split[1])

'''
fr = open('data_3/test_quadruples.txt')
newlist = []
for line in fr.readlines():
    line = line.strip()
    line_split = line.split('\t')
    if (int(line_split[2])<233):
        newlist.append(line_split)
file = open('data_3/test_quadruples.txt','w');
for line in newlist:
    for word in line:
        file.write(str(word) + "\t")
    file.write("\n")
file.close()
'''
'''
fr = open('clear_data/company_industry.txt')
temp = []
for line in fr.readlines():
    line_split = line.split()
    if line_split[2] not in temp:
        temp.append(line_split[2])
print(len(temp))

fr = open('data_1/test_quadruples.txt')
temp = []
for line in fr.readlines():
    line_split = line.split()
    if line_split[2] not in temp and line_split[1]=='15':
        temp.append(line_split[2])
print(len(temp))
'''
'''
company = []
fr = open('raw_data/all.txt')
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[10] not in company:
        company.append(listFromLine[10])
print(len(company))

fr = open('raw_data/company-industry.csv')
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[0] not in company:
        company.append(listFromLine[0])
print(len(company))

result = []
fr = open('result_data/company_industry.txt')
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[0] not in result:
        result.append(listFromLine[0])
print(len(result))

clear = []
fr = open('clear_data/company_industry.txt')
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[0] not in clear:
        clear.append(listFromLine[0])
print(len(clear))
'''


'''
fr = open('clear_data/company_industry.txt')
node = []
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[2] not in node:
        node.append(listFromLine[2])
print(len(node))
print(node)
fr = open('data_5/entity2id.txt')
entity = {}
count = 0
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    entity[listFromLine[0]] = listFromLine[1]
    count = count + 1
    if count>189:
        break
newnode = []
for element in node:
    newnode.append(entity[element])
print(len(newnode))
print(newnode)

#unlist_company_industry
company_industry_quad =[]
fr = open('raw_data/all.txt')
next(fr)
i = 0
for line in fr.readlines():
    i = i +1
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[0].strip('"')=='无人驾驶' or listFromLine[0].strip('"')=='半导体材料' or listFromLine[0].strip('"')=='医疗器械':
        continue
    split = []
    if len(listFromLine)!=14:
        continue
    for element in listFromLine:
        if len(element)!=0:
            split.append(element)
    if len(split) > 5:
        if split[1].strip('"') == '网络层':
            print(i)
        if split[1].strip('"') not in company_industry_quad:
            company_industry_quad.append(split[1].strip('"'))
    else:
        if split[0].strip('"') == '网络层':
            print(i)
        if split[0].strip('"') not in company_industry_quad:
            company_industry_quad.append(split[0].strip('"'))
print(len(company_industry_quad))
print(company_industry_quad)

shouldhave = []
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

shouldhave = get_industry()
print(len(shouldhave))
print(shouldhave)

if '中游设计' in shouldhave:
    print(11)

notin = []
for unlist in company_industry_quad:
    if unlist not in shouldhave:
        notin.append(unlist)
print(len(notin))
print(notin)
'''