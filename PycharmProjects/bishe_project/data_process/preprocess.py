from numpy import *
import operator
from os import listdir


data_path = 'raw_data/'
save_path = 'result_data/'
end_path = 'data/'



#读取产业链节点名
industry_node = []
industry_name = ['大数据','化工新材料','集成电路','人工智能','物联网','半导体材料','石墨烯','超导材料','生物制药','医疗器械']
import xlrd
def get_industry():
    data = []
    wb = xlrd.open_workbook(data_path+'10条产业链层级表汇总.xlsx')# 打开Excel文件
    for table in industry_name:
        sheet = wb.sheet_by_name(table)
        for a in range(sheet.ncols):
            cells = sheet.col_values(a)  # 每列数据赋值给cells
            if cells[0] == '节点公司':
                break
            for i in range(1,len(cells)):
                if len(cells[i])!=0:
                    data.append(cells[i])  # 把每次循环读取的数据插入到list
    return data
industry_node = get_industry()

def keyword(text):
    keyword = []
    for node in industry_node:
        if node in text:
            keyword.append(node)
    keywords = list(set(keyword))
    return keywords
'''
#company
company_quad = []
fr = open(data_path+'company.txt')
numberOfLines = len(fr.readlines())
print('the number of company:'+str(numberOfLines))
fr = open(data_path+'company.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if len(listFromLine)!=5:
        continue
    public_date = listFromLine[2][1:5]

    for node in keyword(listFromLine[3].strip('"')):
        main = []
        main.append(listFromLine[1].strip('"'))
        main.append('company_main')
        main.append(node)
        main.append(public_date)
        company_quad.append(main)

    for node in keyword(listFromLine[4].strip('"')):
        scope = []
        scope.append(listFromLine[1].strip('"'))
        scope.append('company_scope')
        scope.append(node)
        scope.append(public_date)
        company_quad.append(scope)

print(len(company_quad))
file = open(save_path+'company.txt','w');
for qua in company_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()


#zhy_patent
zhy_patent_quad = []
fr = open(data_path+'zhy_patent.txt')
numberOfLines = len(fr.readlines())
print('the number of zhy_patent:'+str(numberOfLines))
fr = open(data_path+'zhy_patent.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if len(listFromLine[2].strip('"'))!=8:
        continue
    public_date = listFromLine[2][1:5]

    name = []
    name.append(listFromLine[0].strip('"'))
    name.append('zhy_patent_name')
    name.append(listFromLine[1].strip('"'))
    name.append(public_date)
    zhy_patent_quad.append(name)

    if len(listFromLine)>3:
        for node in keyword(listFromLine[3].strip('"')):
            digest = []
            digest.append(listFromLine[0].strip('"'))
            digest.append('zhy_patent_digest')
            digest.append(node)
            digest.append(public_date)
            zhy_patent_quad.append(digest)

print(len(zhy_patent_quad))
file = open(save_path+'zhy_patent.txt','w');
for qua in zhy_patent_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#conference_paper
conference_paper_quad = []
fr = open(data_path+'wf_conference_paper.txt')
numberOfLines = len(fr.readlines())
print('the number of wf_conference_paper:'+str(numberOfLines))
fr = open(data_path+'wf_conference_paper.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[6][1:5]

    name = []
    name.append(listFromLine[0].strip('"'))
    name.append('conference_name')
    name.append(listFromLine[1].strip('"'))
    name.append(public_date)
    conference_paper_quad.append(name)

    mot_literature = []
    mot_literature.append(listFromLine[0].strip('"'))
    mot_literature.append('conference_mot_literature')
    mot_literature.append(listFromLine[3].strip('"'))
    mot_literature.append(public_date)
    conference_paper_quad.append(mot_literature)

    if len(listFromLine[4])!=0:
        meeting_name = []
        meeting_name.append(listFromLine[0].strip('"'))
        meeting_name.append('conference_meeting_name')
        meeting_name.append(listFromLine[4].strip('"'))
        meeting_name.append(public_date)
        conference_paper_quad.append(meeting_name)

    if len(listFromLine[5]) != 0:
        host_unit = []
        host_unit.append(listFromLine[0].strip('"'))
        host_unit.append('conference_host_unit')
        host_unit.append(listFromLine[5].strip('"'))
        host_unit.append(public_date)
        conference_paper_quad.append(host_unit)

    if len(listFromLine[2]) != 0:
        keywords = []
        keywords = listFromLine[2].strip('"').split('%')
        for key in keywords:
            temp = []
            temp.append(listFromLine[0])
            temp.append('conference_keyword')
            temp.append(key)
            temp.append(public_date)
            conference_paper_quad.append(temp)

print(len(conference_paper_quad))
file = open(save_path+'conference_paper.txt','w');
for qua in conference_paper_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close();

#journal_paper
journal_paper_quad = []
fr = open(data_path+'wf_journal_paper.txt')
numberOfLines = len(fr.readlines())
print('the number of wf_journal_paper:'+str(numberOfLines))
fr = open(data_path+'wf_journal_paper.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[3][1:5]

    name = []
    name.append(listFromLine[0].strip('"'))
    name.append('journal_name')
    name.append(listFromLine[1].strip('"'))
    name.append(public_date)
    journal_paper_quad.append(name)

    periodical_name = []
    periodical_name.append(listFromLine[0].strip('"'))
    periodical_name.append('journal_periodical')
    periodical_name.append(listFromLine[2].strip('"'))
    periodical_name.append(public_date)
    journal_paper_quad.append(periodical_name)

    if len(listFromLine) == 5:
        if len(listFromLine[4])!=0:
            keywords = []
            keywords = listFromLine[4].strip('"').split('%')
            for key in keywords:
                temp = []
                temp.append(listFromLine[0].strip('"'))
                temp.append('journal_keyword')
                temp.append(key)
                temp.append(public_date)
                journal_paper_quad.append(temp)

print(len(journal_paper_quad))
file = open(save_path+'journal_paper.txt','w');
for qua in journal_paper_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#wf_patent
wf_patent_quad = []
fr = open(data_path+'wf_patent.txt')
numberOfLines = len(fr.readlines())
print('the number of wf_patent:'+str(numberOfLines))
fr = open(data_path+'wf_patent.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[1][1:5]

    name = []
    name.append(listFromLine[0].strip('"'))
    name.append('wf_patent_name')
    name.append(listFromLine[2].strip('"'))
    name.append(public_date)
    wf_patent_quad.append(name)

    if (len(listFromLine) > 3):
        type_code_ys = []
        type_code_ys = listFromLine[3].strip('"').split('%')
        for type in type_code_ys:
            temp = []
            temp.append(listFromLine[0].strip('"'))
            temp.append('wf_patent_type_code_ys')
            temp.append(type)
            temp.append(public_date)
            wf_patent_quad.append(temp)

    if (len(listFromLine) > 4):
        for node in keyword(listFromLine[4].strip('"')):
            digest = []
            digest.append(listFromLine[0].strip('"'))
            digest.append('wf_patent_digest')
            digest.append(node)
            digest.append(public_date)
            wf_patent_quad.append(digest)

    if (len(listFromLine) > 5):
        for node in keyword(listFromLine[5].strip('"')):
            sovereignty = []
            sovereignty.append(listFromLine[0].strip('"'))
            sovereignty.append('wf_patent_sovereignty')
            sovereignty.append(node)
            sovereignty.append(public_date)
            wf_patent_quad.append(sovereignty)

    #这个理论上很好用，但是只有几个数据有这个字段，所以没法用了
    #if(len(listFromLine)==7):
    #    index2 = index2+1
    #    scope_type = []
    #    scope_type.append(listFromLine[0].strip('"'))
    #    scope_type.append('wf_patent_scope_type')
    #    scope_type.append(listFromLine[6].strip('"'))
    #    scope_type.append(public_date)
    #    wf_patent_quad.append(scope_type)

print(len(wf_patent_quad))
file = open(save_path+'wf_patent.txt','w');
for qua in wf_patent_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()
#company_conference
company_conference_quad = []
fr = open(data_path+'company-conference_paper.txt')
numberOfLines = len(fr.readlines())
print('the number of company_conference:'+str(numberOfLines))
fr = open(data_path+'company-conference_paper.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[3][1:5]

    company_conference = []
    company_conference.append(listFromLine[0].strip('"'))
    company_conference.append('company_conference')
    company_conference.append(listFromLine[2].strip('"'))
    company_conference.append(public_date)
    company_conference_quad.append(company_conference)

print(len(company_conference_quad))
file = open(save_path+'company_conference.txt','w');
for qua in company_conference_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#company_company
company_company_quad = []
fr = open(data_path+'company-investor_company.txt')
numberOfLines = len(fr.readlines())
print('the number of company_company:'+str(numberOfLines))
fr = open(data_path+'company-investor_company.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[2][1:5]

    company_company = []
    company_company.append(listFromLine[0].strip('"'))
    company_company.append('company_company')
    company_company.append(listFromLine[1].strip('"'))
    company_company.append(public_date)
    company_company_quad.append(company_company)

print(len(company_company_quad))
file = open(save_path+'company_company.txt','w');
for qua in company_company_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#company_journal
company_journal_quad = []
fr = open(data_path+'company-journal_paper.txt')
numberOfLines = len(fr.readlines())
print('the number of company_journal:'+str(numberOfLines))
fr = open(data_path+'company-journal_paper.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[3][1:5]

    company_journal = []
    company_journal.append(listFromLine[0].strip('"'))
    company_journal.append('company_journal')
    company_journal.append(listFromLine[2].strip('"'))
    company_journal.append(public_date)
    company_journal_quad.append(company_journal)

print(len(company_journal_quad))
file = open(save_path+'company_journal.txt','w');
for qua in company_journal_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#company_wf
company_wf_quad = []
fr = open(data_path+'company-wf_patents.txt')
numberOfLines = len(fr.readlines())
print('the number of company_wf:'+str(numberOfLines))
fr = open(data_path+'company-wf_patents.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[3][1:5]

    company_wf = []
    company_wf.append(listFromLine[0].strip('"'))
    company_wf.append('company_wf')
    company_wf.append(listFromLine[2].strip('"'))
    company_wf.append(public_date)
    company_wf_quad.append(company_wf)

print(len(company_wf_quad))
file = open(save_path+'company_wf.txt','w');
for qua in company_wf_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()

#company_zhy
company_zhy_quad = []
fr = open(data_path+'company-zhy_patent.txt')
numberOfLines = len(fr.readlines())
print('the number of company_zhy:'+str(numberOfLines))
fr = open(data_path+'company-zhy_patent.txt')
next(fr)
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    public_date = listFromLine[3][1:5]

    company_zhy = []
    company_zhy.append(listFromLine[0].strip('"'))
    company_zhy.append('company_zhy')
    company_zhy.append(listFromLine[2].strip('"'))
    company_zhy.append(public_date)
    company_zhy_quad.append(company_zhy)

print(len(company_zhy_quad))
file = open(save_path+'company_zhy.txt','w');
for qua in company_zhy_quad:
    for word in qua:
        file.write(str(word)+"\t")
    file.write("\n")
file.close()


#company_industry
import csv
csv_file=csv.reader(open(data_path+'company-industry.csv','r'))
content=[]
for line in csv_file:
    temp = []
    for word in line:
        if(len(word)!=0):
            temp.append(word.strip('"'))
    content.append(temp)
company_industry_quad = []
for line in content:
    company_industry = []
    company_industry.append(line[0])
    company_industry.append('company_industry')
    company_industry.append(line[len(line) - 1])
    company_industry.append(line[1][0:4])
    company_industry_quad.append(company_industry)

file = open(save_path+'company_industry.txt','w');
index = 0
for qua in company_industry_quad:
    if index>0:
        for word in qua:
            file.write(str(word)+"\t")
        file.write("\n")
    index = index+1
file.close()
'''


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