

data_path = 'data/YAGO/'
fr = open(data_path+'train_quadruples.txt')
numberOfLines = len(fr.readlines())
print('the number of quadruples:'+str(numberOfLines))
fr = open(data_path+'train_quadruples.txt')
head = {}
tail = {}
for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split('\t')
    if listFromLine[0] not in head.keys():
        head[listFromLine[0]] = 1
    else:
        head[listFromLine[0]] = head[listFromLine[0]] + 1
    if listFromLine[2] not in tail.keys():
        tail[listFromLine[2]] = 1
    else:
        tail[listFromLine[2]] = tail[listFromLine[2]] + 1

num_head_1 = 0
num_tail_1 = 0
num_head_n = 0
num_tail_n = 0

for key in head:
    if head[key] == 1:
        num_head_1 = num_head_1+1
for key in tail:
    if tail[key] == 1:
        num_tail_1 = num_tail_1+1
for key in head:
    if head[key] > 10:
        num_head_n = num_head_n+1
for key in tail:
    if tail[key] > 10:
        num_tail_n = num_tail_n+1

print('head')
print(num_head_1)
print(num_head_n)
print(len(head))

print('tail')
print(num_tail_1)
print(num_tail_n)
print(len(tail))

print(head)
print(tail)