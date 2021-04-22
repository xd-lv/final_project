

data_path = 'result_data/'
clear_path = 'clear_data/'
def get_qua(path):
    qua = []
    fr = open(path)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        temp = []
        for word in listFromLine:
            temp.append(word)
        qua.append(temp)
    return qua

def remove_repeat(qua):
    temp = []
    for line in qua:
        flag = 0
        for element in temp:
            if line[0]==element[0] and line[2]==element[2] and line[1]==element[1] and line[3]==element[3]:
                flag = 1
        if flag==0:
            temp.append(line)
    return temp

def save_clear(quas,path):
    file = open(path, 'w');
    for qua in quas:
        for word in qua:
            file.write(str(word) + "\t")
        file.write("\n")
    file.close()

def clear(filename):
    quas = get_qua(data_path + filename)
    print('initial '+filename)
    print(len(quas))
    new_quas = remove_repeat(quas)
    print('clear '+filename)
    print(len(new_quas))
    save_clear(new_quas, clear_path + filename)

import os.path

files = os.listdir('result_data/')
for file in files:
    clear(file)

