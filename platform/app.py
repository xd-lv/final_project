# coding=utf-8

import os
#from __init__ import app
from flask import jsonify
from flask import request, render_template
from os import path
import time
from flask_cors import CORS

import main_pseudocode
Basepath = os.path.abspath(os.path.dirname(__file__))
import os
import zipfile
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import _thread

from flask import Flask
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
#创建项目对象
executor = ThreadPoolExecutor(1)
app = Flask(__name__)
#app = Flask(__name__ ,static_folder='/tmp')
#app._static_folder = 'oldflaskproject/'
CORS(app, supports_credentials=True)



@app.route('/upload', methods=["POST"])
def getDataset():
    f = request.files["file"]
    # print(f.filename)
    base_path = path.abspath(path.dirname(__file__))
    upload_path = path.join(base_path, 'static/upload/')
    # print(upload_path)
    files = os.listdir('static/upload/')  # 读入文件夹
    num_png = len(files)
    # print(num_png)
    file_name = upload_path + str(num_png + 1) + '_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.zip'
    # print(file_name)
    f.save(file_name)
    # return redirect(url_for('index.upload'))
    # return redirect('upload')
    unzip(file_name,num_png+1)
    #executor.submit(training(num_png + 1))
    #training(num_png + 1)
    _thread.start_new_thread(training,( num_png + 1, ) )
    return str(num_png + 1)


def unzip(file,id):
    dst_dir = 'data/tech_'+str(id)
    r = zipfile.is_zipfile(file)
    if r:
        fz = zipfile.ZipFile(file, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')

def training(id):
    train = main_pseudocode.main_pseudocode(id)
    train.model_train()


@app.route('/result', methods=["GET", "POST"])
def resultlist():
    alltask = os.listdir('static/upload/')
    finishtask = os.listdir('static/finish/')
    # print(alltask)
    # print(finishtask)
    result = []
    for element in alltask:
        ttemp = element.split('.')[0]
        strlist = ttemp.split('_')
        tasknumber = strlist[0]
        starttime = strlist[1]
        finishtime, predictnum = isfinish(tasknumber, finishtask)

        temp = []
        temp.append(tasknumber)
        temp.append(starttime)
        temp.append(finishtime)
        temp.append(predictnum)
        result.append(temp)
    sortresult = []
    for i in range(len(alltask)):
        for element in result:
            if element[0] == str(i + 1):
                sortresult.append(element)
    # print(jsonify(sortresult))
    return jsonify(sortresult)


def isfinish(tasknumber, finishtask):
    flag = 0
    predictnum = 0
    time = ''
    for element in finishtask:
        if element.split('_')[0] == tasknumber:
            flag = 1
            time = element.split('_')[1].split('.')[0]
    if flag == 1:
        files = os.listdir('result/result_' + tasknumber + '/')
        for file in files:
            count = len(open(r'result/result_' + tasknumber + '/' + file, 'rU').readlines())
            predictnum = predictnum + count
    if flag == 0:
        time = 'running'
        predictnum = 'running'
    return time, predictnum


@app.route('/download', methods=["GET"])
def downloadfile():
    id = request.values.get("id")
    #id = request.args['id']
    #print(id)
    filename = ''
    finishtask = os.listdir('static/finish/')
    for name in finishtask:
        if name.split('_')[0] == id:
            filename = name
            break
    #return send_from_directory(r"static/finish/" + filename, filename=filename, as_attachment=True)
    url = '/static/finish/'+filename
    #print(filename)
    #print(url)
    return url





@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(port=5000)
