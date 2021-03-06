
import argparse
import json
import logging
import os
import random
import time as Time
import numpy as np
import torch
import shutil
from torch.utils.data import DataLoader
import time
from model import KGEModel
import zipfile
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

class run():
    def __init__(self,isCUDA,savapath,id,last_time):
        self.isCUDA = isCUDA
        self.savepath = savapath
        self.id = id
        self.last_time = last_time


    def save_model(self,model, optimizer, time):

        save_path = "./result/model_"+str(time)+".pth"
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'time':time}

        torch.save(state, save_path)

    def load_model(self,time):
        model_path = "./result/model_"+str(time)+".pth"
        checkpoint = torch.load(model_path)

        model = KGEModel(
            nentity=self.nentity,
            nrelation=self.nrelation,
            hidden_dim=800,
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
        #self.kge_model, self.optimizer = self.load_model(time)
        if self.isCUDA==1:
            self.kge_model = self.kge_model.cuda()

        metrics,predictresult,score = self.kge_model.test_step(self.kge_model, valid_triples, all_true_triples,len(entity2id),len(relation2id),self.isCUDA)
        if not os.path.exists(self.savepath+'result_'+str(self.id)+'/'):
            os.mkdir(self.savepath+'result_'+str(self.id)+'/')
        file = open(self.savepath+'result_'+str(self.id)+'/predict'+str(time)+'.txt','w');
        for qua in predictresult:
            for i in range(40):
            #for word in qua:
                file.write(str(qua[i])+"\t")
            file.write("\n")
        file.close()
        if time == self.last_time:
            self.zipresult()
        evaluation_head = metrics[0]
        evaluation_tail = metrics[1]
        return evaluation_head,evaluation_tail

    def zipresult(self):
            src_dir = 'result/result_' + str(self.id)
            zip_name = str(self.id) + '_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.zip'
            z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
            for dirpath, dirnames, filenames in os.walk(src_dir):
                fpath = dirpath.replace(src_dir, '')
                fpath = fpath and fpath + os.sep or ''
                for filename in filenames:
                    z.write(os.path.join(dirpath, filename), fpath + filename)
            z.close()
            shutil.move(zip_name, 'static/finish/')

    def train(self,train_triples,time,entity2id,relation2id):
        self.nentity = len(entity2id)
        self.nrelation = len(relation2id)

        if time == -1:
            self.kge_model = KGEModel(
                nentity=self.nentity,
                nrelation=self.nrelation,
                hidden_dim=100,
                gamma=24.0,
            )
            current_learning_rate = 0.0001
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.kge_model.parameters()),
                lr=current_learning_rate
            )
        #else:
        #    temp = time - 1
        #    self.kge_model,self.optimizer = self.load_model(temp)

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nrelation, 256, 'head-batch'),
            batch_size=64,
            shuffle=True,
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nrelation, 256, 'tail-batch'),
            batch_size=64,
            shuffle=True,
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
        warm_up_steps = 5000 // 2
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)


        if self.isCUDA == 1:
            self.kge_model = self.kge_model.cuda()

        #start training
        print("start training:%d"%time)
        init_step = 0
        # Training Loop
        starttime = Time.time()
        if time==-1:
            steps =1
            printnum = 1000
        else:
            steps = 1
            printnum = 50
        for step in range(init_step, steps):
            loss = self.kge_model.train_step(self.kge_model, self.optimizer, train_iterator,self.isCUDA)
            '''
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                self.optimizer = torch.optim.Adam(
                   filter(lambda p: p.requires_grad, self.kge_model.parameters()),
                   lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            '''
            if step%printnum==0:
                endtime = Time.time()
                print("step:%d, cost time: %s, loss is %.4f" % (step,round((endtime - starttime), 3),loss))

                #self.save_model(self.kge_model, self.optimizer, time)
                #result_head, result_tail = self.evaluate(valid_triples, valid_triples, relation2id, entity2id, time)
                #print(result_head)
                #print(result_tail)
                #self.kge_model, self.optimizer = self.load_model(time)

        '''
        if time == -1:
            self.save_model(self.kge_model, self.optimizer, time)
        '''
