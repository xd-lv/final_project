import re
import time
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TransE import TransE
from readTrainingData import readData
from generatePosAndCorBatch import generateBatches, dataset
from utils import *

torch.cuda.set_device(3)

class trainTransE:

    def __init__(self):
        self.inAdd = "./data/gdelt/"
        self.outAdd = "./data/gdelt/"
        self.preAdd = "./data/gdelt/"
        self.preOrNot = False
        self.entityDimension = 300
        self.relationDimension = 300
        self.numOfEpochs = 1000
        self.outputFreq = 50
        self.numOfBatches = 100
        self.learningRate = 0.01  # 0.01
        self.weight_decay = 0.001  # 0.005  0.02
        self.margin = 1.0
        self.norm = 2#
        self.top = 10
        self.patience = 10
        self.earlyStopPatience = 5
        self.bestAvFiMR = None

        self.train2id = {}
        self.trainTriple = None

        self.nums = [0, 0, 0]
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        self.headRelation2Tail = {}
        self.tailRelation2Head = {}
        self.positiveBatch = {}
        self.corruptedBatch = {}
        self.entityEmbedding = None
        self.relationEmbedding = None

        self.validate2id = {}
        self.validateHead = None
        self.validateRelation = None
        self.validateTail = None
        self.numOfValidateTriple = 0

        self.test2id = {}
        self.testHead = None
        self.testRelation = None
        self.testTail = None
        self.numOfTestTriple = 0
        self.transE = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:3")
        else:
            self.device = torch.device("cpu")

        #self.start()
        #self.train()
        #self.end()



    def start(self):
        print("-----Training Started at " + time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time())) + "-----")
        print("input address: " + self.inAdd)
        print("output address: " +self.outAdd)
        print("entity dimension: " + str(self.entityDimension))
        print("relation dimension: " + str(self.relationDimension))
        print("number of epochs: " + str(self.numOfEpochs))
        print("output training results every " + str(self.outputFreq) + " epochs")
        print("number of batches: " + str(self.numOfBatches))
        print("learning rate: " + str(self.learningRate))
        print("weight decay: " + str(self.weight_decay))
        print("margin: " + str(self.margin))
        print("norm: " + str(self.norm))
        print("is a continued learning: " + str(self.preOrNot))
        if self.preOrNot:
            print("pre-trained result address: " + self.preAdd)
        print("device: " + str(self.device))
        print("patience: " + str(self.patience))
        print("early stop patience: " + str(self.earlyStopPatience))

    def end(self):
        print("-----Training Finished at " + time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time())) + "-----")

    def train(self, start, seq_len):
        read = readData(self.inAdd, self.train2id, self.headRelation2Tail, self.tailRelation2Head, self.nums, start, seq_len)
        self.trainTriple = read.out()
        self.numOfTriple = self.nums[0]
        self.numOfEntity = self.nums[1]
        self.numOfRelation = self.nums[2]

        self.readValidateTriples(start, seq_len)
        #self.readTestTriples()
        if self.transE==None:
            self.transE = TransE(self.numOfEntity, self.numOfRelation, self.entityDimension, self.relationDimension, self.margin,
                        self.norm)

        if self.preOrNot:
            self.preRead(self.transE)

        self.transE.to(self.device)

        if self.numOfValidateTriple > 0:
            self.bestAvFiMR = self.validate(self.transE)
        self.entityEmbedding = self.transE.entity_embeddings.weight.data.clone()
        self.relationEmbedding = self.transE.relation_embeddings.weight.data.clone()

        criterion = nn.MarginRankingLoss(self.margin, False).to(self.device)
        optimizer = optim.SGD(self.transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)

        dataSet = dataset(self.numOfTriple)
        batchSize = int(self.numOfTriple / self.numOfBatches)
        if batchSize==0:
            batchSize=1
        dataLoader = DataLoader(dataSet, batchSize, True)

        patienceCount = 0

        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            for batch in dataLoader:
                self.positiveBatch = {}
                self.corruptedBatch = {}
                generateBatches(batch, self.train2id, self.positiveBatch, self.corruptedBatch, self.numOfEntity,
                                self.headRelation2Tail, self.tailRelation2Head)
                optimizer.zero_grad()
                positiveBatchHead = self.positiveBatch["h"].to(self.device)
                positiveBatchRelation = self.positiveBatch["r"].to(self.device)
                positiveBatchTail = self.positiveBatch["t"].to(self.device)
                corruptedBatchHead = self.corruptedBatch["h"].to(self.device)
                corruptedBatchRelation = self.corruptedBatch["r"].to(self.device)
                corruptedBatchTail = self.corruptedBatch["t"].to(self.device)
                output = self.transE(positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead,
                                   corruptedBatchRelation, corruptedBatchTail)
                positiveLoss = output.view(2, -1)[0]
                negativeLoss = output.view(2, -1)[1]
                tmpTensor = torch.tensor([-1], dtype=torch.float).to(self.device)
                batchLoss = criterion(positiveLoss, negativeLoss, tmpTensor)
                batchLoss.backward()
                optimizer.step()
                epochLoss += batchLoss

            print("epoch " + str(epoch) + ": , loss: " + str(epochLoss))

            if self.numOfValidateTriple==0 and epoch < 9:#该时间戳没有验证集时，只跑10轮
                continue
            if self.numOfValidateTriple==0 and epoch==9:
                break

            tmpAvFiMR = self.validate(self.transE)
            print("validate MR:", tmpAvFiMR)

            if tmpAvFiMR < self.bestAvFiMR:
                print("best averaged raw mean rank: " + str(self.bestAvFiMR) + " -> " + str(tmpAvFiMR))
                patienceCount = 0
                self.bestAvFiMR = tmpAvFiMR
                self.entityEmbedding = self.transE.entity_embeddings.weight.data.clone()
                self.relationEmbedding = self.transE.relation_embeddings.weight.data.clone()
            else:
                patienceCount += 1
                print("early stop patience: " + str(self.earlyStopPatience) + ", patience count: " + str(patienceCount) + ", current rank: " + str(tmpAvFiMR) + ", best rank: " + str(self.bestAvFiMR))
                if patienceCount == self.patience:
                    if self.earlyStopPatience == 1:
                        break
                    print("learning rate: " + str(self.learningRate) + " -> " + str(self.learningRate / 2))
                    print("weight decay: " + str(self.weight_decay) + " -> " + str(self.weight_decay * 2))
                    self.learningRate = self.learningRate/2
                    self.weight_decay = self.weight_decay*2
                    self.transE.entity_embeddings.weight.data = self.entityEmbedding.clone()
                    self.transE.relation_embeddings.weight.data = self.relationEmbedding.clone()
                    optimizer = optim.SGD(self.transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)
                    patienceCount = 0
                    self.earlyStopPatience -= 1

            #if (epoch+1)%self.outputFreq == 0 or (epoch+1) == self.numOfEpochs:
            #    self.write()
            print("")


        self.transE.entity_embeddings.weight.data = self.entityEmbedding.clone()
        self.transE.relation_embeddings.weight.data = self.relationEmbedding.clone()
        entity_embedding_list = self.entityEmbedding.cpu().detach().numpy().tolist()
        #with open(self.outAdd+'vectorList.json','w') as f:
        #    json.dump(entity_embedding_list, f)
        #self.test(transE)
        # self.fastTest(transE)

    def validate(self, transE):
        meanRank = 0
        for tmpTriple in range(self.numOfValidateTriple):
            rank = transE.fastValidate(self.validateHead[tmpTriple], self.validateRelation[tmpTriple], self.validateTail[tmpTriple])
            meanRank += rank
            #print("validate: ", rank)
            #Ranks=[0, 0]
            #print("test:" ,transE.test(Ranks, self.validateHead[tmpTriple], self.validateRelation[tmpTriple], self.validateTail[tmpTriple], self.trainTriple.to(self.device)))
            #print()
        return meanRank/self.numOfValidateTriple

    def fastTest(self, transE):  # Massive memory is required
        print("-----Fast Test Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
        meanRank = torch.tensor([0., 0.]).to(self.device)
        transE.fastTest(meanRank, self.testHead, self.testRelation, self.testTail,
                    self.trainTriple.to(self.device), self.numOfTestTriple)
        print("-----Result of Link Prediction (Raw)-----")
        print("|  Mean Rank  |  Filter@" + str(self.top) + "  |")
        print("|  " + str(meanRank[0]) + "  |  under implementing  |")
        print("-----Result of Link Prediction (Filter)-----")
        print("|  Mean Rank  |  Filter@" + str(self.top) + "  |")
        print("|  " + str(meanRank[1]) + "  |  under implementing  |")
        print("-----Fast Test Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")

    def test_entity_predict(self):
        print("-----Test Entity_predict Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
        Ranks = torch.tensor([0., 0.]).to(self.device)

        head_ranks, tail_ranks=[], []
        head_hits_1, tail_hits_1, head_hits_3, tail_hits_3, head_hits_10, tail_hits_10=[], [], [], [], [], []
        for tmpTriple in range(self.numOfTestTriple):
            if (tmpTriple+1)%100 == 0:
                print(str(tmpTriple+1) + " test triples processed!")
            ranks = self.transE.test_entity_predict(Ranks, self.testHead[tmpTriple], self.testRelation[tmpTriple], self.testTail[tmpTriple], self.trainTriple.to(self.device))
            #print(ranks)
            head_ranks.append(ranks[0])
            tail_ranks.append(ranks[1])
            head_hits_1.append(ranks[0] <= 1)
            tail_hits_1.append(ranks[1] <= 1)
            head_hits_3.append(ranks[0] <= 3)
            tail_hits_3.append(ranks[1] <= 3)
            head_hits_10.append(ranks[0] <= 10)
            tail_hits_10.append(ranks[1] <= 10)

        #print(head_ranks)
        head_hits_1_prop = sum(head_hits_1) / len(head_hits_1)
        head_hits_3_prop = sum(head_hits_3) / len(head_hits_3)
        head_hits_10_prop = sum(head_hits_10) / len(head_hits_10)
        head_MRR = np.mean([1/a for a in head_ranks])
        head_MR = np.mean(head_ranks)

        tail_hits_1_prop = sum(tail_hits_1) / len(tail_hits_1)
        tail_hits_3_prop = sum(tail_hits_3) / len(tail_hits_3)
        tail_hits_10_prop = sum(tail_hits_10) / len(tail_hits_10)
        tail_MRR = np.mean([1 / a for a in tail_ranks])
        tail_MR = np.mean(tail_ranks)

        print("head_hits_1_prop: %f, head_hits_3_prop: %f, head_hits_10_prop: %f, head_MRR: %f, head_MR: %f" % (head_hits_1_prop, head_hits_3_prop, head_hits_10_prop, head_MRR, head_MR))
        print("tail_hits_1_prop: %f, tail_hits_3_prop: %f, tail_hits_10_prop: %f, tail_MRR: %f, tail_MR: %f" % (tail_hits_1_prop, tail_hits_3_prop, tail_hits_10_prop, tail_MRR, tail_MR))
        return [head_hits_1_prop, tail_hits_1_prop, head_hits_3_prop, tail_hits_3_prop, head_hits_10_prop, tail_hits_10_prop, head_MR, tail_MR, head_MRR, tail_MRR]

    def test_relation_predict(self):
        print("-----Test Relation_predict Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")

        ranks = []
        hits_1= []
        for tmpTriple in range(self.numOfTestTriple):
            if (tmpTriple + 1) % 100 == 0:
                print(str(tmpTriple + 1) + " test triples processed!")
            rank = self.transE.test_relation_predict(self.testHead[tmpTriple], self.testRelation[tmpTriple],
                                                    self.testTail[tmpTriple], self.trainTriple.to(self.device))
            #print(rank)
            ranks.append(rank)
            hits_1.append(rank <= 1)

        # print(head_ranks)
        hits_1_prop = sum(hits_1) / len(hits_1)
        MR = np.mean(ranks)


        print("hits_1_prop: %f, MR: %f" % (hits_1_prop, MR))

        return [hits_1_prop, MR]

        '''    rMR += Ranks[0]
            fMR += Ranks[1]
            if Ranks[0] <= self.top:
                rHit += 1
            if Ranks[1] <= self.top:
                fHit += 1
        print("-----Result of Link Prediction (Head Filter)-----")
        print("|  Head Mean Rank  |  Filter@" + str(self.top) + "  |")
        print("|  " + str(rMR/self.numOfTestTriple) + "  |  " + str(rHit/self.numOfTestTriple) + "  |")
        print("-----Result of Link Prediction (Tail Filter)-----")
        print("|  Tail Mean Rank  |  Filter@" + str(self.top) + "  |")
        print("|  " + str(fMR/self.numOfTestTriple) + "  |  " + str(fHit/self.numOfTestTriple) + "  |")
        print("-----Test Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")'''

    def write(self):
        print("-----Writing Training Results to " + self.outAdd + "-----")
        entity2vecAdd = self.outAdd + "/entity2vec.pickle"
        relation2vecAdd = self.outAdd + "/relation2vec.pickle"
        entityOutput = open(entity2vecAdd, "wb")
        relationOutput = open(relation2vecAdd, "wb")
        pickle.dump(self.entityEmbedding, entityOutput)
        pickle.dump(self.relationEmbedding, relationOutput)
        entityOutput.close()
        relationOutput.close()

    def preRead(self, transE):
        print("-----Reading Pre-Trained Results from " + self.preAdd + "-----")
        entityInput = open(self.preAdd + "/entity2vec.pickle", "r")
        relationInput = open(self.preAdd + "/relation2vec.pickle", "r")
        tmpEntityEmbedding = pickle.load(entityInput)
        tmpRelationEmbedding = pickle.load(relationInput)
        entityInput.close()
        relationInput.close()
        transE.entity_embeddings.weight.data = tmpEntityEmbedding
        transE.relation_embeddings.weight.data = tmpRelationEmbedding

    def readTestTriples(self, start, seq_len):
        fileName = "test_quadruples.txt"
        print("-----Reading Test Triples from " + self.inAdd + fileName + "-----")
        count = 0
        self.test2id["h"] = []
        self.test2id["r"] = []
        self.test2id["t"] = []
        test_TKG, maxtime = load_TKG(self.inAdd, fileName)
        SKG = get_static_graph(test_TKG, start, seq_len)
        #SKG = load_static_test_graph(self.inAdd, fileName, seq_len, start)
        #inputData = open(self.inAdd + fileName)
        #line = inputData.readline()
        self.numOfTestTriple = len(SKG)
        #line = inputData.readline()
        for triple in SKG:
            tmpHead = int(triple[0])
            tmpTail = int(triple[2])
            tmpRelation = int(triple[1])
            self.test2id["h"].append(tmpHead)
            self.test2id["r"].append(tmpRelation)
            self.test2id["t"].append(tmpTail)
        self.testHead = torch.LongTensor(self.test2id["h"]).to(self.device)
        self.testRelation = torch.LongTensor(self.test2id["r"]).to(self.device)
        self.testTail = torch.LongTensor(self.test2id["t"]).to(self.device)

    def readValidateTriples(self, start, seq_len):
        fileName = "valid_quadruples.txt"
        print("-----Reading Validation Triples from " + self.inAdd + fileName + "-----")
        count = 0
        self.validate2id["h"] = []
        self.validate2id["r"] = []
        self.validate2id["t"] = []
        valid_TKG, maxtime = load_TKG(self.inAdd, fileName)
        SKG = get_static_graph(valid_TKG, start, seq_len)
        self.numOfValidateTriple = len(SKG)
        for triple in SKG:
            tmpHead = int(triple[0])
            tmpTail = int(triple[2])
            tmpRelation = int(triple[1])
            self.validate2id["h"].append(tmpHead)
            self.validate2id["r"].append(tmpRelation)
            self.validate2id["t"].append(tmpTail)
        self.validateHead = torch.LongTensor(self.validate2id["h"]).to(self.device)
        self.validateRelation = torch.LongTensor(self.validate2id["r"]).to(self.device)
        self.validateTail = torch.LongTensor(self.validate2id["t"]).to(self.device)

        '''
        inputData = open(self.inAdd + fileName)
        line = inputData.readline()
        self.numOfValidateTriple = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[1])
                tmpRelation = int(re.findall(r"\d+", line)[2])
                self.validate2id["h"].append(tmpHead)
                self.validate2id["r"].append(tmpRelation)
                self.validate2id["t"].append(tmpTail)
                count += 1
            else:
                print("error in " + fileName + " at Line " + str(count + 2))
            line = inputData.readline()
        inputData.close()
        if count == self.numOfValidateTriple:
            print("number of validation triples: " + str(self.numOfValidateTriple))
            self.validateHead = torch.LongTensor(self.validate2id["h"]).to(self.device)
            self.validateRelation = torch.LongTensor(self.validate2id["r"]).to(self.device)
            self.validateTail = torch.LongTensor(self.validate2id["t"]).to(self.device)
        else:
            print("count: " + str(count))
            print("expected number of validation triples: " + str(self.numOfValidateTriple))
            print("error in " + fileName)
        '''

if __name__ == '__main__':
    maxtime=366
    static_bound=300
    trainTransE = trainTransE()
    trainTransE.train(0, static_bound)
    trainTransE.readTestTriples(0, static_bound)

    #print(trainTransE.test_entity_predict())
    #print(trainTransE.test_relation_predict())


    entity_pre_result = []
    relation_pre_result = []
    test_num=0
    for t in range(0, maxtime):
        print("time: ", t)
        trainTransE.train(t, 1)
        trainTransE.readTestTriples(t, 1)
        if trainTransE.numOfTestTriple==0:
            continue
        entity_pre_result.append([elm*trainTransE.numOfTestTriple for elm in trainTransE.test_entity_predict()])
        relation_pre_result.append([elm*trainTransE.numOfTestTriple for elm in trainTransE.test_relation_predict()])
        test_num+=trainTransE.numOfTestTriple


    #trainTransE.readTestTriples(0, maxtime)
    #meanRank = trainTransE.transE.validate(trainTransE.numOfValidateTriple, trainTransE.validateHead, trainTransE.validateRelation, trainTransE.validateTail)
    #print("validate rank:",meanRank)
    #trainTransE.test()
    print(np.sum(entity_pre_result, axis=0)/test_num)
    print(np.sum(relation_pre_result, axis=0) / test_num)



