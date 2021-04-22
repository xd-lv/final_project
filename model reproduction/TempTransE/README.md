# TransE-Pytorch
An implementation of TransE* in Pytorch.

## Overview
Test results on FB15K with default parameters:

        -----Result of Link Prediction (Raw)-----
      |  Mean Rank  |  Filter@10  |
      |  tensor(353.5773, device='cuda:0')  |  0.101488039816  |
        -----Result of Link Prediction (Filter)-----
      |  Mean Rank  |  Filter@10  |
      |  tensor(276.4282, device='cuda:0')  |  0.171166900848  |

Better performance can be achieved by tunning the parameters.

* Bordes A, Usunier N, Garcia-Duran A, et al. Translating embeddings for modeling multi-relational data[C]//Advances in neural information processing systems. 2013: 2787-2795.


## Parameters

Please configure parameters in Train.py:

self.inAdd = "./data/FB15K"  # input address

self.outAdd = "./data/outputData"  # output address

self.preAdd = "./data/outputData"  # address of the existing pre-trained embeddings

self.preOrNot = False  # continue training based on the existing embeddings in self.preAdd or not

self.entityDimension = 100  # the dimension of entity embedding

self.relationDimension = 100  # the dimension of relation embedding

self.numOfEpochs = 1000  # number of epoch

self.outputFreq = 50  # output the learning results every self.outputFreq epoches

self.numOfBatches = 100  # the number of batches

self.learningRate = 0.01  # 0.01  # the learning rate of SGD optimizer

self.weight_decay = 0.001  # 0.005  0.02  #the weight decay of SGD optimizer

self.margin = 1.0  # the margin of the loss function

self.norm = 2  # the norm of the loss function

self.top = 10  # the test metric Hit@self.top

self.patience = 10  # change the learning rate and weight decay when the validation result is not getting better after self.patience epoches

self.earlyStopPatience = 5  # stop the training and output the learning results after changing the learning rate and weight decay self.earlyStopPatience times


## Data

`Training Data`

train2id.txt: the first line is the number of triples; in the following, each line is in the format of "head_id tail_id relation_id".

entity2id.txt: the first line is the number of entities; in the following, each line is in the format of "entity \t entity_id".

relation2id.txt: the first line is the number of relations; in the following, each line is in the format of "relation \t relation_id".

valid2id.txt: the first line is the number of validation triples; in the following, each line is in the format of "head_id tail_id relation_id".

Note: head_id and relation_id are consistent with entity_id and relation_id. For example, if there is a triple "head tail relation" for training, and we can find "head \t 0" and "tail \t 1" in entity2id.txt, and "relation \t 0" in relation2id.txt. Then, train2id.txt should contain a line "0 1 0".

`Test Data`

test2id.txt: the first line is the number of test triples; in the following, each line is in the format of "head_id tail_id relation_id".

`Output Data`

entity2vec.pickle: the pickle file which stores the embedding vectors of entities (refer to transE.entity_embeddings.weight.data in TransE.py).

relation2vec.pickle: the pickle file which stores the embedding vectors of relations (refer to transE.relation_embeddings.weight.data in TransE.py).

Note: the function preRead() implementated in Train.py can be used to read the pickle files.
