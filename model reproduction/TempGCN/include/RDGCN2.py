import math
import scipy
import scipy.spatial
from random import uniform, sample
import numpy as np
from include.utils import *
from include.Test2 import *
import json
import os
import torch
import time

#device = torch.device('cuda:1')

def rfunc(KG, e, r_num):
    head = {}
    tail = {}
    cnt = {}
    for i in range(r_num):
        head[i] = set()
        tail[i] = set()
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    #r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])

        r_mat_ind.append([tri[2], tri[0]])
        r_mat_val.append(tri[1])

    r_mat_ind = torch.Tensor(r_mat_ind).transpose(0, 1).long().cuda()
    r_mat_val = torch.Tensor(r_mat_val).long().cuda()
    r_mat = torch.sparse.FloatTensor(r_mat_ind, r_mat_val, torch.Size([e, e])).cuda()#RuntimeError: expected scalar type Long but found Float
    return head, tail, head_r, tail_r, r_mat

def get_mat(e, KG):
    du = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du

# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    #print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        M_arr[fir][sec] = 1.0
    #ind = np.asarray(ind).T
    ind = torch.tensor(ind).transpose(0, 1).cuda()
    val = torch.Tensor(val).cuda()
    M = torch.sparse.FloatTensor(ind, val, torch.Size([e, e])).cuda()
    return M, M_arr

class diag_layer(torch.nn.Module):
    def __init__(self, dimension):
        super(diag_layer, self).__init__()
        self.w0 = torch.nn.Parameter(torch.ones(1, dimension).cuda())
    def forward(self, inlayer, M, act_func):
        #print('adding a diag layer...')
        tosum = torch.sparse.mm(M, torch.mul(inlayer, self.w0))
        if act_func is None:
            return tosum
        else:
            return act_func(tosum)

class sparse_att_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1, kerner_size=1, stride=1):
        super(sparse_att_layer, self).__init__()
        self.conv1d=torch.nn.Conv1d(in_channels, out_channels, kerner_size, stride).cuda()
        torch.nn.init.xavier_uniform_(self.conv1d.weight)
    def forward(self, inlayer, dual_layer, r_mat, act_func):
        #print('adding sparse attention layer...')
        dual_transform = self.conv1d(dual_layer.transpose(0,1).unsqueeze(dim=0)).reshape(-1, 1)
        logits = dual_transform.index_select(0, r_mat._values()).reshape([-1])
        #lrelu = torch.sparse.FloatTensor(r_mat.indices, torch.nn.functional.leaky_relu(logits), r_mat.size())
        #coefs=sparse_softmax(lrelu)
        coefs = torch.sparse.FloatTensor(r_mat._indices(), torch.nn.functional.softmax(torch.nn.functional.leaky_relu(logits)), r_mat.size()).cuda()
        #vals = torch.spmm(coefs, inlayer) This API doesn't support gradient of sparse-tensor
        vals = torch.sparse.mm(coefs, inlayer)
        if act_func is None:
            return vals
        else:
            return act_func(vals)

class dual_att_layer(torch.nn.Module):
    def __init__(self, dimension, hid_dim, kernel_size=1, stride=1):
        super(dual_att_layer, self).__init__()
        self.conv1d = torch.nn.Conv1d(dimension, hid_dim, kernel_size, stride).cuda()
        self.conv1d1 = torch.nn.Conv1d(hid_dim, 1, kernel_size, stride).cuda()
        self.conv1d2 = torch.nn.Conv1d(hid_dim, 1, kernel_size, stride).cuda()
        torch.nn.init.xavier_uniform_(self.conv1d.weight)
        torch.nn.init.xavier_uniform_(self.conv1d1.weight)
        torch.nn.init.xavier_uniform_(self.conv1d2.weight)
    def forward(self, inlayer, inlayer2, adj_mat, act_func):#adj_mat numpy数组
        #print('adding dual attention layer...')
        in_fts = self.conv1d(inlayer2.transpose(0,1).unsqueeze(dim=0))
        f_1 = self.conv1d1(in_fts).transpose(1,2).reshape(-1, 1)#Tensorflow和pytorch一维卷积输入的第2，3个维度是相反的
        f_2 = self.conv1d2(in_fts).transpose(1,2).reshape(-1, 1)#
        logits = f_1 + f_2.transpose(0, 1)

        adj_tensor = torch.Tensor(adj_mat).cuda()#Constant
        bias_mat = torch.Tensor(-1e9 * (1.0 - (adj_mat > 0))).cuda()
        logoits = torch.mul(adj_tensor, logits)
        coefs = torch.nn.functional.softmax(torch.nn.functional.leaky_relu(logits) + bias_mat)

        vals = torch.mm(coefs, inlayer)
        if act_func is None:
            return vals
        else:
            return act_func(vals)

class self_att_layer(torch.nn.Module):
    def __init__(self, in_channels, hid_dim, kernel_size=1, stride=1):
        super(self_att_layer, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, hid_dim, kernel_size, stride, bias=False).cuda()#为什么不用bias？
        self.conv1d1 = torch.nn.Conv1d(hid_dim, 1, kernel_size, stride).cuda()
        self.conv1d2 = torch.nn.Conv1d(hid_dim, 1, kernel_size, stride).cuda()
        torch.nn.init.xavier_uniform_(self.conv1d.weight)
        torch.nn.init.xavier_uniform_(self.conv1d1.weight)
        torch.nn.init.xavier_uniform_(self.conv1d2.weight)
    def forward(self, inlayer, adj_mat, act_func):
        #print('adding self attention layer...')
        in_fts = self.conv1d(inlayer.transpose(0, 1).unsqueeze(dim=0))
        f_1 = self.conv1d1(in_fts).transpose(1, 2).reshape(-1, 1)  # Tensorflow和pytorch一维卷积输入的第2，3个维度是相反的
        f_2 = self.conv1d2(in_fts).transpose(1, 2).reshape(-1, 1)  #
        logits = f_1 +f_2.transpose(0, 1)

        adj_tensor = torch.Tensor(adj_mat).cuda()#Constant
        bias_mat = torch.Tensor(-1e9 * (1.0 - (adj_mat > 0))).cuda()
        logoits = torch.mul(adj_tensor, logits)
        coefs = torch.nn.functional.softmax(torch.nn.functional.leaky_relu(logits) + bias_mat)

        vals = torch.mm(coefs, inlayer)
        if act_func is None:
            return vals
        else:
            return act_func(vals)

class highway_layer(torch.nn.Module):
    def __init__(self, dimension):
        super(highway_layer, self).__init__()
        init_range = np.sqrt(6.0 / (dimension + dimension))#Xavier
        self.kernel_gate = torch.nn.init.uniform(torch.nn.Parameter(torch.Tensor(dimension, dimension).cuda()), a=-init_range, b=init_range)
        self.bias_gate = torch.nn.Parameter(torch.zeros(dimension).cuda())
    def forward(self, layer1, layer2):
        transform_gate = torch.mm(layer1, self.kernel_gate)+self.bias_gate
        transform_gate = torch.nn.functional.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        return transform_gate * layer2 + carry_gate * layer1

def compute_r(inlayer, head_r, tail_r):
    head_l = torch.Tensor(head_r).transpose(0,1).cuda()
    tail_l = torch.Tensor(tail_r).transpose(0,1).cuda()
    L = torch.mm(head_l, inlayer) / \
        (head_l.sum(axis=-1).unsqueeze(dim=-1)+0.000000001)#除法操作，可导致时间戳KG的0-shot关系表示为nan
    R = torch.mm(tail_l, inlayer) / \
        (tail_l.sum(axis=-1).unsqueeze(dim=-1)+0.000000001)
    r_embeddings = torch.cat((L, R), dim=-1)
    return r_embeddings


def get_input_embeddings(inpath, filename):
    with open(file=os.path.join(inpath, filename), mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    return embedding_list


class RDGCN_layer(torch.nn.Module):
    def __init__(self, dimension, hid_dim=600):#e,ILL,KG从外层模型获取
        super(RDGCN_layer, self).__init__()
        self.add_self_att_layer = self_att_layer(hid_dim, hid_dim)#channels of dual_X_1 is 600
        self.add_sparse_att_layer_1 = sparse_att_layer(hid_dim)
        self.add_sparse_att_layer_2 = sparse_att_layer(hid_dim)
        self.add_dual_att_layer = dual_att_layer(hid_dim, hid_dim)
        self.add_diag_layer_1 = diag_layer(dimension)
        self.add_diag_layer_2 = diag_layer(dimension)
        self.highway_1 = highway_layer(dimension)
        self.highway_2 = highway_layer(dimension)
        self.has_dual_A = False

    def get_input_layer(self, embedding_list):
        print('adding the primal input layer...')
        ent_embeddings = torch.Tensor(embedding_list).cuda()
        ent_embeddings.requires_grad = True
        self.ent_embeddings = torch.nn.Parameter(torch.nn.functional.normalize(ent_embeddings, 2, 1)).cuda()
        #return torch.nn.functional.normalize(ent_embeddings, 2, 1)
        return self.ent_embeddings

    def get_dual_input(self, inlayer, head, tail, head_r, tail_r):
        dual_X = compute_r(inlayer, head_r, tail_r)
        #print('computing the dual input...')
        count_r = len(head)
        if self.has_dual_A == False:
            print("compute dual A")
            self.has_dual_A = True
            self.dual_A = np.zeros((count_r, count_r))
            for i in range(count_r):
                for j in range(count_r):
                    h_den = len(head[i] | head[j])
                    if h_den > 0:
                        a_h = len(head[i] & head[j]) / h_den
                    else:
                        a_h = 0
                    t_den = len(tail[i] | tail[j])
                    if t_den > 0:
                        a_t = len(tail[i] & tail[j]) / t_den
                    else:
                        a_t = 0
                    self.dual_A[i][j] = a_h + a_t
        return dual_X, self.dual_A

    def forward(self, primal_X_0, e, KG, head, tail, head_r, tail_r, r_mat, M, M_arr, act_func, alpha, beta, gamma, k):
        #print('first interaction...')
        dual_X_1, dual_A_1 = self.get_dual_input(primal_X_0, head, tail, head_r, tail_r)
        dual_H_1 = self.add_self_att_layer(dual_X_1, dual_A_1, torch.nn.functional.relu)
        primal_H_1 =self.add_sparse_att_layer_1(primal_X_0, dual_H_1, r_mat, torch.nn.functional.relu)
        primal_X_1 = primal_X_0 + alpha * primal_H_1


        #print('second interaction...')
        dual_X_2, dual_A_2 = self.get_dual_input(primal_X_1, head, tail, head_r, tail_r)
        dual_H_2 = self.add_dual_att_layer(dual_H_1, dual_X_2, dual_A_2, torch.nn.functional.relu)
        primal_H_2 = self.add_sparse_att_layer_2(primal_X_1, dual_H_2, r_mat, torch.nn.functional.relu)
        primal_X_2 = primal_X_0 + beta * primal_H_2


        #print('gcn layers...')
        gcn_layer_1 = self.add_diag_layer_1(primal_X_2, M, act_func)
        gcn_layer_1 = self.highway_1(primal_X_2, gcn_layer_1)
        gcn_layer_2 = self.add_diag_layer_2(gcn_layer_1, M, act_func)
        output = self.highway_2(gcn_layer_1, gcn_layer_2)


        dual_X_3, dual_A_3 = self.get_dual_input(output, head, tail, head_r, tail_r)
        dual_H_3 = self.add_dual_att_layer(dual_H_2, dual_X_3, dual_A_3, torch.nn.functional.relu)

        return output, dual_H_3

    #
    def get_loss(self, output_layer, dual_X, KG, negs, num, k, gamma):
        KG_numpy = np.array(KG)
        t = len(KG)
        head_index = torch.Tensor(KG_numpy[:, 0]).long().cuda()
        rel_index = torch.Tensor(KG_numpy[:, 1]).long().cuda()
        tail_index = torch.Tensor(KG_numpy[:, 2]).long().cuda()
        head_emb = output_layer.index_select(0, head_index)
        rel_emb = dual_X.index_select(0, rel_index)
        tail_emb = output_layer.index_select(0, tail_index)
        scores_pos = distance_func(head_emb, tail_emb, rel_emb)#

        negs_numpy = np.array(negs)
        head_index = torch.Tensor(negs_numpy[:, 0]).long().cuda()
        rel_index = torch.Tensor(negs_numpy[:, 1]).long().cuda()
        tail_index = torch.Tensor(negs_numpy[:, 2]).long().cuda()
        head_emb = output_layer.index_select(0, head_index)
        rel_emb = dual_X.index_select(0, rel_index)
        tail_emb = output_layer.index_select(0, tail_index)
        scores_neg = distance_func(head_emb, tail_emb, rel_emb)#

        C = -scores_neg.reshape([t, num])
        D = scores_pos + gamma
        L1 = torch.nn.functional.relu(C.add(torch.reshape(D, [t, 1])))
        loss = L1.sum()/(num * t)
        return loss

        '''
        L = torch.ones((t, num)).cuda() * (scores_pos.reshape(t, 1))
        L = L.reshape((t * num,))

        L - scores_neg + gamma
        '''


    def get_all_pos_entities(self, KG):
        self.pos_entities=set()
        for triplet in KG:
            head = triplet[0]
            tail = triplet[2]
            self.pos_entities.add(head)
            self.pos_entities.add(tail)

    # get negative samples
    # pos_entities is a list that stores the set of entities occurred in triples
    def get_k_nearst_neigh(self, output_layer, k):
        neg = []
        t = len(self.pos_entities)
        KG_vec = output_layer.cpu().detach().numpy()
        pos_vec = np.array([KG_vec[e1] for e1 in self.pos_entities])
        sim = scipy.spatial.distance.cdist(pos_vec, KG_vec, metric='cityblock')
        rank_k = sim.argsort(1)[:,:k]
        self.rank_k_dict={}
        for i, e in enumerate(self.pos_entities):
            self.rank_k_dict[e]=rank_k[i].tolist()
        #return self.rank_k_dict

    #get all neg triplets
    def get_negs(self, KG, output_layer, num, k, e_num):
        #self.get_k_nearst_neigh(output_layer, k)
        negs=[]
        for tri in KG:
            tri_negs = self.get_tri_negs(tri, num, e_num)
            negs.extend(tri_negs)
        return negs


    #get num negetive triplets of a pos triplet
    def get_tri_negs(self, triplet, num, e_num):
        tri_negs = []
        for i in range(num):
            while True:
                corrupted_triplet = self.getCorruptedTriplet(triplet, e_num)
                if corrupted_triplet not in self.SKG:
                    tri_negs.append(corrupted_triplet)
                    break
        return tri_negs

    #get a negetive triplet
    def getCorruptedTriplet(self, triplet, e_num):
        head = triplet[0]
        tail = triplet[2]
        i = uniform(-1, 1)
        #print(type(self.rank_k_dict[head]))
        #assert False
        if i < 0:  # 小于0，打坏三元组的头实体
            while True:
                entityTemp = sample(range(e_num), 1)[0]
                if entityTemp != head:
                    break
            corruptedTriplet = (entityTemp, triplet[1], tail)
        else:  # 大于等于0，打坏三元组的尾实体
            while True:
                entityTemp = sample(range(e_num), 1)[0]
                if entityTemp != tail:
                    break
            corruptedTriplet = (head, triplet[1], entityTemp)
        return corruptedTriplet


def training(model, primal_X_0, act_func, alpha, beta, gamma, num, k, e, r_num, KG, SKG, learning_rate, epochs, valid_KG):
    model.get_all_pos_entities(KG)
    model.SKG = SKG
    J = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':model.ent_embeddings, 'lr':0.1*learning_rate}], lr=learning_rate)
    M, M_arr = get_sparse_tensor(e, KG)
    head, tail, head_r, tail_r, r_mat = rfunc(KG, e, r_num)
    best_valid_mrr=0
    best_valid_epoch=0
    for i in range(epochs):
        #primal_X_0 = get_input_layer(embedding_list, dimension)
        output, dual_X = model(primal_X_0, e, KG, head, tail, head_r, tail_r, r_mat, M, M_arr, act_func, alpha, beta,
                               gamma, k)
        if i % 5 == 0:
            #output, dual_X = model(primal_X_0, e,  KG, head, tail, head_r, tail_r, r_mat, M, M_arr, act_func, alpha, beta,
            #               gamma, k)
            negs = model.get_negs(KG, output, num, k, e)
            valid_mrr = fast_validate(output, dual_X, valid_KG, KG)
            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                best_valid_epoch=i

        loss = model.get_loss(output, dual_X, KG, negs, num, k, gamma )
        #print("loss: ", loss)
        loss.backward(retain_graph=True)
        #print(model.ent_embeddings.grad.norm())
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            #output = model(primal_X_0, e, ILL, KG, head, tail, head_r, tail_r, r_mat, M, M_arr, act_func, alpha, beta,
            #               gamma, k, lang)
            #loss = model.get_loss(output, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
            #if i >= 30 and loss.item() > J[-1] and J[-1] > J[-2]:
            #    break
            J.append(loss.item())
            print('%d/%d' % (i + 1, epochs), 'epochs...loss:', loss.item(), "current valid mrr: ", valid_mrr, "best valid mrr: ", best_valid_mrr, "best valid epoch: ", best_valid_epoch)
        else:
            print('%d/%d' % (i + 1, epochs), 'epochs...loss:', loss.item())
    output, dual_X = model(primal_X_0, e, KG, head, tail, head_r, tail_r, r_mat, M, M_arr, act_func, alpha, beta,
                           gamma, k)
    #model.has_dual_A = False
    return output, dual_X, J


