import numpy as np
import scipy
import torch
from include.utils import *

def evaluate_relation(out, dual, KG, all_KG, numOfRelation):
    output = out.cpu()
    dual_H = dual.cpu()
    numOfEntity = len(output)
    numOfTestTriple = len(KG)
    AllTriples = torch.Tensor(all_KG).int()
    ranks_raw = []
    hits_1_raw = []
    hits_3_raw = []
    hits_10_raw = []
    ranks_filter = []
    hits_1_filter = []
    hits_3_filter = []
    hits_10_filter = []
    i = 0
    for test_triplet in KG:
        i += 1
        testHead = test_triplet[0]
        testRelation = test_triplet[1]
        testTail = test_triplet[2]
        testHeadEmbedding = output[testHead]
        testRelationEmbedding = dual_H[testRelation]
        testTailEmbedding = output[testTail]
        #targetLoss = torch.abs(torch.cat((testHeadEmbedding, testTailEmbedding)) - testRelationEmbedding).sum().repeat(
        #    numOfRelation, 1)  # 测试三元组的得分
        targetLoss = distance_func(testHeadEmbedding, testTailEmbedding, testRelationEmbedding).repeat(
            numOfRelation, 1)  # 测试三元组的得分
        tmpHeadEmbedding = testHeadEmbedding.repeat(numOfRelation, 1)
        tmpRelationEmbedding = testRelationEmbedding.repeat(numOfRelation, 1)
        tmpTailEmbedding = testTailEmbedding.repeat(numOfRelation, 1)


        tmpRelationLoss = distance_func(tmpHeadEmbedding, tmpTailEmbedding, dual_H).view(-1,1)

        unCorrR = torch.nonzero(torch.nn.functional.relu(targetLoss - tmpRelationLoss))[:, 0]

        numOfWrongRelation = unCorrR.size()[0]

        numOfFilterRelation = 0

        for wrongRelation in unCorrR:
            if AllTriples[(AllTriples[:, 0] == testHead) & (AllTriples[:, 1] == wrongRelation) & (
                    AllTriples[:, 2] == testTail)].size()[0]:
                numOfFilterRelation += 1

        ranks_raw.append(numOfWrongRelation + 1)
        hits_1_raw.append(numOfWrongRelation + 1 <= 1)
        hits_3_raw.append(numOfWrongRelation + 1 <= 3)
        hits_10_raw.append(numOfWrongRelation + 1 <= 10)

        ranks_filter.append(numOfWrongRelation + 1 - numOfFilterRelation)
        hits_1_filter.append(numOfWrongRelation + 1 - numOfFilterRelation <= 1)
        hits_3_filter.append(numOfWrongRelation + 1 - numOfFilterRelation <= 3)
        hits_10_filter.append(numOfWrongRelation + 1 - numOfFilterRelation <= 10)

    hits_1_prop_raw = sum(hits_1_raw) / len(hits_1_raw)
    hits_3_prop_raw = sum(hits_3_raw) / len(hits_3_raw)
    hits_10_prop_raw = sum(hits_10_raw) / len(hits_10_raw)
    MRR_raw = np.mean([1 / a for a in ranks_raw])
    MR_raw = np.mean(ranks_raw)

    hits_1_prop_filter = sum(hits_1_filter) / len(hits_1_filter)
    hits_3_prop_filter = sum(hits_3_filter) / len(hits_3_filter)
    hits_10_prop_filter = sum(hits_10_filter) / len(hits_10_filter)
    MRR_filter = np.mean([1 / a for a in ranks_filter])
    MR_filter = np.mean(ranks_filter)

    print("hits_1_raw: %f, hits_3_raw: %f, hits_10_raw: %f, MR raw: %f MRR_raw: %f" % (
        hits_1_prop_raw, hits_3_prop_raw, hits_10_prop_raw, MR_raw, MRR_raw))
    print("hits_1_filter: %f, hits_3_filter: %f, hits_10_filter: %f, MR_filter:%f MRR_filter: %f" % (
        hits_1_prop_filter, hits_3_prop_filter, hits_10_prop_filter, MR_filter, MRR_filter))
    return [hits_1_prop_raw, hits_1_prop_filter, MR_filter, MRR_filter]


def evaluate_entity(out, dual, KG, all_KG):
    output = out.cpu()
    dual_H = dual.cpu()
    numOfEntity = len(output)
    numOfTestTriple = len(KG)
    AllTriples = torch.Tensor(all_KG).int()
    head_ranks_raw = []
    tail_ranks_raw = []
    head_hits_1_raw = []
    tail_hits_1_raw = []
    head_hits_3_raw = []
    tail_hits_3_raw = []
    head_hits_10_raw = []
    tail_hits_10_raw = []
    head_ranks_filter = []
    tail_ranks_filter = []
    head_hits_1_filter = []
    tail_hits_1_filter = []
    head_hits_3_filter = []
    tail_hits_3_filter = []
    head_hits_10_filter = []
    tail_hits_10_filter = []
    i=0
    for test_triplet in KG:
        i+=1
        #print("Test %d triplets" %(i))
        testHead = test_triplet[0]
        testRelation = test_triplet[1]
        testTail = test_triplet[2]
        testHeadEmbedding = output[testHead]
        testRelationEmbedding = dual_H[testRelation]
        testTailEmbedding = output[testTail]
        #targetLoss = torch.abs(torch.cat((testHeadEmbedding, testTailEmbedding)) - testRelationEmbedding).sum().repeat(numOfEntity, 1)  # 测试三元组的得分
        targetLoss = distance_func(testHeadEmbedding, testTailEmbedding, testRelationEmbedding).repeat(
            numOfEntity, 1)  # 测试三元组的得分
        tmpHeadEmbedding = testHeadEmbedding.repeat(numOfEntity, 1)
        tmpRelationEmbedding = testRelationEmbedding.repeat(numOfEntity, 1)
        tmpTailEmbedding = testTailEmbedding.repeat(numOfEntity, 1)

        #tmpHeadLoss = torch.abs(torch.cat((output, tmpTailEmbedding), 1) - tmpRelationEmbedding).sum(axis=1).view(-1,1)
        #tmpTailLoss = torch.abs(torch.cat((tmpHeadEmbedding, output), 1) - tmpRelationEmbedding).sum(axis=1).view(-1,1)
        tmpHeadLoss = distance_func(output, tmpTailEmbedding, tmpRelationEmbedding).view(-1, 1)
        tmpTailLoss = distance_func(tmpHeadEmbedding, output, tmpRelationEmbedding).view(-1, 1)

        unCorrH = torch.nonzero(torch.nn.functional.relu(targetLoss - tmpHeadLoss))[:, 0]
        unCorrT = torch.nonzero(torch.nn.functional.relu(targetLoss - tmpTailLoss))[:, 0]

        numOfWrongHead = unCorrH.size()[0]
        numOfWrongTail = unCorrT.size()[0]

        numOfFilterHead = 0
        numOfFilterTail = 0

        for wrongHead in unCorrH:
            if AllTriples[(AllTriples[:, 0] == wrongHead) & (AllTriples[:, 1] == testRelation) & (
                    AllTriples[:, 2] == testTail)].size()[0]:
                numOfFilterHead += 1
        for wrongTail in unCorrT:
            if AllTriples[(AllTriples[:, 0] == testHead) & (AllTriples[:, 1] == testRelation) & (
                    AllTriples[:, 2] == wrongTail)].size()[0]:
                numOfFilterTail += 1

        head_ranks_raw.append(numOfWrongHead + 1)
        tail_ranks_raw.append(numOfWrongTail + 1)
        head_hits_1_raw.append(numOfWrongHead + 1 <= 1)
        tail_hits_1_raw.append(numOfWrongTail + 1 <= 1)
        head_hits_3_raw.append(numOfWrongHead + 1 <= 3)
        tail_hits_3_raw.append(numOfWrongTail + 1 <= 3)
        head_hits_10_raw.append(numOfWrongHead + 1 <= 10)
        tail_hits_10_raw.append(numOfWrongTail + 1 <= 10)

        head_ranks_filter.append(numOfWrongHead + 1 - numOfFilterHead)
        tail_ranks_filter.append(numOfWrongTail + 1 - numOfFilterTail)
        head_hits_1_filter.append(numOfWrongHead + 1 - numOfFilterHead <= 1)
        tail_hits_1_filter.append(numOfWrongTail + 1 - numOfFilterTail <= 1)
        head_hits_3_filter.append(numOfWrongHead + 1 - numOfFilterHead <= 3)
        tail_hits_3_filter.append(numOfWrongTail + 1 - numOfFilterTail <= 3)
        head_hits_10_filter.append(numOfWrongHead + 1 - numOfFilterHead <= 10)
        tail_hits_10_filter.append(numOfWrongTail + - numOfFilterTail <= 10)


    #hits_1_prop_raw = sum(hits_1_raw) / len(hits_1_raw)
    #hits_3_prop_raw = sum(hits_3_raw) / len(hits_3_raw)
    #hits_10_prop_raw = sum(hits_10_raw) / len(hits_10_raw)
    #MRR_raw = np.mean([1 / a for a in ranks_raw])
    #MR_raw = np.mean(ranks_raw)

    head_hits_1_prop_filter = sum(head_hits_1_filter) / len(head_hits_1_filter)
    head_hits_3_prop_filter = sum(head_hits_3_filter) / len(head_hits_3_filter)
    head_hits_10_prop_filter = sum(head_hits_10_filter) / len(head_hits_10_filter)
    head_MRR_filter = np.mean([1 / a for a in head_ranks_filter])
    head_MR_filter = np.mean(head_ranks_filter)

    tail_hits_1_prop_filter = sum(tail_hits_1_filter) / len(tail_hits_1_filter)
    tail_hits_3_prop_filter = sum(tail_hits_3_filter) / len(tail_hits_3_filter)
    tail_hits_10_prop_filter = sum(tail_hits_10_filter) / len(tail_hits_10_filter)
    tail_MRR_filter = np.mean([1 / a for a in tail_ranks_filter])
    tail_MR_filter = np.mean(tail_ranks_filter)

    #print("hits_1_raw: %f, hits_3_raw: %f, hits_10_raw: %f, MR raw: %f MRR_raw: %f" % (
    #hits_1_prop_raw, hits_3_prop_raw, hits_10_prop_raw, MR_raw, MRR_raw))
    print("head_hits_1_filter: %f, head_hits_3_filter: %f, head_hits_10_filter: %f, head_MR_filter:%f head_MRR_filter: %f" % (
    head_hits_1_prop_filter, head_hits_3_prop_filter, head_hits_10_prop_filter, head_MR_filter, head_MRR_filter))
    print("tail_hits_1_filter: %f, tail_hits_3_filter: %f, tail_hits_10_filter: %f, tail_MR_filter:%f tail_MRR_filter: %f" % (
    tail_hits_1_prop_filter, tail_hits_3_prop_filter, tail_hits_10_prop_filter, tail_MR_filter, tail_MRR_filter))
    return [head_hits_1_prop_filter, tail_hits_1_prop_filter, head_hits_3_prop_filter, tail_hits_3_prop_filter, head_hits_10_prop_filter, tail_hits_10_prop_filter, head_MR_filter, tail_MR_filter, head_MRR_filter, tail_MRR_filter]

def fast_validate(out, dual, valid_KG, all_KG):
    numOfEntity = len(out)
    rank_reciprocal=[]
    for valid_triple in valid_KG:
        validHeadEmbedding = out[valid_triple[0]]
        validRelationEmbedding = dual[valid_triple[1]]
        validTailEmbedding = out[valid_triple[2]]

        targetLoss = distance_func(validHeadEmbedding, validTailEmbedding, validRelationEmbedding).repeat(
            numOfEntity, 1)  # 测试三元组的得分
        tmpHeadEmbedding = validHeadEmbedding.repeat(numOfEntity, 1)
        tmpRelationEmbedding = validRelationEmbedding.repeat(numOfEntity, 1)
        tmpTailEmbedding = validTailEmbedding.repeat(numOfEntity, 1)

        tmpHeadLoss = distance_func(out, tmpTailEmbedding, tmpRelationEmbedding).view(-1, 1)
        tmpTailLoss = distance_func(tmpHeadEmbedding, out, tmpRelationEmbedding).view(-1, 1)

        unCorrH = torch.nonzero(torch.nn.functional.relu(targetLoss - tmpHeadLoss))[:, 0]
        unCorrT = torch.nonzero(torch.nn.functional.relu(targetLoss - tmpTailLoss))[:, 0]

        numOfWrongHead = unCorrH.size()[0]
        numOfWrongTail = unCorrT.size()[0]

        rank_reciprocal.append(2/(numOfWrongTail+numOfWrongHead+2))
    return np.mean(rank_reciprocal)




