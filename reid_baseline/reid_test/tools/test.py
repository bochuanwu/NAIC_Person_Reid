# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
from sklearn.preprocessing import normalize
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
import numpy as np
import torch
import json
from scipy.spatial.distance import cdist

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
re_rank = True

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def euclidean_distance(qf,gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
        torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    dist_mat.addmm_(1,-2,qf,gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf,gf):
    epsilon = 0.00001
    qf = torch.from_numpy(qf)
    gf = torch.from_numpy(gf)
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True) #mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True) #nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1/qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1+epsilon,1-epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def get_result(distmat, query_ids=None, gallery_ids=None):
    m, n = distmat.shape
    print(m,n)
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    print(len(query_ids))
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    print('indices:',indices.shape)
    #print(gallery_ids.shape)
    dic = {}
    for i,ids in enumerate(query_ids):
        #dic[str(id)]= gallery_ids[indices[i]<200].tolist()
        dic[str(ids)]= gallery_ids[indices[i]][:200].tolist()
    jsonData = json.dumps(dic)
    fileObject = open('./baseline.json', 'w')
    fileObject.write(jsonData)
    fileObject.close()
    return indices


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    query_dataset, test_dataset, query_loader, test_loader = make_data_loader(cfg)
    model = build_model(cfg, 4768)

    model.load_param(cfg.TEST.WEIGHT)

    model.to(DEVICE)
    model.eval()


    with torch.no_grad():
        query = np.concatenate([model(img.to(DEVICE)).detach().cpu().numpy()
                                for img, pid, camid in query_loader])
        test = np.concatenate([model(img.to(DEVICE)).detach().cpu().numpy()
                               for img, pid, camid in test_loader])

    query_ids = [pid1 for _, pid1, _ in query_loader]
    queryid = []
    for qid in query_ids:
        qid = list(qid)
        queryid=queryid + qid
    test_ids = [pid2 for _, pid2, _ in test_loader]

    testid = []
    for tid in test_ids:
        tid=list(tid)
        testid = testid + tid
    feat_norm = cfg.TEST.FEAT_NORM
    if feat_norm == 'yes':
        print("The test feature is normalized")
        feat = np.concatenate((query,test))
        feat = torch.from_numpy(feat)
        feat = torch.nn.functional.normalize(feat, dim=1, p=2).numpy()  # along channel
        query = feat[:3147]
        test = feat[3147:]

    print(len(query))
    if re_rank:
        query = torch.from_numpy(query)
        test = torch.from_numpy(test)
        dist = re_ranking(query, test, k1=8, k2=3, lambda_value=0.6)
    else:
        #dist = cdist(query,test,metric='mahalanobis')
        dist = cosine_similarity(query, test)
    #dist = normalize(dist, axis=1, norm='l2')
    #jsonData = json.dumps(dist.tolist())
    #fileObject = open('./distresult1.json', 'w')
    #fileObject.write(jsonData)
    #fileObject.close()
    rank = get_result(dist, queryid, testid)

if __name__ == '__main__':
    main()
