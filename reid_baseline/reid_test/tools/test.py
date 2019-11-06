# encoding: utf-8
"""
@author:  wubochuan
@contact: 1670306646@qq.com
"""

import argparse
import os
import sys
from os import mkdir

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
re_rank = False
def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
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
    #print(test_ids)
    #print(len(test_ids))
    testid = []
    for tid in test_ids:
        tid=list(tid)
        testid = testid + tid
    print(query.shape)
    #print(test.shape)
    #print(len(test))
    if re_rank:
        q_g_dist = np.dot(query, np.transpose(test))
        q_q_dist = np.dot(query, np.transpose(query))
        g_g_dist = np.dot(test, np.transpose(test))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    else:
        dist = cosine_similarity(query, test)
    rank = get_result(dist, queryid, testid)

if __name__ == '__main__':
    main()
