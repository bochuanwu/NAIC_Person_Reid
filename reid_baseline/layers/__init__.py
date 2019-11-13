# encoding: utf-8
"""
@author:  bochuan Wu
@contact: 1670306646@qq.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .focalloss import FocalLoss
from .reanked_loss import RankedLoss, CrossEntropyLabelSmooth
from .reanked_clu_loss import CRankedLoss
from .OSMLoss import OSM_CAA_Loss

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    criterion_osm_caa = OSM_CAA_Loss()
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
        ranked_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK, cfg.SOLVER.ALPHA, cfg.SOLVER.TVAL)  # ranked_loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
        cranked_loss = CRankedLoss(cfg.SOLVER.MARGIN_RANK, cfg.SOLVER.ALPHA, cfg.SOLVER.TVAL)  # cranked_loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet,ranked_loss,cranked_loss'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    '''
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        print("label smooth on, numclasses:", num_classes)
    '''
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = FocalLoss(gamma=0.5)  # new add by luo
        print("focal label on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'ranked_loss':
        def loss_func(score, feat, target):
            return ranked_loss(feat, target)
    elif cfg.DATALOADER.SAMPLER == 'cranked_loss':
        def loss_func(score, feat, target):
            return cranked_loss(feat, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'softmax_rank':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cfg.SOLVER.WEIGHT * ranked_loss(feat, target)  # new add by zzg, open label smooth
                else:
                    return F.cross_entropy(score, target) + ranked_loss(feat, target)  # new add by zzg, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cfg.SOLVER.WEIGHT * cranked_loss(feat, target)[0]  # new add by zzg, open label smooth
                else:
                    return F.cross_entropy(score, target) + cranked_loss(feat, target)[0]  # new add by zzg, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes,model):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048
    criterion_osm_caa = OSM_CAA_Loss()
    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
        ranked_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK, cfg.SOLVER.ALPHA, cfg.SOLVER.TVAL)  # ranked_loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
        cranked_loss = CRankedLoss(cfg.SOLVER.MARGIN_RANK, cfg.SOLVER.ALPHA, cfg.SOLVER.TVAL)  # cranked_loss
    else:
        print('expected METRIC_LOSS_TYPE should be center,triplet_center,ranked_loss,cranked_loss'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        #xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        xent = FocalLoss(gamma=0.5)      # new add by wu
        print("focal label on, numclasses:", num_classes)
        #print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        criterion_osm_caa(feat, target,model.classifier.weight.t())
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion
