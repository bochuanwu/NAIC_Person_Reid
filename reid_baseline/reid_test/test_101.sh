python3 tools/test.py --config_file='configs/softmax_triplet_with_center_ibn.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('./dataset')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./market1501/resnet152_with-center-allgcn/resnet152_model_400.pth')"

